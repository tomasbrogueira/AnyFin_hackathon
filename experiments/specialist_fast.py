"""Fit and export an Anyfin specialist, or predict with standard-library inference."""

import argparse
import csv
import gzip
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from anyfin.ledger_pairs import operational_checking
from anyfin.paths import ROOT, TRAINING_CSV, VALIDATION_CSV
from anyfin.transactions import load_rows, read_fallback_predictions, without_labels, write_predictions
from experiments.independent_frontier import combined_predictions
from experiments.specialist_features import features


UNKNOWN = "unknown"


def character_features(text, maximum):
    for word in text.split():
        padded = f" {word} "
        for width in range(2, min(maximum, len(padded)) + 1):
            for i in range(len(padded) - width + 1):
                yield padded[i:i + width]


def word_features(text):
    words = text.split()
    yield from words
    yield from (f"{left} {right}" for left, right in zip(words, words[1:]))


def add_weights(scores, weights, scale):
    for i, weight in enumerate(weights):
        scores[i] += weight * scale


def add_text(scores, vocabulary, terms, sublinear):
    counts = Counter(term for term in terms if term in vocabulary)
    values = []
    for term, count in counts.items():
        idf, weights = vocabulary[term]
        magnitude = idf * (1 + math.log(count) if sublinear else count)
        values.append((magnitude, weights))
    if values:
        normalizer = 1 / math.sqrt(sum(value * value for value, _ in values))
        for value, weights in values:
            add_weights(scores, weights, value * normalizer)


class Specialist:
    def __init__(self, data):
        self.classes = data["classes"]
        self.training_customers = frozenset(data["training_customers"])
        self.intercepts = data["intercepts"]
        self.minimum_score, self.minimum_margin = data["thresholds"]
        self.body, self.words, self.raw, self.metadata = data["features"]
        self.predict_features = lru_cache(maxsize=8192)(self._predict_features)

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return cls(json.load(stream))

    def predict(self, row):
        body, raw, metadata = features(row)
        return self.predict_features(body, raw, tuple(metadata.items()))

    def _predict_features(self, body, raw, metadata):
        scores = self.intercepts.copy()
        add_text(scores, self.body, character_features(body, 5), sublinear=True)
        add_text(scores, self.words, word_features(body), sublinear=False)
        add_text(scores, self.raw, character_features(raw, 4), sublinear=False)
        for key, value in metadata:
            weights = self.metadata.get(f"{key}={value}")
            if weights is not None:
                add_weights(scores, weights, 1.0)
        best, runner_up, *_ = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        if scores[best] < self.minimum_score or scores[best] - scores[runner_up] < self.minimum_margin:
            return UNKNOWN
        return self.classes[best]


def export_model(classifier, thresholds, output, customers):
    columns = classifier.model.coef_.T
    vocabulary = []
    offset = 0
    for vector, scale in ((classifier.body_vector, 1.0), (classifier.word_vector, 0.85),
                          (classifier.raw_vector, 0.40)):
        group = {}
        for term, index in vector.vocabulary_.items():
            group[term] = [float(vector.idf_[index]), (scale * columns[offset + index]).tolist()]
        vocabulary.append(group)
        offset += len(group)
    metadata = {term: (0.22 * columns[offset + index]).tolist()
                for term, index in classifier.meta_vector.vocabulary_.items()}
    export = {"classes": classifier.model.classes_.tolist(), "intercepts": classifier.model.intercept_.tolist(),
              "thresholds": thresholds, "features": [*vocabulary, metadata],
              "training_customers": sorted(customers)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=3) as stream:
        json.dump(export, stream, separators=(",", ":"), allow_nan=False)


def fit_export(ledger, opus, seed, output, holdout, render_formats):
    from threadpoolctl import threadpool_limits

    from experiments.specialist_seed_selection import matched_seed_rows
    from experiments.specialist_quantity import expand
    from experiments.specialist_sparse import SparseSpecialist, decisions, select
    from experiments.specialist_templates import synthetic_rows

    with opus.open(encoding="utf-8") as stream:
        teacher = json.load(stream)
    matched = matched_seed_rows(synthetic_rows(family="bank"), teacher, seed)
    examples, _ = expand(matched, "format" if render_formats else "base")
    development = [row for row in ledger if row["customer"] in {"c1", "c2"}]
    training = [row for row in ledger if row["customer"] != holdout]
    validation = without_labels(ledger)
    with threadpool_limits(limits=1):
        regularization, thresholds, _ = select(development, "weak", examples)
        classifier = SparseSpecialist(training, "weak", examples).fit(regularization)
        expected = decisions(classifier.ranked(classifier.transform(validation)), thresholds)
    export_model(classifier, thresholds, output, {row["customer"] for row in training})
    specialist = Specialist.load(output)
    actual = [specialist.predict(row) for row in validation]
    if actual != expected:
        raise ValueError("Exported predictions differ from the fitted classifier on its parity check.")
    print(f"Exported {len(examples)} generated examples, C={regularization}, thresholds={thresholds}. "
          f"Verified {len(actual)} predictions against scikit-learn.")


def predict(rows, ledger, specialist, account_activity=False, recurring_income=False):
    inputs = without_labels(rows)
    training = [row for row in ledger if row["customer"] in specialist.training_customers]
    first = combined_predictions(inputs, training)
    if account_activity or recurring_income:
        first = operational_checking(inputs, first)
    if recurring_income:
        from experiments.evaluate_income_context import recurring_pay

        first = recurring_pay(inputs, first, training)
    return [specialist.predict(row) if initial == UNKNOWN else initial
            for row, initial in zip(inputs, first, strict=True)]


def benchmark(model, runs, account_activity=False, recurring_income=False):
    source = load_rows(VALIDATION_CSV)
    count = 10_000
    measurements = {}
    with tempfile.TemporaryDirectory() as directory:
        input_csv, output_csv = (Path(directory) / filename for filename in ("input.csv", "predictions.csv"))
        for name in ("repeated", "numeric_references", "alphabetic_suffixes"):
            rows = []
            for i in range(count):
                copy, offset = divmod(i, len(source))
                row = dict(source[offset], id=f"timing-{i}")
                for field in ("customer", "account"):
                    row[field] = f"{copy}-{row[field]}"
                if name == "numeric_references":
                    row["text"] += f" REF{i:08d}"
                elif name == "alphabetic_suffixes":
                    suffix = "".join(chr(65 + i // (26 ** exponent) % 26) for exponent in (2, 1, 0))
                    row["text"] += f" REF{suffix}"
                rows.append(row)
            with input_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            elapsed = []
            for _ in range(runs):
                start = time.perf_counter()
                command = [sys.executable, "-m", "experiments.specialist_fast", "predict", str(input_csv),
                           "--model", str(model.resolve()), "-o", str(output_csv)]
                if account_activity:
                    command.append("--account-activity")
                if recurring_income:
                    command.append("--recurring-income")
                subprocess.run(command, cwd=ROOT, check=True)
                elapsed.append(time.perf_counter() - start)
                read_fallback_predictions(output_csv, rows)
                with output_csv.open(encoding="utf-8", newline="") as stream:
                    if [row["id"] for row in csv.DictReader(stream)] != [row["id"] for row in rows]:
                        raise ValueError("Cold benchmark did not preserve transaction order.")
            median = statistics.median(elapsed)
            measurements[name] = {"seconds": elapsed, "median_seconds": median,
                                  "rows_per_second": count / median,
                                  "completed_within_one_second": sum(value <= 1 for value in elapsed)}
    return {"rows": count, "fit_excluded": True, "model_bytes": model.stat().st_size, "timings": measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Train and export with existing generated examples.")
    fit.add_argument("--opus", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=23)
    fit.add_argument("--render-formats", action="store_true", help="Add controlled bank-format variants.")
    fit.add_argument("--holdout", choices=("c3",), help="Exclude c3 for a customer-holdout model.")
    fit.add_argument("--model", type=Path, required=True)
    classify = commands.add_parser("predict", help="Classify locally using an exported model.")
    classify.add_argument("input", type=Path)
    classify.add_argument("--model", type=Path, required=True)
    classify.add_argument("-o", "--output", type=Path)
    classify.add_argument("--account-activity", action="store_true",
                          help="Resolve transfers using account activity before invoking the specialist.")
    classify.add_argument("--recurring-income", action="store_true",
                          help="Apply account activity and recurring payroll before the specialist.")
    timing = commands.add_parser("benchmark", help="Time cold CSV-to-CSV inference on 10,000 inputs.")
    timing.add_argument("--model", type=Path, required=True)
    timing.add_argument("--runs", type=int, default=5)
    timing.add_argument("-o", "--output", type=Path)
    timing.add_argument("--account-activity", action="store_true")
    timing.add_argument("--recurring-income", action="store_true")
    args = parser.parse_args()
    if args.command == "benchmark":
        if args.runs < 1 or (args.output and args.output.resolve() in
                            {args.model.resolve(), VALIDATION_CSV.resolve(), TRAINING_CSV.resolve()}):
            parser.error("Choose positive repetitions and an output path distinct from the inputs.")
        text = json.dumps(benchmark(args.model, args.runs, args.account_activity, args.recurring_income), indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return
    ledger = load_rows(TRAINING_CSV)
    if args.command == "fit":
        if args.model.resolve() in {TRAINING_CSV.resolve(), args.opus.resolve()}:
            parser.error("Model output overlaps an input.")
        fit_export(ledger, args.opus, args.seed, args.model, args.holdout, args.render_formats)
        return
    protected = {TRAINING_CSV.resolve(), args.model.resolve(), args.input.resolve()}
    if args.output and args.output.resolve() in protected:
        parser.error("Output path overlaps an input.")
    rows = load_rows(args.input)
    specialist = Specialist.load(args.model)
    if {row["id"] for row in ledger if row["customer"] in specialist.training_customers} & {row["id"] for row in rows}:
        parser.error("Training and prediction IDs overlap.")
    write_predictions(rows, predict(rows, ledger, specialist, args.account_activity, args.recurring_income), args.output)


if __name__ == "__main__":
    main()
