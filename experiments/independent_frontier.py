"""Compare Anyfin routing strategies offline or classify a new ledger with the combined route."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

from anyfin.paths import PAIRED_RESULTS, ROOT, SUBMISSIONS_DIR, TRAINING_CSV, VALIDATION_CSV
from anyfin.ledger_pairs import operational_checking
from anyfin.prompts import BASE_PROMPT, CATEGORIES
from anyfin.routing import fallback_groups, local_predictions
from anyfin.scoring import row_score
from anyfin.standalone import predict_ledger
from anyfin.transactions import load_rows, read_fallback_predictions, without_labels, write_predictions
from experiments.counterparties import description
from experiments.independent_rules import infer


VALIDATION_BASELINE = SUBMISSIONS_DIR / "anyfin-track-a-validation-claude-haiku-5-5-baseline.csv"
UNKNOWN = "unknown"


def rules_predictions(rows, training):
    return [label for label, _, _ in infer(rows, training)]


def combine_labels(first, second):
    if first == UNKNOWN:
        return second
    if second in (first, UNKNOWN):
        return first
    return UNKNOWN


def combined_predictions(rows, training):
    return [combine_labels(first, second) for first, second in
            zip(predict_ledger(rows, training), rules_predictions(rows, training), strict=True)]


def score(labels, rows):
    counts = Counter(row_score(label, row["category"]) for label, row in zip(labels, rows, strict=True))
    return {"correct": counts[1], "wrong": counts[-1], "abstentions": counts[0],
            "points_per_100": round(100 * (counts[1] - counts[-1]) / len(rows), 3) if rows else None}


def scatter(labels, groups, representative_labels):
    result = labels.copy()
    for group, label in zip(groups, representative_labels, strict=True):
        for index in group:
            result[index] = label
    return result


def audit_strategy(rows, labels, saved, batch_size, training_descriptions):
    groups = fallback_groups(without_labels(rows), labels)
    pending = labels.count(UNKNOWN)
    covered = Counter(row["customer"] for row, label in zip(rows, labels, strict=True) if label != UNKNOWN)
    stats = {
        "rows": len(rows), "local_decisions": len(rows) - pending,
        "individual_fallback_inputs": pending, "grouped_fallback_inputs": len(groups),
        "individual_batches": ceil(pending / batch_size), "grouped_batches": ceil(len(groups) / batch_size),
        "local_decisions_by_customer": dict(covered),
    }
    has_gold = all(row.get("category") for row in rows)
    if has_gold:
        novel = [i for i, row in enumerate(rows) if description(row) not in training_descriptions]
        stats["local_score"] = score(labels, rows)
        stats["novel_description_rows"] = len(novel)
        stats["novel_description_score"] = score([labels[i] for i in novel], [rows[i] for i in novel])
        stats["groups_with_gold_conflicts"] = sum(len({rows[i]["category"] for i in group}) > 1 for group in groups)
    if saved is not None:
        individual = [guess if label == UNKNOWN else label for label, guess in zip(labels, saved, strict=True)]
        grouped = scatter(labels, groups, [saved[group[0]] for group in groups])
        stats["changed_by_grouping"] = sum(a != b for a, b in zip(individual, grouped, strict=True))
        if has_gold:
            stats["individual_replay_score"] = score(individual, rows)
            stats["grouped_replay_score"] = score(grouped, rows)
            cluster_deltas = defaultdict(list)
            for row, before, after in zip(rows, saved, grouped, strict=True):
                truth = row["category"]
                cluster_deltas[row["text"]].append(row_score(after, truth) - row_score(before, truth))
            stats["exact_description_clusters"] = len(cluster_deltas)
            stats["improved_description_clusters"] = sum(sum(delta) > 0 for delta in cluster_deltas.values())
            stats["worsened_description_clusters"] = sum(sum(delta) < 0 for delta in cluster_deltas.values())
        else:
            stats["individual_replay_disagreements_with_haiku"] = sum(
                a != b for a, b in zip(individual, saved, strict=True))
            stats["grouped_replay_disagreements_with_haiku"] = sum(
                a != b for a, b in zip(grouped, saved, strict=True))
    return stats


def evaluate(ledger, validation, batch_size):
    with PAIRED_RESULTS.open(encoding="utf-8-sig", newline="") as stream:
        saved_rows = {row["id"]: row for row in csv.DictReader(stream)}
    c3 = [row for row in ledger if row["customer"] == "c3"]
    if any(saved_rows.get(row["id"], {}).get("truth") != row["category"] for row in c3):
        raise ValueError("Saved Haiku comparison must match the c3 answer key by ID.")
    reference = {row_id: row["baseline"] for row_id, row in saved_rows.items()}
    records = {}
    for fold in ("c3", "c1", "c2", "validation"):
        training = [row for row in ledger if row["customer"] != fold]
        rows = validation if fold == "validation" else [row for row in ledger if row["customer"] == fold]
        test = without_labels(rows)
        saved = None
        if fold == "c3":
            saved = [reference[row["id"]] for row in rows]
        elif fold == "validation":
            saved = read_fallback_predictions(VALIDATION_BASELINE, rows)
        standalone = predict_ledger(test, training)
        rules = rules_predictions(test, training)
        strategies = {
            "haiku": [UNKNOWN] * len(rows),
            "existing_local": local_predictions(test, training),
            "standalone": standalone,
            "rules": rules,
            "combined": [combine_labels(a, b) for a, b in zip(standalone, rules, strict=True)],
        }
        seen = {description(row) for row in training}
        records[fold] = {
            "customers": dict(Counter(row["customer"] for row in rows)),
            "strategies": {name: audit_strategy(rows, labels, saved, batch_size, seen)
                           for name, labels in strategies.items()},
        }
    return {"batch_size": batch_size, "folds": records,
            "distillation": audit_distillation(ledger, validation, reference)}


def audit_distillation(ledger, validation, reference):
    from experiments.independent_lexical import LexicalClassifier, accept, calibrate

    # These rows are separate from the scored population. Their labels come from the
    # saved teacher output, so they cannot also serve as an evaluation ledger here.
    teacher = read_fallback_predictions(VALIDATION_BASELINE, validation)
    pseudo = [dict(row, category=label) for row, label in zip(validation, teacher, strict=True)
              if label != UNKNOWN]
    parameters, _ = calibrate([row for row in ledger if row["customer"] != "c3"])
    result = {"pseudo_labelled_rows": len(pseudo), "frozen_similarity_and_agreement": parameters, "folds": {}}
    for fold in ("c3", "c1", "c2"):
        training = [row for row in ledger if row["customer"] != fold]
        rows = [row for row in ledger if row["customer"] == fold]
        test = without_labels(rows)
        base = combined_predictions(test, training)
        pending = [i for i, label in enumerate(base) if label == UNKNOWN]

        def run(extra):
            started = time.perf_counter()
            classifier = LexicalClassifier(training + extra)
            fitted = time.perf_counter()
            predictions = accept(classifier.ranked([test[i] for i in pending]), parameters)
            predicted = time.perf_counter()
            filled = base.copy()
            for i, label in zip(pending, predictions, strict=True):
                filled[i] = label
            stats = {"fit_seconds": fitted - started, "residual_inference_seconds": predicted - fitted,
                     "local_score": score(filled, rows),
                     "fill_score": score(predictions, [rows[i] for i in pending])}
            if fold == "c3":
                replay = [reference[row["id"]] if label == UNKNOWN else label
                          for row, label in zip(rows, filled, strict=True)]
                stats["individual_replay_score"] = score(replay, rows)
                stats["grouped_fallback_inputs"] = len(fallback_groups(test, filled))
            return stats

        result["folds"][fold] = {"all_teacher_rows": run(pseudo)}
        if fold == "c3":
            result["folds"][fold]["exclude_each_teacher_bank"] = {
                bank: run([row for row in pseudo if row["bank"] != bank])
                for bank in sorted({row["bank"] for row in pseudo})
            }
    return result


def benchmark_local(validation, repeats, account_activity=False):
    result = {}
    with tempfile.TemporaryDirectory() as directory:
        source, target = (Path(directory) / name for name in ("input.csv", "output.csv"))
        for distinct in (False, True):
            ids = [f"frontier-bench-{index}" for index in range(10_000)]
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(validation[0]))
                writer.writeheader()
                for index, row_id in enumerate(ids):
                    copy, offset = divmod(index, len(validation))
                    row = dict(validation[offset], id=row_id)
                    for field in ("customer", "account"):
                        row[field] = f"{copy}-{row[field]}"
                    if distinct:
                        row["text"] += f" REF{index:08d}"
                    writer.writerow(row)
            elapsed = []
            for _ in range(repeats):
                started = time.perf_counter()
                command = [sys.executable, "-m", "experiments.independent_frontier", str(source), "-o", str(target)]
                if account_activity:
                    command.append("--account-activity")
                subprocess.run(command, cwd=ROOT, check=True)
                elapsed.append(time.perf_counter() - started)
                read_fallback_predictions(target, [{"id": row_id} for row_id in ids])
                with target.open(encoding="utf-8", newline="") as stream:
                    if [row["id"] for row in csv.DictReader(stream)] != ids:
                        raise ValueError("Benchmark output did not preserve the input order.")
            median = statistics.median(elapsed)
            name = "unique_reference_suffixes" if distinct else "repeated_validation_narratives"
            result[name] = {"seconds": elapsed, "median_seconds": median,
                            "median_rows_per_second": 10_000 / median,
                            "within_one_second": sum(value <= 1 for value in elapsed)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="Classify a new CSV. Omit for the offline benchmark.")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--benchmark-runs", type=int, default=0,
                        help="Include cold 10,000-row timings in the offline benchmark.")
    parser.add_argument("--group-fallback", action="store_true")
    parser.add_argument("--account-activity", action="store_true",
                        help="Resolve transfers between active checking accounts in the input ledger.")
    fallback = parser.add_mutually_exclusive_group()
    fallback.add_argument("--fallback-predictions", type=Path)
    fallback.add_argument("--live-haiku", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.benchmark_runs < 0:
        parser.error("Batch size and workers must be positive and benchmark runs nonnegative.")
    protected = [TRAINING_CSV, VALIDATION_CSV, VALIDATION_BASELINE, PAIRED_RESULTS,
                 args.input, args.fallback_predictions]
    if args.output and args.output.resolve() in {path.resolve() for path in protected if path is not None}:
        parser.error("Input and output paths must differ.")
    ledger = load_rows(TRAINING_CSV)
    if not args.input:
        if args.account_activity:
            parser.error("--account-activity requires an input ledger.")
        if args.live_haiku or args.fallback_predictions or args.group_fallback or args.model:
            parser.error("Fallback controls require an input CSV.")
        validation = load_rows(VALIDATION_CSV)
        stats = evaluate(ledger, validation, args.batch_size)
        if args.benchmark_runs:
            stats["combined_local_cold_10000"] = benchmark_local(validation, args.benchmark_runs)
        report = json.dumps(stats, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        else:
            print(report, end="")
        return
    if args.benchmark_runs or (args.model and not args.live_haiku):
        parser.error("Benchmark timings require no input and model selection requires live Haiku.")
    if args.group_fallback and not (args.live_haiku or args.fallback_predictions):
        parser.error("Grouped fallback needs a live model or a saved prediction CSV.")
    rows = load_rows(args.input)
    if {row["id"] for row in rows} & {row["id"] for row in ledger}:
        parser.error("Training and input IDs must be disjoint.")
    if any(row.get("category") not in CATEGORIES for row in ledger):
        parser.error("Training rows require valid categories.")
    test = without_labels(rows)
    labels = combined_predictions(test, ledger)
    if args.account_activity:
        labels = operational_checking(test, labels)
    if args.live_haiku or args.fallback_predictions:
        groups = (fallback_groups(test, labels) if args.group_fallback else
                  [[index] for index, label in enumerate(labels) if label == UNKNOWN])
        if args.fallback_predictions:
            saved = read_fallback_predictions(args.fallback_predictions, rows)
            labels = scatter(labels, groups, [saved[group[0]] for group in groups])
        elif groups:
            from anyfin.classify import classify_representatives
            from anyfin.claude import MODEL

            representatives = classify_representatives(test, groups, args.model or MODEL,
                                                        BASE_PROMPT, args.batch_size, args.workers)
            labels = scatter(labels, groups, representatives)
    write_predictions(rows, labels, args.output)


if __name__ == "__main__":
    main()
