"""Classify a Swedish transaction ledger locally with the Python standard library."""

import argparse
from pathlib import Path

from .ledger_pairs import operational_checking, paired_labels
from .local_router import LocalRouter
from .merchant_knowledge import lookup_merchant
from .paths import TRAINING_CSV, resolve_input
from .prompts import CATEGORIES
from .semantic_router import SemanticRouter
from .transactions import load_rows, write_predictions


def predict_ledger(rows, training_rows, *, account_activity=False):
    if not training_rows or any(row.get("category") not in CATEGORIES for row in training_rows):
        raise ValueError("Training data must have valid category labels.")
    matched = paired_labels(rows)
    lexical = LocalRouter(training_rows)
    semantic = SemanticRouter(training_rows)
    labels = []
    for i, row in enumerate(rows):
        label = matched.get(i)
        if label is None:
            label = lexical.predict(row)
        if label == "unknown":
            label = semantic.predict(row)
        if label == "unknown":
            label = lookup_merchant(row)
        labels.append(label)
    return operational_checking(rows, labels) if account_activity else labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=resolve_input)
    parser.add_argument("--train", type=resolve_input, default=TRAINING_CSV)
    parser.add_argument("--account-activity", action="store_true",
                        help="Resolve transfers between checking accounts with observed merchant spending.")
    parser.add_argument("--output", "-o", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    if args.output.resolve() in {args.train.resolve(), args.input.resolve()}:
        parser.error("Choose an output path distinct from both input ledgers.")
    try:
        rows, training = load_rows(args.input), load_rows(args.train)
        if {row["id"] for row in rows}.intersection(row["id"] for row in training):
            parser.error("Training and prediction IDs overlap.")
        write_predictions(rows, predict_ledger(rows, training, account_activity=args.account_activity), args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
