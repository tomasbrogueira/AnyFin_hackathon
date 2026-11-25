"""Read the event ledger and align saved predictions by transaction ID."""

import csv
import sys
from contextlib import nullcontext

from .paths import resolve_input
from .prompts import CATEGORIES

FEATURES = ("account_name", "account_type", "amount_sek", "text")


def without_labels(rows):
    """Keep the answer key out of the inputs passed to a classifier."""
    return [{key: value for key, value in row.items() if key != "category"} for row in rows]


def load_rows(path):
    """Load labelled or unlabelled event data, retaining gold labels for local scoring."""
    with resolve_input(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"id", *FEATURES}.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every transaction needs a unique, nonempty id.")
    return rows


def read_fallback_predictions(path, rows):
    """Read saved predictions for exactly this ledger and align them by transaction ID."""
    with resolve_input(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "category"]:
            raise ValueError("Saved fallback predictions must have the header id,category.")
        by_id = {}
        for row in reader:
            if row["id"] in by_id or row["category"] not in CATEGORIES:
                raise ValueError("Saved fallback predictions contain a duplicate ID or invalid category.")
            by_id[row["id"]] = row["category"]
    if by_id.keys() != {row["id"] for row in rows}:
        raise ValueError("Saved fallback predictions must match the IDs in the input ledger.")
    return [by_id[row["id"]] for row in rows]


def write_predictions(rows, labels, output_path=None):
    """Write the event's two-column CSV in input order, using stdout by default."""
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output = output_path.open("w", encoding="utf-8", newline="") if output_path else nullcontext(sys.stdout)
    with output as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "category"))
        writer.writerows((row["id"], label) for row, label in zip(rows, labels, strict=True))
