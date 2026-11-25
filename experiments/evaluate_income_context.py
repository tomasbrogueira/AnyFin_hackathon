"""Evaluate recurring pay on top of the combined classifier with account activity."""

import argparse
import re
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from time import perf_counter

from anyfin.ledger_pairs import _account_role, operational_checking
from anyfin.local_router import _describe
from anyfin.paths import TRAINING_CSV, VALIDATION_CSV
from anyfin.transactions import load_rows, write_predictions
from experiments.independent_frontier import combined_predictions


_EXCLUDED = re.compile(r"\b(?:swish|transfer|overforing|ovf|aterbetal\w*|aterkop\w*|retur\w*|refund\w*|kontant\w*|pension\w*|bidrag\w*|forsakr\w*)\b")


def recurring_pay(rows, seeds, training):
    salaries = [float(r["amount_sek"]) for r in training if r["category"] == "income.employment"
                and float(r["amount_sek"]) > 0]
    if not salaries:
        return list(seeds)
    floor, ceiling = 0.5 * min(salaries), 2.0 * max(salaries)
    sides, inflows, families = defaultdict(set), defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        try:
            amount = Decimal(row["amount_sek"])
            day = date.fromisoformat(row["date"])
            if not amount.is_finite() or not amount:
                continue
        except (InvalidOperation, KeyError, TypeError, ValueError):
            continue
        customer, account = row.get("customer"), row.get("account")
        if not customer or not account:
            continue
        sides[customer, day.toordinal(), amount].add(account)
        if amount < 0 or _account_role(row) not in {"checking", "everyday"}:
            continue
        month = 12 * day.year + day.month
        inflows[customer, account, month].append(amount)
        text, _, _, context, body = _describe(row)
        if context[0] not in {"other", "deposit"} or not body or _EXCLUDED.search(text):
            continue
        families[customer, account, body].append((i, day, month, amount))
    candidates = defaultdict(list)
    for (customer, account, _), group in families.items():
        months = {m for _, _, m, _ in group}
        values = [amount for _, _, _, amount in group]
        if (len(months) < 4 or len(months) != len(group) or max(months) - min(months) + 1 != len(months)
                or min(values) < floor or max(values) > ceiling or max(values) > Decimal("1.5") * min(values)):
            continue
        if not all(22 <= day.day <= 28 and day.weekday() < 5 for _, day, _, _ in group):
            continue
        if any(seeds[i] not in {"unknown", "income.employment"} for i, _, _, _ in group):
            continue
        mirrors = (sides.get((customer, day.toordinal() + offset, -a), set()) - {account}
                   for _, day, _, a in group for offset in (-2, -1, 0, 1, 2))
        if any(mirrors):
            continue
        shares = [amount / sum(inflows[customer, account, m]) for _, _, m, amount in group]
        if median(shares) >= Decimal("0.60"):
            candidates[customer, account].append(group)
    predictions = list(seeds)
    for groups in candidates.values():
        if len(groups) == 1:
            for i, _, _, _ in groups[0]:
                predictions[i] = "income.employment"
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", choices=("c1", "c2", "c3", "validation"), required=True)
    parser.add_argument("-o", "--output", type=Path, help="Write the independent candidate's id,category CSV.")
    args = parser.parse_args()
    if args.output and args.output.resolve() in {TRAINING_CSV.resolve(), VALIDATION_CSV.resolve()}:
        parser.error("Select a prediction path distinct from the ledgers.")
    ledger = load_rows(TRAINING_CSV)
    if args.holdout == "validation":
        train, test = ledger, load_rows(VALIDATION_CSV)
    else:
        train = [r for r in ledger if r["customer"] in {"c1", "c2"} - {args.holdout}]
        test = [r for r in ledger if r["customer"] == args.holdout]
    seeds = operational_checking(test, combined_predictions(test, train))
    start = perf_counter()
    preds = recurring_pay(test, seeds, train)
    elapsed = 1000 * (perf_counter() - start)
    delta = [(r, y) for r, s, y in zip(test, seeds, preds) if s != y]
    if args.output:
        write_predictions(test, preds, args.output)
    scores = Counter("correct" if y == r["category"] else "wrong" for r, y in delta if r.get("category"))
    print(f"{args.holdout}: {len(test)} rows, {len(delta)} changes, {dict(scores)}, "
          f"{preds.count('unknown')} remaining abstentions, {elapsed:.2f} ms")
    if args.holdout in {"c1", "c2"}:
        blind = recurring_pay(test, ["unknown"] * len(test), train)
        outcomes = Counter("correct" if y == r["category"] else "wrong" for r, y in zip(test, blind) if y != "unknown")
        print(f"Independent of the baseline's seeds: {dict(outcomes)}.")


if __name__ == "__main__":
    main()
