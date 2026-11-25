"""Frozen deterministic classifier for the Anyfin customer holdout."""

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from anyfin.scoring import row_score
from anyfin.transactions import load_rows, read_fallback_predictions, without_labels, write_predictions


SIGNALS = {
    "spend.groceries": r"ica|coop|willys|hemkop|lidl|aldi|city gross|mathem|pingo doce|supermark(?:et|t)\w*|livsmedel\w*",
    "spend.dining": r"espresso house|burger king|mcdonalds|foodora|wolt|uber eats|max (?:gbg|hamburgare)|restaurang\w*|restaurant\w*|(?:ka|ca)fe\w*|sushi|pizz\w*|bistro|burger|foodtruck|salladsbar\w*|cervejaria",
    "spend.transport": r"uber (?:trip|ride)|bolt (?:trip|ride)|sl centralen|sl biljett|voi (?:technology|scooter)|cp comboios|vasttrafik|skanetrafiken|sj ab|mtr express|easypark|parkster|preem|ingo|circle k|okq8|taxi\w*|parkering\w*|parking|trangselskatt\w*|bensin\w*",
    "spend.shopping": r"zalando|lyko|h m|amazon|ikea|jysk|jula|bauhaus|byggmax|elgiganten|clas ohlson|zara|arket|hm|stadium|kjell|lindex|kappahl|intersport|biltema|bokhandel\w*|byggvar\w*",
    "spend.leisure": r"sats|friskis|nordic wellness|spotify|netflix|viaplay|disney plus|hbo max|sf bio|filmstaden|steamgames|steam games|liseberg|grona lund|ticketmaster|booking com|airbnb|museu\w*|hotell?\w*|airlines?|bio(?:graf)?|gym\w*|sas",
    "spend.housing": r"telia|tele2|telenor|bahnhof|bredband\w*|ellevio|vattenfall|eon|e on|heimstaden|bostader|bostadsbolag\w*|bostadsavgift\w*|flytthjalp\w*|hyra|hyran|elnat\w*|elhandel\w*|fjarrvarme\w*",
    "spend.health": r"apotek\w*|apotea|kronans apotek|pharmacy|dental\w*|tandvard\w*|tandlak\w*|folktandvard\w*|vardcentral\w*|sjukvard\w*",
    "spend.family": r"forskola\w*|barnomsorg\w*|skolavgift\w*|fritidshem\w*|dagis\w*",
    "spend.gambling": r"svenska spel|atg|unibet|betsson|bet365|betfair|leovegas|casin\w*|kasin\w*|lotter\w*",
}

SIGNAL_PATS = {label: re.compile(r"\b(?:" + pattern + r")\b") for label, pattern in SIGNALS.items()}
RAILS = [
    ("card", re.compile(r"^(?:kortkop|kortbetalning|varukop|korttransaktion|card purchase)\b\s*")),
    ("refund", re.compile(r"^(?:kortretur|retur|aterkop|aterbetalning|refund)\b\s*")),
    ("bill", re.compile(r"^(?:betalning (?:bg|pg)|autogiro|bankgiro|plusgiro|bg|pg)\b\s*")),
    ("swish", re.compile(r"^swish\b\s*(?:(?:betalning|inbetalning|utbetalning|mottagen|skickad|till|fran|in|ut)\b\s*)?")),
    ("transfer", re.compile(r"^(?:overforing|overf|ovf|transfer)\b\s*")),
    ("deposit", re.compile(r"^(?:insattning|utbetalning|inbetalning)\b\s*")),
]
PROCESSOR = re.compile(r"^(?:zettle|sumup|iz|paypal|trustly|klarna)\s+")
EXACT_SKIP = {"transfer", "swish"}
NUMERIC = re.compile(r"(?:\b\d+\b|\b\w*\d{3,}\w*\b)")
CURRENCY_TAIL = re.compile(r"\b(?:eur|usd|gbp|dkk|nok|pln|sek)\s+\d.*$")
SAVINGS = re.compile(r"\b(?:spar(?:konto|ande)?|buffert(?:konto)?|savings)\b")
EVERYDAY = re.compile(r"\b(?:personkonto|lonekonto|privatkonto|vardagskonto|transaktionskonto|checking|eget konto|egna konton)\b")
SIMPLE_TRANSFER = re.compile(r"\b(?:till|fran|mellan|eget|egna|konto|konton|spar(?:konto|ande)?|buffert(?:konto)?|savings|personkonto|lonekonto|privatkonto|vardagskonto|transaktionskonto)\b")
CARD_PROVIDER = re.compile(r"\b(?:mastercard|visa|amex|kreditkort|kortfaktura)\w*\b")
CARD_RECEIPT = re.compile(r"^(?:inbetalning(?: tack| kreditkort)?|betalning mottagen|payment received|payment thank you)$")
CASH = re.compile(r"^(?:uttag\b.*\bbankomat|bankomat(?:uttag)?\b|kontantuttag\b|atm withdrawal\b)")
SALARY = re.compile(r"^(?:lon|loneutbetalning|manadslon|salary)\b")
BENEFIT = re.compile(r"\b(?:barnbidrag|foraldrapenning|underhallsstod|sjukpenning|bostadsbidrag|forsakringskassan)\b")
CAPITAL = re.compile(r"^(?:ranta|ranteinkomst|insattningsranta|sparranta|utdelning|dividend)\b")
SALES = re.compile(r"\b(?:tradera|vinted|sellpy|forsaljning|salde)\b")
COLLECTIONS = re.compile(r"\b(?:inkasso\w*|kronofogden|intrum|lowell|alek tum|alektum)\b")
LOANS = re.compile(r"\b(?:bolan|hypotek|amortering|lanebetalning|kortfaktura|kreditkortsfaktura)\w*\b")
LENDERS = re.compile(r"\b(?:klarna|qliro|walley|csn|santander consumer|resurs bank|nordax bank|anyfin)\b")
SWISH_PERSON = re.compile(r"^[a-z]+ [a-z]$")
DAY = timedelta(days=1)
TOKENS = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=8192)
def fold(value):
    value = value.casefold()
    if not value.isascii():
        value = unicodedata.normalize("NFKD", value)
        value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(TOKENS.findall(value))


@dataclass(frozen=True, slots=True)
class Memo:
    full: str
    body: str
    rail: str
    amount: Decimal
    role: str


def describe(row):
    full = fold(row["text"])
    amount = Decimal(row["amount_sek"])
    name, kind = fold(row["account_name"]), row["account_type"]
    role = "savings" if kind == "savings" or SAVINGS.search(name) else kind
    rail, raw_body = "other", full
    for candidate, prefix in RAILS:
        match = prefix.match(full)
        if match:
            rail, raw_body = candidate, full[match.end():]
            break
    # A BNPL invoice and a card purchase via its processor carry different labels.
    raw_body = CURRENCY_TAIL.sub("", raw_body)
    raw_body = NUMERIC.sub("", raw_body)
    body = " ".join(raw_body.split())
    if rail in {"card", "other", "refund"}:
        body = PROCESSOR.sub("", body)
    return Memo(full, body, rail, amount, role)


def exact_key(memo):
    if memo.rail in EXACT_SKIP or not memo.body or not memo.amount:
        return None
    if memo.rail in {"card", "other", "refund"} and (memo.amount < 0 or memo.rail in {"card", "refund"}):
        return "merchant", memo.body
    return memo.rail, memo.amount > 0, memo.body


def pair_labels(rows, memos):
    transfers, bills, receipts = defaultdict(list), defaultdict(list), defaultdict(list)
    for index, (row, memo) in enumerate(zip(rows, memos, strict=True)):
        if not memo.amount or (memo.rail != "transfer" and not CARD_PROVIDER.search(memo.full)
                               and not CARD_RECEIPT.fullmatch(memo.full)):
            continue
        key = row["customer"], date.fromisoformat(row["date"]), abs(memo.amount)
        if memo.rail == "transfer" and memo.role in {"checking", "savings"}:
            trade_body = SIMPLE_TRANSFER.sub("", memo.body)
            if not trade_body.strip():
                transfers[key].append(index)
        elif memo.rail == "bill" and memo.role != "credit_card" and memo.amount < 0:
            bills[key].append(index)
        elif memo.role == "credit_card" and memo.amount > 0:
            receipts[key].append(index)

    labels = {}
    for members in transfers.values():
        if len(members) != 2:
            continue
        first, second = members
        a, b = memos[first], memos[second]
        if a.amount == -b.amount and rows[first]["account"] != rows[second]["account"]:
            label = "transfer.savings" if "savings" in {a.role, b.role} else "transfer.own"
            labels[first] = labels[second] = label
    for (customer, day, amount), members in bills.items():
        matching = receipts.get((customer, day, amount), []) + receipts.get((customer, day + DAY, amount), [])
        if len(members) == len(matching) == 1:
            receipt = matching[0]
            receipt_day = date.fromisoformat(rows[receipt]["date"])
            rivals = bills.get((customer, receipt_day, amount), []) + bills.get((customer, receipt_day - DAY, amount), [])
            if len(rivals) == 1:
                labels[members[0]], labels[receipt] = "credit.loan", "transfer.own"
    return labels


def semantic_rule(memo):
    text, body, rail, amount, role = memo.full, memo.body, memo.rail, memo.amount, memo.role
    if amount < 0 and CASH.match(text):
        return "transfer.cash"
    if amount > 0 and rail not in {"card", "refund"}:
        if SALARY.match(text):
            return "income.employment"
        if BENEFIT.search(text):
            return "income.public"
        if CAPITAL.match(text) or (rail == "deposit" and CAPITAL.match(body)):
            return "income.capital"
        if SALES.search(text):
            return "income.sales"
        if role == "credit_card" and CARD_RECEIPT.fullmatch(text):
            return "transfer.own"
    if amount < 0 and rail not in {"card", "refund"}:
        if COLLECTIONS.search(body):
            return "credit.collections"
        if LOANS.search(body) or (rail == "bill" and (LENDERS.search(body) or CARD_PROVIDER.search(body))):
            return "credit.loan"
    if rail == "transfer":
        if role == "savings" or SAVINGS.search(body):
            return "transfer.savings"
        if EVERYDAY.search(body):
            return "transfer.own"
        return "unknown"
    if rail == "swish" and SWISH_PERSON.fullmatch(body):
        return "transfer.person"
    if SIGNAL_PATS["spend.gambling"].search(body):
        return "spend.gambling"
    if not amount or (amount > 0 and rail not in {"card", "refund", "swish"}):
        return "unknown"
    matches = [label for label, pattern in SIGNAL_PATS.items() if pattern.search(body)]
    return matches[0] if len(matches) == 1 else "unknown"


def infer(rows, training):
    memos = [describe(row) for row in rows]
    pairs = pair_labels(rows, memos)
    exact = defaultdict(set)
    for row in training:
        key = exact_key(describe(row))
        if key:
            exact[key].add(row["category"])
    records = []
    for index, memo in enumerate(memos):
        key = exact_key(memo)
        seen = exact.get(key, set())
        rule = semantic_rule(memo)
        label = pairs.get(index, rule)
        source = "pair" if index in pairs else "rule" if rule != "unknown" else "unknown"
        if source == "unknown" and len(seen) == 1:
            label, source = next(iter(seen)), "exact"
        records.append((label, source, bool(seen)))
    return records


def summarize(labels, truth):
    outcome = Counter(row_score(label, answer) for label, answer in zip(labels, truth, strict=True))
    count = len(labels)
    return f"n={count} +{outcome[1]} -{outcome[-1]} abstain={outcome[0]} net={100 * (outcome[1] - outcome[-1]) / max(1, count):.1f}"


def replay_predictions(rows, labels, path, use_paired_holdout, grouped):
    if use_paired_holdout:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            by_id = {row["id"]: row["baseline"] for row in csv.DictReader(stream)}
        baseline = [by_id[row["id"]] for row in rows]
    else:
        baseline = read_fallback_predictions(path, rows)
    if grouped:
        from anyfin.routing import fallback_groups

        groups = fallback_groups(rows, labels)
    else:
        groups = [[index] for index, label in enumerate(labels) if label == "unknown"]
    replayed = labels.copy()
    for group in groups:
        for index in group:
            replayed[index] = baseline[group[0]]
    return replayed, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="data/transactions.csv")
    parser.add_argument("--holdout", default="c3")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--replay", type=Path, help="Saved id,category predictions or paired c3 holdout CSV.")
    parser.add_argument("--grouped", action="store_true", help="Replay only one saved prediction per referral group.")
    args = parser.parse_args()
    if args.grouped and not args.replay:
        parser.error("--grouped needs --replay")
    if args.replay and not args.input and args.holdout != "c3":
        parser.error("the paired saved replay covers only the c3 holdout")
    inputs = {Path(path).resolve() for path in (args.training, args.input, args.replay) if path}
    outputs = [path.resolve() for path in (args.output, args.details) if path]
    if any(path in inputs for path in outputs) or len(outputs) != len(set(outputs)):
        parser.error("input and output paths must all be distinct")
    ledger = load_rows(args.training)
    training = [row for row in ledger if args.input or row["customer"] != args.holdout]
    rows = load_rows(args.input) if args.input else [row for row in ledger if row["customer"] == args.holdout]
    if not rows:
        parser.error("no transactions to classify")
    if {row["id"] for row in rows}.intersection(row["id"] for row in training):
        parser.error("training and prediction transaction IDs overlap")
    start = perf_counter()
    records = infer(without_labels(rows), training)
    elapsed = perf_counter() - start
    predictions = [label for label, _, _ in records]
    print(f"rows={len(rows)} seconds={elapsed:.6f} rows_per_second={len(rows) / elapsed:.0f}")
    print(f"routes={dict(Counter(source for _, source, _ in records))}")
    if args.replay:
        predictions, groups = replay_predictions(rows, predictions, args.replay, not args.input, args.grouped)
        print(f"fallback_representatives={len(groups)}")
        if not args.input:
            print("saved-Haiku replay", summarize(predictions, [row["category"] for row in rows]))
            print("mixed-truth groups", sum(len({rows[index]["category"] for index in group}) > 1 for group in groups))
    if args.output:
        write_predictions(rows, predictions, args.output)
    if not args.input:
        outcomes = [(row["category"], label, source, seen) for row, (label, source, seen) in zip(rows, records, strict=True)]
        for name, group in [
            ("all", outcomes),
            ("rules-only", [(truth, label if source != "exact" else "unknown", source, seen)
                            for truth, label, source, seen in outcomes]),
            ("known descriptor", [outcome for outcome in outcomes if outcome[-1]]),
            ("novel descriptor", [outcome for outcome in outcomes if not outcome[-1]]),
            *[(route, [outcome for outcome in outcomes if outcome[2] == route]) for route in ("pair", "rule", "exact")],
        ]:
            print(name, summarize([out[1] for out in group], [out[0] for out in group]))
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with args.details.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "customer", "prediction", "source", "seen_descriptor"])
            writer.writerows((row["id"], row["customer"], *record) for row, record in zip(rows, records, strict=True))


if __name__ == "__main__":
    main()
