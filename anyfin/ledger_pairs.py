"""Find high-confidence internal transfers using both sides of one customer's ledger."""

import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache

from .local_router import _fold


_WORDS = re.compile(r"[a-z]+|[0-9]+")


@lru_cache(maxsize=16384)
def _tokens(text):
    return " ".join(_WORDS.findall(_fold(text)))


def _words(value):
    return _tokens(str(value or ""))


_TRANSFER = re.compile(r"^(?:overforing|overf|ovf|transfer)\b")
_TRANSFER_WORDS = set(
    "till fran mellan eget egna konto konton accounts to from own account "
    "spar sparkonto sparande buffert buffertkonto personkonto lonekonto privatkonto "
    "vardagskonto transaktionskonto checking savings".split()
)
_SAVINGS = re.compile(r"\b(?:sparkonto|sparande|buffert(?:konto)?|savings)\b")
_EVERYDAY = re.compile(r"\b(?:lonekonto|personkonto|privatkonto|vardagskonto|transaktionskonto|checking|current)\b")
_CARD_BILL = re.compile(r"^(?:betalning bg|bg|bankgiro|autogiro|kreditkortsfaktura)\b")
_CARD_NAME = re.compile(r"\b(?:mastercard|visa|amex|kreditkort|kreditkortsfaktura)\b")
_CARD_RECEIPTS = {"inbetalning tack", "inbetalning kreditkort", "betalning mottagen", "payment received", "payment thank you"}
_DAY = timedelta(days=1)


def _account_role(row):
    kind = _words(row.get("account_type"))
    name = _words(row.get("account_name"))
    if kind == "credit card":
        return "card"
    if kind == "savings" or (kind == "checking" and _SAVINGS.search(name)):
        return "savings"
    if kind == "checking":
        return "everyday" if _EVERYDAY.search(name) else "checking"
    return ""


def _match_transfers(groups, labels):
    for debit, credit in groups.values():
        if len(debit) != 1 or len(credit) != 1:
            continue
        payer, receiver = debit[0], credit[0]
        if payer[1] == receiver[1]:
            continue
        roles = {payer[2], receiver[2]}
        if "savings" in roles:
            label = "transfer.savings"
        elif roles == {"everyday"}:
            label = "transfer.own"
        else:
            continue
        # Ignore account numbers, directions and the account names present in this pair.
        # An unexplained merchant or purpose in either memo blocks the match.
        allowed = _TRANSFER_WORDS | set(payer[4].split()) | set(receiver[4].split())
        if all(token.isdigit() or token in allowed for leg in (payer, receiver)
               for token in leg[3].split()[1:]):
            labels[payer[0]] = labels[receiver[0]] = label


def _match_card_bills(bills, receipts, labels):
    for (customer, posted, amount), payers in bills.items():
        if len(payers) != 1:
            continue
        matches = [leg for day in (posted, posted + _DAY)
                   for leg in receipts.get((customer, day, amount), ())]
        if len(matches) != 1:
            continue
        receiver, account, received = matches[0]
        competing = sum(len(bills.get((customer, day, amount), ())) for day in (received - _DAY, received))
        payer, paying_account = payers[0]
        if competing == 1 and account != paying_account:
            labels[payer] = "credit.loan"
            labels[receiver] = "transfer.own"


def paired_labels(rows):
    """Return row-indexed labels for unambiguous pairs. Unmatched rows are omitted."""
    transfers = defaultdict(lambda: ([], []))
    bills = defaultdict(list)
    receipts = defaultdict(list)
    for i, row in enumerate(rows):
        customer, account = row.get("customer"), row.get("account")
        if not customer or not account:
            continue
        role, memo = _account_role(row), _words(row.get("text"))
        ordinary = role in {"savings", "everyday", "checking"} and _TRANSFER.match(memo)
        bill = role in {"checking", "everyday"} and _CARD_BILL.match(memo) and _CARD_NAME.search(memo)
        receipt = role == "card" and memo in _CARD_RECEIPTS
        if not (ordinary or bill or receipt):
            continue
        try:
            posted = date.fromisoformat(row["date"])
            amount = Decimal(str(row["amount_sek"]))
        except (KeyError, TypeError, ValueError, InvalidOperation):
            continue
        if not amount.is_finite() or amount == 0 or posted in (date.min, date.max):
            continue
        key = (customer, posted, amount.copy_abs())
        if ordinary:
            transfers[key][int(amount > 0)].append((i, account, role, memo, _words(row.get("account_name"))))
        elif bill and amount < 0:
            bills[key].append((i, account))
        elif receipt and amount > 0:
            receipts[key].append((i, account, posted))

    labels = {}
    _match_transfers(transfers, labels)
    _match_card_bills(bills, receipts, labels)
    return labels


def operational_checking(rows, seeds):
    """Infer an everyday account role from independent, previously labelled purchases."""
    spend_days = defaultdict(set)
    transfers = defaultdict(list)
    for i, (row, label) in enumerate(zip(rows, seeds, strict=True)):
        if _account_role(row) not in {"checking", "everyday"}:
            continue
        account = row.get("customer"), row.get("account")
        if not all(account):
            continue
        try:
            day = date.fromisoformat(row["date"])
            amount = Decimal(row["amount_sek"])
            if not amount.is_finite() or not amount:
                continue
        except (InvalidOperation, KeyError, TypeError, ValueError):
            continue
        if _TRANSFER.match(_words(row.get("text"))):
            transfers[account[0], day, amount].append((i, account))
        elif label.startswith("spend.") and amount < 0:
            spend_days[account].add(day)
    result = seeds.copy()
    for (person, day, amount), debits in transfers.items():
        if amount >= 0 or len(debits) != 1:
            continue
        credits = transfers.get((person, day, -amount), ())
        if len(credits) != 1:
            continue
        (i, source), (j, dest) = debits[0], credits[0]
        if (source != dest and min(len(spend_days[source]), len(spend_days[dest])) >= 2
                and {seeds[i], seeds[j]} <= {"unknown", "transfer.own"}):
            result[i] = result[j] = "transfer.own"
    return result
