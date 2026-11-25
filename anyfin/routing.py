"""Resolve local decisions and group equivalent rows for hybrid Claude fallback."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from .ledger_pairs import paired_labels
from .local_router import LocalRouter


def local_predictions(rows, training_rows):
    """Use ledger evidence first, then the model fitted to labelled descriptions."""
    router = LocalRouter(training_rows)
    paired = paired_labels(rows)
    return [paired[i] if i in paired else router.predict(row) for i, row in enumerate(rows)]


_GENERIC_PAYMENT_WORDS = set(
    "bg pg bet betalning inbetalning insättning autogiro kort kortköp köp uttag bankgiro "
    "plusgiro ocr ref se sek ab från till via konto bank transfer".split()
)


def fallback_groups(rows, local_labels):
    """Group unresolved entries from the same account using the visible payment context."""
    groups = {}
    for i, (row, label) in enumerate(zip(rows, local_labels, strict=True)):
        if label != "unknown":
            continue
        description = " ".join(unicodedata.normalize("NFKC", row["text"]).casefold().split())
        # A six-digit posting date immediately after Kortköp varies across repeat purchases.
        description = re.sub(r"^(kortköp|kortkop) \d{6} (?=\S)", r"\1 ", description)
        try:
            amount = Decimal(row["amount_sek"])
        except InvalidOperation:
            amount = None
        if amount is None or not amount.is_finite():
            groups[("invalid amount", i)] = [(i, Decimal(0))]
            continue
        # Keep the exact amount for opaque bank references or personal/internal transfers.
        # Named purchases must still agree on direction and order of magnitude.
        words = re.findall(r"[^\W\d_]+", description)
        specific = any(len(word) >= 3 and word not in _GENERIC_PAYMENT_WORDS for word in words)
        personal = any(word.startswith(("överför", "overfor", "swish", "transfer")) for word in words)
        amount_key = (amount.compare(0), amount.adjusted()) if specific and not personal else amount
        account = tuple(row.get(k, "") for k in ("customer", "bank", "account", "account_type", "account_name"))
        key = (account, description, amount_key)
        groups.setdefault(key, []).append((i, amount))
    result = []
    for group in groups.values():
        ranked = sorted(group, key=lambda item: (abs(item[1]), rows[item[0]]["id"]))
        representative = ranked[len(ranked) // 2][0]
        result.append([representative, *(i for i, _ in group if i != representative)])
    return result
