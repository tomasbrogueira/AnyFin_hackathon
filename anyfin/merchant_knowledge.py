"""Offline purchase categorization from the vendored Name Suggestion Index."""

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


_CARD = re.compile(r"^(?:kortkop|kortbetalning|varukop|korttransaktion|kort|card purchase)\b[\s:*_-]*(?:\d{6,8}\s+)?")
_SWISH = re.compile(r"^swish\s+(?:(?:betalning|till|skickad)\s+)?")
_REFUND = re.compile(r"^(?:kortretur|retur|aterkop|aterbetalning|refund)\b[\s:*_-]*(?:\d{6,8}\s+)?")
_PROCESSOR = re.compile(r"^(?:klarna|paypal|zettle|sumup|iz)[\s]*[*_]+[\s*_]*")
_TOKENS = re.compile(r"[a-z0-9]+")
_FINANCE = re.compile(r"\b(?:bank|banken|forsakring|insurance|finans|finance|kredit|kreditkort|inkasso|loan|salary|payroll)\b")


@lru_cache(maxsize=1)
def _brand_index():
    data = json.loads(Path(__file__).with_name("nsi_merchants.json").read_text(encoding="utf-8"))
    index = {tuple(alias.split()): category for category, aliases in data["aliases"].items() for alias in aliases}
    index.update((tuple(alias.split()), "unknown") for alias in data["blocked"])
    return index, max(map(len, index))


@lru_cache(maxsize=8192)
def _lookup(text, incoming):
    text = text.casefold().strip().replace("’", "").replace("'", "")
    if not text.isascii():
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    card = _CARD.match(text)
    if card:
        text = text[card.end():]
    refund = _REFUND.match(text)
    if incoming and not (refund or card):
        return "unknown"
    if refund:
        text = _CARD.sub("", text[refund.end():])
    elif not card:
        text = _SWISH.sub("", text)
    text = _PROCESSOR.sub("", text)
    if _FINANCE.search(text):
        return "unknown"
    tokens = tuple(_TOKENS.findall(text))
    index, max_length = _brand_index()
    for length in range(min(max_length, len(tokens)), 0, -1):
        label = index.get(tokens[:length])
        if label is not None:
            return label
    return "unknown"


def lookup_merchant(row):
    """Classify a merchant at the start of the payee field when the context permits."""
    try:
        amount = float(row.get("amount_sek"))
    except (ValueError, TypeError):
        return "unknown"
    text = row.get("text")
    if not isinstance(text, str) or not math.isfinite(amount) or amount == 0:
        return "unknown"
    return _lookup(text, amount > 0)
