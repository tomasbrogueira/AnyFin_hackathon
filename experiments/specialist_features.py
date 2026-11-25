"""Normalize Anyfin narrative and account fields for a sparse classifier."""

import re
import unicodedata


CARD = re.compile(r"^(?:kortkop|kortbetalning|varukop|korttransaktion|card purchase|kort)\b[\s:_-]*")
REFUND = re.compile(r"^(?:aterkop|kortretur|retur|refund|aterbetalning)\b[\s:_-]*")
WRAPPER = re.compile(r"^(?:zettle|izettle|sumup|iz|paypal|klarna|trustly)[\s_*]*[*_][\s_*]*")
DATE = re.compile(r"^\d{6,8}\b[\s:_-]*")
DIGITS = re.compile(r"\d+")
SWISH = re.compile(r"^swish(?:\s+(?:betalning|inbetalning|till|fran|skickad))?\s+")
GIRO = re.compile(r"^(?:(?:betalning|inbet)\s+)?(?:bg|pg|bankgiro|plusgiro)\b[\s:_-]*")
TRANSFER = re.compile(r"^(?:overforing|overf|ovf|intern overforing|transfer)\b[\s:_-]*")
DIRECT = re.compile(r"^autogiro\b[\s:_-]*")
SAVINGS = re.compile(r"spar|buffert|framtid|semester|reserv")
NAME = re.compile(r"^[a-z]+\s+[a-z]$")
CURRENCY = re.compile(r"\b(?:eur|usd|gbp|dkk|nok|pln|try|sek)\s*\d+[.,]?\d*\b")

def fold_text(text):
    cleaned = unicodedata.normalize("NFKD", text.casefold().replace("’", "'")).encode("ascii", "ignore").decode()
    return " ".join(cleaned.split())


def split_text(text):
    raw = fold_text(text)
    for rail, pattern in (("refund", REFUND), ("card", CARD), ("swish", SWISH), ("giro", GIRO),
                          ("transfer", TRANSFER), ("direct", DIRECT)):
        prefix = pattern.match(raw)
        if prefix:
            body = raw[prefix.end():]
            if rail in {"card", "refund"}:
                body = DATE.sub("", CARD.sub("", body))
            if rail == "swish" and NAME.fullmatch(body):
                body = "privateperson"
            return rail, WRAPPER.sub("", body), raw
    return "bare", WRAPPER.sub("", raw), raw


def features(row):
    rail, body, raw = split_text(row["text"])
    amount = float(row["amount_sek"])
    direction = "in" if amount > 0 else "out" if amount < 0 else "zero"
    account = row["account_type"]
    amount_band = "small" if abs(amount) < 100 else "middle" if abs(amount) < 1500 else "large"
    saving = bool(SAVINGS.search(fold_text(row["account_name"])))
    def clean(value):
        return " ".join(re.findall(r"[a-z]+|#", DIGITS.sub("#", CURRENCY.sub("", value))))
    return clean(body) or "empty", clean(raw) or "empty", {
        "rail": rail, "direction": direction, "account": account, "rail_direction": f"{rail}_{direction}",
        "account_direction": f"{account}_{direction}", "account_rail": f"{account}_{rail}",
        "saving_account": str(saving), "amount_band": amount_band,
    }
