"""Normalize payment descriptions and group named debits for merchant stress tests."""

import re
import unicodedata


def description(row):
    """A description group ignores capitalization, punctuation and varying digits."""
    folded = unicodedata.normalize("NFKC", row["text"]).casefold()
    return " ".join(re.findall(r"[^\W\d_]+|#", re.sub(r"\d+", "#", folded)))


def merchant_tokens(text):
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c)).replace("ø", "o")
    return re.findall(r"[a-z][a-z0-9]*(?:&[a-z0-9]+)*", folded)


def counterparty_token(row):
    """Parse debits starting with a named counterparty, optionally after a card prefix."""
    text = re.sub(r"^(?:kortköp|kortkop|kortbetalning|varuköp|korttransaktion|kort)\b\s*(?:\d{6,8}\s+)?",
                  "", row["text"], flags=re.I)
    text = re.sub(r"^(?:klarna|zettle|iz|sumup|paypal|trustly)[\s_*]*[*_][\s_*]*",
                  "", text, flags=re.I)
    words = merchant_tokens(text)
    if not words or float(row["amount_sek"]) >= 0:
        return None
    operations = {"autogiro", "betalning", "inbetalning", "bg", "pg", "swish", "overforing",
                  "uttag", "bankomat", "avgift", "ranta", "ocr", "faktura", "transfer",
                  "bankgiro", "plusgiro", "ovf", "kort", "insattning", "utbetalning"}
    return words[0] if words and len(words[0]) >= 2 and words[0] not in operations else None
