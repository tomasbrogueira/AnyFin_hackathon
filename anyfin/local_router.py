"""Selective on-device routing for Swedish transaction descriptors."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache


@lru_cache(maxsize=16384)
def _fold(text):
    folded = text.casefold()
    if folded.isascii():
        return folded
    return "".join(char for char in unicodedata.normalize("NFKD", folded)
                   if not unicodedata.combining(char))


_CARD = re.compile(r"^(?:kortkop|kortbetalning|varukop|kort(?:transaktion)?|debitkort|visa|mastercard|card purchase)\b[\s:*_-]*")
_SWISH = re.compile(r"^swish\b[\s:*_-]*(?:(?:inbetalning|utbetalning|betalning|mottagen|skickad|till|fran|in|ut)\b[\s:*_-]*)?")
_BILL = re.compile(r"^(?:autogiro|betalning\s*(?:bg|pg)?|bankgiro|plusgiro|bg|pg)\b\s*")
_DEPOSIT = re.compile(r"^(?:insattning|utbetalning|inbetalning)\b\s*")
_WRAPPER = re.compile(r"^(?:klarna|zettle|sumup|iz|paypal)[\s_*]+(?=\w)")
_TOKENS = re.compile(r"[a-z0-9]+")
_CURRENCY = re.compile(r"\s+(?:eur|usd|gbp|dkk|nok|sek)\s+[\d.,]+\s*$")
_LEADING_DATE = re.compile(r"^\d{6,8}\s+")

# Swedish business terms and common, single-purpose chains. Keep intermediaries
# and multi-service brands out: the underlying merchant or activity matters.
_MERCHANT_CUES = {
    label: re.compile(r"\b(?:" + terms + r")\b")
    for label, terms in {
        "spend.groceries": r"livsmedel|livs|matbutik|supermarket|superstore|stormarknad|hemkop|willys|lidl|city gross|coop|ica (?:maxi|nara|kvantum|supermarket)",
        "spend.dining": r"restaurang\w*|restaurant\w*|pizzeria\w*|cafe\w*|kafe\w*|bageri\w*|sushi|bistro|grill|krog|pizzahut|mcdonalds|burger king|foodora|wolt|uber\W+eats",
        "spend.shopping": r"bokhandel\w*|kladbutik\w*|elektronik\w*|byggvar\w*|mobel\w*|ikea|jysk|clas ohlson|elgiganten|zalando|kappahl|lindex|stadium|intersport|biltema|bauhaus",
        "spend.health": r"apoteks?\w*|pharmacy|tandlak\w*|tandvard\w*|vardcentral\w*|sjukvard\w*|dental\w*|folktandvarden",
        "spend.transport": r"parkering\w*|parking|kollektivtrafik\w*|bussbiljett\w*|taxi\w*|bensin\w*|drivmedel\w*|trangselskatt\w*|skanetrafiken|vasttrafik|easypark|parkster|circle k|okq8|preem|uber\W+(?:trip|ride)",
        "spend.housing": r"hyra|hyran|bostadsavgift\w*|bostader|hyresvard\w*|elnat\w*|elhandel\w*|fjarrvarme\w*|bredband\w*|flytthjalp\w*|ellevio|vattenfall|telia|telenor|tele2|bahnhof|comhem",
        "spend.family": r"forskola\w*|fritidshem\w*|skolavgift\w*|barnomsorg\w*",
        "spend.leisure": r"cinema\w*|biograf\w*|filmstaden|museum\w*|museet|hotell?\w*|flygbolag\w*|gym|gymkort|spotify|netflix|viaplay|disney\+?|hbo max|nordic wellness|friskis",
        "spend.gambling": r"casino\w*|kasino\w*|lotteri\w*|svenska spel|atg|betsson|unibet|bet365",
    }.items()
}


@lru_cache(maxsize=2048)
def _parse_text(text):
    for prefix, kind in ((_CARD, "card"), (_SWISH, "swish"), (_BILL, "bill"), (_DEPOSIT, "deposit")):
        match = prefix.match(text)
        if match:
            body = text[match.end():].lstrip(" :-")
            break
    else:
        body, kind = text, "other"
    body = _LEADING_DATE.sub("", body)
    if kind == "card":
        body = _WRAPPER.sub("", body)
    tokens = [token for token in _TOKENS.findall(_CURRENCY.sub("", body)) if not token.isdecimal()]
    return kind, " ".join(tokens)


def _describe(row):
    text = _fold(row.get("text") or "").strip()
    try:
        amount = float(row.get("amount_sek"))
    except (TypeError, ValueError):
        amount = math.nan
    if not math.isfinite(amount) or amount == 0:
        return text, "0" if amount == 0 else "", "", ("invalid",), ""
    direction = "+" if amount > 0 else "-"
    account = _fold((row.get("account_name") or "") + " " + (row.get("account_type") or ""))
    kind, body = _parse_text(text)
    if kind in {"card", "swish", "bill", "deposit"}:
        context = (kind, "" if kind == "card" else direction)
    else:
        # Generic transfer text has different meanings on a savings or credit account.
        context = (kind, direction, account)
    return text, direction, account, context, body


def _rules(text, sign, account, kind, body):
    if sign == "-" and re.match(r"^(?:uttag\b.*\bbankomat|bankomatuttag\b|kontantuttag\b|atm withdrawal\b)", text):
        return "transfer.cash"
    if sign == "+" and kind != "card":
        if re.search(r"\b(?:lon|loneutbetalning|manadslon|salary)\b", text):
            return "income.employment"
        if re.search(r"\b(?:barnbidrag|foraldrapenning|sjukpenning|bostadsbidrag|forsakringskassan)\b", text):
            return "income.public"
        if re.search(r"\b(?:ranteinkomst|insattningsranta|sparranta|utdelning|dividend)\b", text):
            return "income.capital"
    if sign == "-" and kind != "card":
        if re.search(r"\b(?:inkasso\w*|kronofogden|intrum|lowell)\b", text):
            return "credit.collections"
        if re.search(r"\b(?:bolan\w*|hypotek\w*|amortering\w*|lanebetalning\w*|kortfaktura\w*|kreditkortsfaktura\w*)\b", text):
            return "credit.loan"
    if kind == "bill" and sign == "-" and re.search(r"\b(?:klarna|qliro|walley|csn)\b", body):
        return "credit.loan"
    if re.search(r"\b(?:overforing|ovf|transfer)\b", text):
        if re.search(r"\b(?:sparkonto\w*|sparande\w*|buffert\w*|savings)\b", text + " " + account):
            return "transfer.savings"
    if kind == "swish" and re.fullmatch(r"[a-z]+ [a-z]", body):
        return "transfer.person"
    if kind != "deposit" or sign == "-":
        matches = [label for label, cue in _MERCHANT_CUES.items() if cue.search(text)]
        if len(matches) == 1:
            return matches[0]
    return None


@lru_cache(maxsize=4096)
def _word_grams(token):
    padded = "^" + token + "$"
    return tuple(padded[i:i + 3] for i in range(len(padded) - 2))


def _grams(body):
    grams = set()
    for token in body.split():
        if token in {"ab", "se", "com"}:
            continue
        grams.update(_word_grams(token))
    return grams


class LocalRouter:
    """Learn labelled descriptions and defer ambiguous or unfamiliar rows."""

    def __init__(self, training_rows):
        self.exact = defaultdict(set)
        for row in training_rows:
            _, _, _, context, body = _describe(row)
            label = row.get("category")
            if not label:
                raise ValueError("Local router training rows must have category labels.")
            if not body:
                continue
            self.exact[context, body].add(label)
        self.index = defaultdict(list)
        self.weight = {}
        gram_counts = Counter()
        for context, body in self.exact:
            gram_counts.update(_grams(body))
        self.idf = {gram: 1 + math.log((1 + len(self.exact)) / (1 + count))
                    for gram, count in gram_counts.items()}
        self.novel_weight = 1 + math.log(1 + len(self.exact))
        for key, labels in self.exact.items():
            context, body = key
            if context[0] not in {"card", "bill"} or not all(c.startswith("spend.") for c in labels):
                continue
            grams = _grams(body)
            self.weight[key] = sum(self.idf[g] ** 2 for g in grams)
            for gram in grams:
                self.index[gram].append(key)

    def predict(self, row):
        text, sign, account, context, body = _describe(row)
        if not body:
            return "unknown"
        labels = self.exact.get((context, body))
        if labels:
            return next(iter(labels)) if len(labels) == 1 else "unknown"
        rule = _rules(text, sign, account, context[0], body)
        if rule:
            return rule
        if context[0] == "deposit" and sign == "+":
            return "unknown"
        # Compare merchant descriptions across banks. The asymmetric overlap
        # permits extra bank text while penalizing evidence absent from training.
        overlap = Counter()
        grams = _grams(body)
        query_weight = sum(self.idf.get(g, self.novel_weight) ** 2 for g in grams)
        for gram in grams:
            for key in self.index.get(gram, ()):
                overlap[key] += self.idf[gram] ** 2
        scores = Counter()
        for key, common in overlap.items():
            recall = common / self.weight[key]
            precision = common / query_weight
            score = math.sqrt(recall * (0.5 + 0.5 * precision))
            for label in self.exact[key]:
                scores[label] = max(scores[label], score)
        best = scores.most_common(2)
        if best and best[0][1] >= 0.80 and (len(best) == 1 or best[0][1] - best[1][1] >= 0.15):
            return best[0][0]
        return "unknown"
