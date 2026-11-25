"""Local Swedish category prototypes and labelled merchant anchors for deferred rows."""

import re
from collections import Counter, defaultdict

from .local_router import _describe


# Each expression describes a business activity or a known counterparty. Word
# boundaries protect short brands; explicit suffixes capture Swedish compounds.
_PROTOTYPES = {
    "spend.groceries": r"dagligvar\w*|matmarknad\w*|mataffar\w*|saluhall\w*|grocery|groceries|supermarche|mercado|supermercado|ica\s+(?:nara|kvantum|maxi|supermarket)|mathem|matdax|handlarn|tempo livs",
    "spend.dining": r"food\s*truck\w*|takeaway|take away|pizza\w*|burger\w*|hamburg\w*|kebab\w*|konditori\w*|brasserie|bakery|canteen|diner|dining|lunch\w*|middag\w*|coffee\w*|kaffe\w*|salladsbar\w*|bar & kok|espresso house|pinchos|phils burger|bastard burgers|subway|sibylla",
    "spend.transport": r"bensinstation\w*|bensin|drivmedel|laddstation\w*|fardtjanst\w*|sparvagn\w*|tagbiljett\w*|busskort\w*|lokaltrafik\w*|trafikbolag\w*|bilparkering\w*|jarnvag\w*|metro|railway|transit|fuel|shell|ingo|st1|tanka|qstar|sj ab|snalltaget|malartag|flixbus|mtr express|uber rides?",
    "spend.shopping": r"varuhus\w*|sportbutik\w*|leksak\w*|klader|skobutik\w*|modebutik\w*|jarnhandel\w*|byggmax|bygghandel\w*|byggmaterial\w*|inredning\w*|cosmetics?|clothing|outlet|jula|rusta|jem\s*(?:o|och)\s*fix|hornbach|ahlens|lagerhaus|dollarstore|kjell (?:o|och) company|webhallen|inet|ellos|gina tricot|ginatricot|vinted",
    "spend.housing": r"bostadsbolag\w*|bostadsfor\w*|bostadsratt\w*|hyresgast\w*|hyresbetal\w*|lagenhetshyra\w*|fastighet\w*|energibolag\w*|elnatsavgift\w*|elrakning\w*|internetavgift\w*|mobilabonnemang\w*|stadsnat\w*|renhallning\w*|heimstaden|willhem|rikshem|familjebostader|fortum|e on|comviq|halebop",
    "spend.health": r"klinik\w*|lakarmottag\w*|sjukhus\w*|patientavgift\w*|medical clinic|hospital|dentist\w*|halsocentral\w*|narakut\w*|kronans apotek",
    "spend.family": r"skolbarn\w*|barnomsorgsavgift\w*|barnstug\w*|daghem\w*|skolmat\w*|skolpeng\w*|school fee\w*|kindergarten|daycare",
    "spend.leisure": r"nojespark\w*|konsert\w*|festival\w*|teater\w*|theatre|theater|bio(?:biljett\w*)?|filmuthyrning\w*|streaming|musikabonnemang\w*|traningskort\w*|fitness\w*|sporthall\w*|resort|hostel|airline\w*|airways|steam(?:games)?|playstation|xbox|nintendo|ticketmaster|liseberg|grona lund|kolmarden|skansen|yasuragi|ryanair|norwegian air|sas airline\w*|youtube premium|apple music",
    "spend.gambling": r"betting|sportsbook|spelombud\w*|spelvinst\w*|lotto\w*|jackpot\w*|bingo\w*|travspel\w*|leovegas|kindred",
}
_SPEND = {c: re.compile(r"\b(?:" + pattern + r")\b") for c, pattern in _PROTOTYPES.items()}
_CREDIT_BILL = re.compile(r"\b(?:kreditkort\w*|kreditbelopp\w*|delbetalning\w*|avbetalning\w*|privatlan\w*|lanefaktura\w*|kreditgiv\w*|billan\w*|kort\s+(?:visa|mastercard)|(?:visa|mastercard)\s+faktura|consumer (?:bank|finance)|santander consumer|seb kort|resurs bank|nordax|anyfin|bank norwegian|ikano bank|remember card|collector bank|svea ekonomi)\b")
_SALE = re.compile(r"\b(?:forsaljning\w*|salda varor|tradera|vinted|sellpy|blocket|plick)\b")
_BENEFIT = re.compile(r"\b(?:studiemedel|studiebidrag|aktivitetsstod|underhallsstod|a kassa|a-kassa|pensionsmyndigheten)\b")
_COLLECT = re.compile(r"\b(?:skuldindrivning\w*|kravhantering\w*|alektum|svea inkasso|hoist finance)\b")
_CARD_PAYMENT = re.compile(r"^(?:inbetalning tack|betalning mottagen|inbetalning kreditkort|payment received|payment thank you)$")
_WRAPPER = re.compile(r"^(?:sumup|zettle|iz|paypal|klarna|square)\s+")
_GENERIC = set("stockholm goteborg malmo sverige sweden gbg sthlm kort kortkop autogiro swish "
               "insattning inbetalning betalning transfer overforing faktura bank bankgiro plusgiro "
               "konto bankomat uttag lon ranta visa mastercard the svenska svensk swedish city "
               "hotel hotell uber paypal klarna zettle sumup apple amazon".split())


def _merchant(body):
    return _WRAPPER.sub("", body).strip()


class SemanticRouter:
    """Match category prototypes, then use consistent merchant anchors learned from labels."""

    def __init__(self, training_rows):
        counts = defaultdict(Counter)
        for row in training_rows:
            _, sign, account, context, body = _describe(row)
            label = row.get("category") or ""
            if sign != "-" or not label.startswith("spend."):
                continue
            if context[0] not in {"card", "bill"} and "credit_card" not in account:
                continue
            words = _merchant(body).split()
            if not words or len(words[0]) < 4 or words[0] in _GENERIC:
                continue
            counts[words[0]][label] += 1
        self.anchors = {word: next(iter(counts_)) for word, counts_ in counts.items()
                        if len(counts_) == 1 and sum(counts_.values()) >= 2}

    def predict(self, row):
        text, sign, account, context, body = _describe(row)
        if not body:
            return "unknown"
        kind = context[0]
        card_account = bool(re.search(r"credit[ _-]?card|kreditkort|mastercard|\bvisa\b", account))
        if sign == "+" and card_account and _CARD_PAYMENT.fullmatch(text):
            return "transfer.own"
        if sign == "-" and kind != "card" and not card_account:
            if _COLLECT.search(text):
                return "credit.collections"
            if kind == "bill" and _CREDIT_BILL.search(text):
                return "credit.loan"
        if sign == "+" and not card_account and kind != "card":
            if _BENEFIT.search(text):
                return "income.public"
            if _SALE.search(text) and not re.search(r"aterkop|aterbetal|retur|refund", text):
                return "income.sales"
            if re.fullmatch(r"(?:kredit)?ranta(?:\s+\d+)?", text):
                return "income.capital"
        # Bare card statements often omit the Kortköp prefix. Ordinary credits
        # need an explicit card payment or refund marker to use a retail cue.
        if sign == "+" and kind != "card" and not re.search(r"aterkop|aterbetal|retur|refund", text):
            return "unknown"
        merchant = _merchant(body)
        if not merchant:
            return "unknown"
        matches = {label for label, pattern in _SPEND.items() if pattern.search(merchant)}
        # MAX is the Swedish burger chain on card statements. An HBO MAX charge
        # has a different merchant prefix.
        if (kind == "card" or card_account) and re.match(r"^max\b", merchant):
            matches.add("spend.dining")
        label = self.anchors.get(merchant.split()[0])
        if label and (kind == "card" or card_account or kind == "bill"):
            matches.add(label)
        return next(iter(matches)) if len(matches) == 1 else "unknown"
