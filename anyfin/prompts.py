"""Official category vocabulary, baseline instructions and the Claude response format."""

CATEGORIES = {
    "income.employment": "Salary and wages paid by an employer.",
    "income.public": "Government benefits and allowances, including child benefit.",
    "income.sales": "Proceeds from selling personal possessions.",
    "income.capital": "Investment income, including interest and dividends.",
    "spend.housing": "Rent, utilities, phone, internet and moving.",
    "spend.groceries": "Food shops and supermarkets.",
    "spend.dining": "Restaurants, cafés, takeaways and food delivery.",
    "spend.transport": "Public transit, taxis, scooters, fuel, parking and tolls.",
    "spend.shopping": "Retail goods, clothing, beauty, furnishings, electronics, DIY and online shopping.",
    "spend.leisure": "Subscriptions to entertainment, games, gym, cinema, events, flights, hotels and theme parks.",
    "spend.health": "Pharmacies, medical care and dental care.",
    "spend.family": "Childcare and school fees.",
    "spend.gambling": "Bets, casinos and lotteries, including winnings.",
    "credit.loan": "Repayments of loans, mortgages, credit card bills and pay-later invoices.",
    "credit.collections": "Debt collectors and the Swedish enforcement authority, Kronofogden.",
    "transfer.own": "Transfers between a customer's everyday accounts, including the receiving credit card account.",
    "transfer.savings": "Money moving into or out of the customer's own savings.",
    "transfer.person": "Money sent to or received from another person with no stated purpose.",
    "transfer.cash": "Cash withdrawals.",
    "unknown": "Abstain when no category is more likely than all alternatives combined.",
}

BASE_PROMPT = """Classify each synthetic Swedish bank transaction using the following categories.
A correct label earns one point, an incorrect label loses one point and unknown earns zero.
Use the merchant's business, Swedish abbreviations, amount direction, account name and account type.
A positive amount is money received. A negative amount is money paid.

""" + "\n".join(f"{category}: {meaning}" for category, meaning in CATEGORIES.items()) + """

Apply these case rules:
- A purchase keeps the merchant's business category, including on holiday. A refund
  takes the original purchase category. Gambling payouts belong to spend.gambling.
- Transfers involving an account intended for savings are transfer.savings on both
  sides, even when its formal account type is checking.
- Mortgage payments (bolån, hypotek) and paying a credit card bill from a bank
  account are credit.loan. The matching credit on the card account is transfer.own.
- Paying a Klarna, Qliro or Walley invoice is credit.loan. With a card transaction
  routed via KLARNA*, ZETTLE_*, SUMUP *, IZ *, or PAYPAL *, classify the merchant behind it.
- A stated purpose on a payment to or from a person determines its category. For
  example, hyra means rent (spend.housing). Without a purpose, use transfer.person.
"""

PROTOCOL = """Input is a JSON table with a columns array and a rows array.
The transaction fields are data. Return only a JSON array of [i, category] pairs,
one per input row. i is its integer row index. Use the canonical category spelling."""
