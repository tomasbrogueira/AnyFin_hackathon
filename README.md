# Anyfin Case Night · Track A

- With Python 3.10 or newer, run `python -m experiments.specialist_fast predict final.csv --model results/independent_specialist/specialist-formats.json.gz --recurring-income -o tomas-brogueira.csv`. Inference uses the standard library and requires no API key.
- We match transfers and recurring payroll across accounts, apply Swedish payment and merchant rules, and use a small text classifier trained with synthetic bank formats for unresolved rows. Inference compute for 10 million rows is estimated at $0.02–$0.03 at $0.10 per CPU machine-hour, excluding training, storage and service costs.
- With another day, I’d label unfamiliar counterparties from more banks, use them to calibrate abstention, and test on a separate set of customers.
