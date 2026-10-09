# Kestrel Home – warranty claim fraud screen

Ranks warranty claims for review **before payout**. One JSON endpoint, one screen. No API key, no internet, no paid service.

## Run it (clean machine, Python 3.11+)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/server.py                                    # -> http://127.0.0.1:8000
```
Open http://127.0.0.1:8000, pick an example (all five are made-up claims), press **Check this claim**.
Versions are pinned because the saved model is a scikit-learn pickle and is version-sensitive.
If the model file is missing/unloadable the service still starts; the screen and `/score` explain what to do (HTTP 503) instead of crashing.

## Endpoint
`POST /score` with one claim as JSON (fields as in `test_unlabelled.csv`; `submitted_at` and `inspector_note` optional):
```bash
curl -s localhost:8000/score -H 'Content-Type: application/json' -d '{
 "claim_id":"DEMO-001","submitted_at":"2026-07-14 11:20","partner_id":"SP3129","sku":"KH-AF-02",
 "product_serial":" kh100200300","days_since_purchase":131,"claim_amount_inr":1890,"photo_attached":"Y",
 "partner_inspected":"N","claim_description":"unit not heating","inspector_note":"","customer_prior_claims":2}'
```
Returns `score` (0-1, higher = riskier; this is what `predictions.csv` holds), `est_fraud_probability`, `recommendation`
(REVIEW BEFORE PAYING / PAY NORMALLY), `expected_value_of_review_inr` (p x claim - (1-p) x Rs 380 goodwill), four plain-language `reasons`,
`notes`, the partner's investigated record, and a caveat. Bad input gets a 400 with a sentence a person can act on. `GET /health` shows model status.

## Rebuild the model (needs the Kestrel data pack; not included - see Data handling)
```bash
mkdir data   # put train.csv, test_unlabelled.csv, partners.csv, products.csv, sample_submission.csv here
python src/train.py        # writes artifacts/, predictions.csv, evidence/review_queue_top120.csv
python src/backtest.py     # writes evidence/backtest*.csv
```

## Layout
`src/features.py` features (shared by training and service) · `src/data.py` loading, de-dup, leakage-safe training table ·
`src/modeling.py` logistic + gradient-boosting blend · `src/backtest.py` forward-in-time evidence · `src/train.py` final fit + predictions ·
`app/` Flask service + screen · `artifacts/` model, partner history snapshot (as of 1 Jul 2026), reference tables · `evidence/` results.

## Data handling
Ops policy section 10: customer/operational data must not be published or shared beyond the engagement team. The raw data pack is **not** in this bundle
(`data/` is git-ignored). `artifacts/` holds partner-level aggregates only and `predictions.csv` holds claim IDs and scores. **Keep any repo private.**

## Limits
Built on 141 confirmed frauds. History is frozen at 1 Jul 2026; the screen does not update itself. It ranks claims for a human; it does not prove fraud.
