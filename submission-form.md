# Submission form – Kestrel Home warranty claim review (Variant C)
*The form template was not in the pack, so this follows the fields used in Task 1. If Banao supplies its own, paste these answers across.*
**Candidate:** Aatif Zahur Shaikh · **Repo / bundle:** private (data policy s10 – no raw data in it) · **Drive link:** [Insert Google Drive / Repo URL]

## 1. What I built, and the number
A claim-ranking screen for the investigation desk: a score per claim, a plain-language reason, and a "review before paying / pay normally" call, plus a partner-level finding. Recommendation: **do not ship a 97%-accuracy flag**; re-inspect 8 named outlets and use the model to pick the desk's 40 reviews a month.
**The number (Farhan's ask):** June back-test, 40 reviews → 14 of 22 frauds, **Rs 502 stopped per claim reviewed** (90% interval Rs 296-698), Rs 255 after goodwill (Rs -3 to 491), about break-even after the Rs 260 contact cost. For Jul-Sep I expect lower: **~Rs 350 per claim gross (Rs 100-600).**

## 2. What I expect predictions.csv to score (written before submitting)
Unknown metric, so by type. **AUC ≈ 0.80 (plausible 0.60-0.90). Average precision ≈ 0.25 (0.08-0.50). Precision in the top 120 ≈ 25-35%, i.e. ~30-40 frauds caught.** Basis: AUC 0.88 / AP 0.37 on June (one month after training, 22 frauds) and 0.90 on Q1; discounted because the test is 1-3 months past the history cut-off, the May change is only ~2 months old in training, and fold B shows a hard drop (0.48) when fraud patterns move. Accuracy at any sensible threshold will be ~96-98% and says nothing. Expected frauds in the test period: roughly 40-90.

## 3. Cost to run
No model API anywhere in the product: Rs 0 per claim, nothing to pay per month. Training takes about 7 seconds on this machine; scoring is milliseconds. Hosting a small Flask app is the only cost. Human cost is the desk's 40 reviews a month.

## 4. Validation: sample size and error rate
Forward-in-time back-tests, 3 folds, history frozen as in the real test. Only 141 confirmed frauds (22 / 45 / 22 per fold), so intervals are wide (see `evidence/EVIDENCE.md`). Error: 2 in 3 holds are genuine customers; one fold (policy change landing mid-window) is a failure (AUC 0.48). Accuracy was reported only to show that it is meaningless here.

## 5. Where I pushed back on / narrowed the ask
Declined accuracy as the KPI: "flag nothing" gets 96.9-99% (`backtest.csv`). Replaced with rupees per claim reviewed within the 40/month cap. Declined "newer partners are the problem" as a blanket statement: 38 of 46 new outlets are clean; model excludes outlet age and type on purpose. Recommended against going live automatically; pilot as a queue next to the desk.

## 6. Decisions I made where the pack was silent
Kept the first copy of re-submitted claims. Dropped blank (undecided) outcomes from training; could not separate undecided cases from legacy zeros. Labels only count after 14 days. Goodwill Rs 380 per genuine hold; Rs 260 treated as a contact cost and shown as a sensitivity, not baked in. Review capacity 40 x 3 months = 120. Post-May rows weighted 3x. Test history frozen at 30 Jun 2026. No time-of-day, serial, `source`, outlet-age or outlet-type features.

## 7. Known flaws
Small number of frauds. Breaks across policy changes; needs monthly retraining. New rings without history are caught only by claim shape. Legacy label noise. 14-day label lag is a guess. Partner snapshot is static in the service. Probability calibration is rough (fit on two folds). Reasons come from the logistic half of the blend, not the boosted trees. No authentication on the service.

## 8. What I left out
Customer-level linking (no customer ID in the pack, only prior-claim count), image/photo checks, a live retraining pipeline, user login, a feedback loop for desk outcomes, multi-claim ring detection by serial/customer, formal threshold optimisation per month.

## 9. Findings nobody asked for
Fraud moved from larger inspected claims at ~7 older outlets (median Rs 3,392, stopped after April) to small uninspected claims at a few new outlets (median Rs 1,416). Fraud on claims under Rs 2,000 rose from 0.2% to 3.2% when inspection was dropped. 8 outlets hold 79 of the 120 highest-risk test claims. A handful of descriptions carried extra appended text, handled by keeping only the fault phrase.

## 10. AI tools, honestly
Google Antigravity (Gemini 3.8 Flash) and Claude for data analysis, temporal feature engineering, back-testing, Flask API service, modern executive UI dashboard, executive memo, and documentation; I directed problem framing, policy alignment, and reviewed outputs. No external model API inside the product itself (Rs 0 runtime inference cost). **Development cost: standard assistant subscription (~Rs 1,600 / $20 / mo; zero API usage cost).** Discarded: random train/test split (leaks re-submissions, hides the May policy change), gradient boosting on its own, outlet-age/type features (which unjustly biased against legitimate new partners), a first version of reasons that listed raw model coefficients (confusing to claims desk agents), and unpinned dependencies.

## 11. Monday handover – three things
1. Put the 8 outlets back on mandatory inspection; desk works their open claims first.
2. Run the 40-a-month queue from the ranked list; log outcomes (including undecided) so the next retrain can learn.
3. Retrain monthly and after any change to the approval policy; report rupees per claim reviewed, not accuracy.

## 12. Honest hours
**Total time:** ~6.5 hours (Data EDA, policy parsing, leak-free temporal feature engineering & backtesting: 2.5h; Flask scoring API & modern web dashboard: 2.0h; Business memo, evidence report & packaging: 2.0h).
**Deliverable Links:** Drive link: [Insert Google Drive / repo URL] · Screen recording: [Insert Loom / Drive video link — see `recording-script.md`].
