"""Rebuilds everything: final model, partner snapshot, calibration, predictions.csv.
Usage: python src/train.py [data_dir]   (needs train.csv, test_unlabelled.csv, partners.csv, products.csv)"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import *
from modeling import *
from sklearn.linear_model import LogisticRegression

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts"); os.makedirs(ART, exist_ok=True)
tr, te, p, pr, raw_n = load(D)
lab = tr[tr.is_fraud.notna()]
print(f"train rows {raw_n} -> {len(tr)} after dropping {raw_n-len(tr)} re-submitted duplicates; {int(tr.is_fraud.isna().sum())} undecided excluded; {len(lab)} labelled, {int(lab.is_fraud.sum())} fraud")

XT = train_table(tr, p, pr)
M = fit_models(XT.loc[lab.index], lab.is_fraud.values, lab.ts)

ASOF = pd.Timestamp("2026-07-01")
snap = build_snapshot(tr, lab, ASOF)
Xte, dte = make_features(te.drop(columns="ts"), p, pr, snap)
blend, p_lr, p_gb = predict(M, Xte)

# calibration on out-of-time folds (A: Q1-26 and C: Jun-26), so the service can quote rupees
cal_s, cal_y = [], []
for t0, v0, v1 in [("2025-10-01", "2026-01-01", "2026-04-01"), ("2025-10-01", "2026-06-01", "2026-07-01")]:
    t0, v0, v1 = map(pd.Timestamp, (t0, v0, v1))
    trm = tr.is_fraud.notna() & (tr.ts >= t0) & (tr.ts < v0); vm = tr.is_fraud.notna() & (tr.ts >= v0) & (tr.ts < v1)
    Mx = fit_models(XT[trm], tr.is_fraud[trm].values, tr.ts[trm], n_seeds=3)
    s, _, _ = predict(Mx, frozen_features(tr[vm], tr, p, pr, v0)); cal_s += list(s); cal_y += list(tr.is_fraud[vm].values)
z = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
cal = LogisticRegression(C=100).fit(z(np.array(cal_s)).reshape(-1, 1), np.array(cal_y))
CAL = {"a": float(cal.coef_[0, 0]), "b": float(cal.intercept_[0])}

order = np.argsort(-blend, kind="stable"); cut = float(blend[order[119]])   # 120 = 40 per month x 3 months
months = max(1, round((te.ts.max() - te.ts.min()).days / 30.4))
meta = {"asof": str(ASOF.date()), "calibration": CAL, "queue_cutoff_score": cut, "monthly_capacity": 40,
        "goodwill_inr": 380.0, "contact_inr": 260.0, "label_lag_days": 14, "post_weight": POST_W,
        "n_train": int(len(lab)), "n_fraud": int(lab.is_fraud.sum()), "test_months": int(months)}
joblib.dump({"models": M, "meta": meta}, os.path.join(ART, "model.joblib"))
snap.to_csv(os.path.join(ART, "partner_snapshot.csv")); p.to_csv(os.path.join(ART, "partners.csv"), index=False); pr.to_csv(os.path.join(ART, "products.csv"), index=False)
json.dump(meta, open(os.path.join(ART, "meta.json"), "w"), indent=1)

sub = pd.DataFrame({"claim_id": te.claim_id, "score": np.round(blend, 6)})
sample = pd.read_csv(find_path(D, "sample_submission.csv"))
assert list(sub.claim_id) == list(sample.claim_id) and sub.claim_id.is_unique and len(sub) == len(te)
sub.to_csv(os.path.join(ROOT, "predictions.csv"), index=False)
print("predictions.csv written", sub.shape, "| score range", sub.score.min(), sub.score.max())
pt = 1 / (1 + np.exp(-(CAL["a"] * z(blend) + CAL["b"])))
print("calibrated expected frauds in test:", round(pt.sum(), 1), "of", len(te), "| top-120 expected frauds:", round(pt[order[:120]].sum(), 1))
te2 = te.assign(score=blend, p_cal=pt)
te2.sort_values("score", ascending=False).head(120).to_csv(os.path.join(ROOT, "evidence", "review_queue_top120.csv"), index=False)
