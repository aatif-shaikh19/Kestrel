import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import *
from modeling import *
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("DATA", os.path.join(ROOT, "data"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "evidence"))
os.makedirs(OUT, exist_ok=True)
tr, te, p, pr, raw_n = load(D)
XT = train_table(tr, p, pr)
lab_mask = tr.is_fraud.notna()
GOODWILL = 380.0; CONTACT = 260.0
rs = np.random.RandomState(0)

def money(y, amt, score, k, extra=0.0):
    idx = np.argsort(-score, kind="stable")[:k]
    caught = y[idx] == 1
    stopped = amt[idx][caught].sum(); holds = int((~caught).sum())
    net = stopped - GOODWILL * holds - extra * k
    return dict(k=k, frauds_caught=int(caught.sum()), precision=float(caught.mean()), stopped_inr=float(stopped),
                genuine_held=holds, net_inr=float(net), net_per_checked=float(net / k), gross_per_checked=float(stopped / k),
                net_per_checked_with_contact=float((stopped - GOODWILL * holds - CONTACT * k) / k))

folds = [("A: Q1-26 (old regime)", "2025-10-01", "2026-01-01", "2026-04-01"),
         ("B: Apr-Jun 26 (shift lands mid-window)", "2025-10-01", "2026-04-01", "2026-07-01"),
         ("C: Jun 26 (new regime, trained thru May)", "2025-10-01", "2026-06-01", "2026-07-01")]
rows = []; keep = {}
for name, t0, v0, v1 in folds:
    t0, v0, v1 = map(pd.Timestamp, (t0, v0, v1))
    trm = lab_mask & (tr.ts >= t0) & (tr.ts < v0)
    vm = lab_mask & (tr.ts >= v0) & (tr.ts < v1)
    Xv = frozen_features(tr[vm], tr, p, pr, v0)
    yt = tr.is_fraud[trm].values; yv = tr.is_fraud[vm].values; av = tr.claim_amount_inr[vm].values
    k = 40 * max(1, round((v1 - v0).days / 30.4))
    M = fit_models(XT[trm], yt, tr.ts[trm])
    blend, p_lr, p_gb = predict(M, Xv)
    sc = {"predict-everything-legit": np.zeros(len(yv)), "random": rs.rand(len(yv)),
          "rule: partner's past fraud rate": Xv.p_fraud_rate_sm.values, "logistic": p_lr, "gradient boosting": p_gb, "FINAL: blend": blend}
    for mn, s in sc.items():
        r = dict(fold=name, model=mn, n=len(yv), frauds=int(yv.sum()), base_rate=float(yv.mean()),
                 accuracy=float(((s > 0.5) == yv).mean()) if mn != "predict-everything-legit" else float(1 - yv.mean()),
                 auc=float(roc_auc_score(yv, s)) if s.std() > 0 else 0.5, ap=float(average_precision_score(yv, s)))
        r.update(money(yv, av, s, k)); rows.append(r)
    keep[name] = (yv, av, blend, k)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/backtest.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R[["fold","model","frauds","accuracy","auc","ap","k","frauds_caught","precision","stopped_inr","genuine_held","net_per_checked","net_per_checked_with_contact"]].round(3).to_string())

# bootstrap uncertainty for the final blend
bs = []
for name, (yv, av, s, k) in keep.items():
    res = []
    for b in range(1000):
        i = rs.randint(0, len(yv), len(yv)); yy, aa, ss = yv[i], av[i], s[i]
        if yy.sum() == 0: continue
        kk = max(1, int(round(k * len(yv) / len(yv))))
        m = money(yy, aa, ss, k); res.append([roc_auc_score(yy, ss), average_precision_score(yy, ss), m["precision"], m["net_per_checked"], m["gross_per_checked"]])
    a = np.array(res); lo, hi = np.percentile(a, [5, 95], axis=0)
    bs.append(dict(fold=name, **{f"{n}_{t}": v for n, l, h in zip(["auc","ap","precision","net_per_checked","gross_per_checked"], lo, hi) for t, v in (("p5", l), ("p95", h))}))
pd.DataFrame(bs).to_csv(f"{OUT}/backtest_bootstrap90.csv", index=False)
print(pd.DataFrame(bs).round(2).T.to_string())
