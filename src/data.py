import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import *

def find_path(datadir, filename):
    direct = os.path.join(datadir, filename)
    if os.path.exists(direct):
        return direct
    import glob
    matches = glob.glob(os.path.join(datadir, f"*{filename}"))
    if matches:
        return matches[0]
    return direct

def load(datadir):
    tr = pd.read_csv(find_path(datadir, "train.csv"))
    te = pd.read_csv(find_path(datadir, "test_unlabelled.csv"))
    p = pd.read_csv(find_path(datadir, "partners.csv"))
    pr = pd.read_csv(find_path(datadir, "products.csv"))
    for d in (tr, te): d["ts"] = pd.to_datetime(d.submitted_at)
    raw_n = len(tr)
    tr = tr.sort_values("ts").drop_duplicates("claim_id", keep="first").reset_index(drop=True)   # re-submissions
    return tr, te, p, pr, raw_n

def train_table(tr, p, pr):
    """Leakage-safe training features: each row sees the partner snapshot taken at the start of its month."""
    lab = tr[tr.is_fraud.notna()]
    parts = []
    months = pd.period_range(tr.ts.min(), tr.ts.max(), freq="M")
    for m in months:
        rows = tr[(tr.ts >= m.start_time) & (tr.ts <= m.end_time)]
        if rows.empty: continue
        snap = build_snapshot(tr, lab, m.start_time)
        X, d = make_features(rows.drop(columns="ts"), p, pr, snap)
        X.index = rows.index
        parts.append(X)
    return pd.concat(parts).loc[tr.index]

def frozen_features(rows, tr, p, pr, asof):
    lab = tr[tr.is_fraud.notna()]
    snap = build_snapshot(tr[tr.ts < asof], lab[lab.ts < asof], asof)
    X, d = make_features(rows.drop(columns="ts"), p, pr, snap)
    X.index = rows.index
    return X
