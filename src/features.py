"""Feature engineering shared by training, backtest and the live service.
Every feature is computable from ONE claim record + partners/products reference
+ a partner 'snapshot' table (history known at scoring time). No future info.
"""
import re
import numpy as np
import pandas as pd

POLICY_CHANGE = pd.Timestamp("2026-05-01")
SMALL_LIMIT = 2000
DESC_LIST = ["remote not working", "loud noise while running", "power button not working", "filter indicator stuck",
             "motor not running", "water leaking", "not charging", "blade jammed", "tripping mcb", "burning smell",
             "unit not heating", "display not working"]
NOTE_LEVELS = ["Customer has bill, serial verified", "PCB replaced under warranty", "Photos match fault, approved",
               "Heating element open circuit", "Minor fault, part swapped", "Motor winding failure confirmed",
               "Unit inspected, fault confirmed"]
PRIOR = 0.012
K = 20.0
SNAP_COLS = ["p_n_prior", "p_fraud_prior", "p_fraud_rate_sm", "p_vol_90d", "p_near_thr_share",
             "p_uninsp_share", "p_small_share", "p_fraud_recent", "p_su_n", "p_su_fraud", "p_su_rate_sm"]
LABEL_LAG = 14

_DESC_CLEAN = re.compile(r"[.;].*$")

def clean_desc(s):
    # partner-typed free text: keep only the leading fault phrase
    return _DESC_CLEAN.sub("", str(s)).strip().lower()

def norm_serial(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def build_snapshot(claims, labelled, asof, label_lag_days=None):
    label_lag_days = LABEL_LAG if label_lag_days is None else label_lag_days
    """Partner history as known on `asof`. Labels only count once the claim is older than
    label_lag_days (investigations take time); behaviour stats use the trailing 90 days."""
    asof = pd.Timestamp(asof)
    lab = labelled[labelled.ts < asof - pd.Timedelta(days=label_lag_days)]
    g = lab.groupby("partner_id").is_fraud.agg(["size", "sum"]).rename(columns={"size": "p_n_prior", "sum": "p_fraud_prior"})
    rec = claims[(claims.ts < asof) & (claims.ts >= asof - pd.Timedelta(days=90))]
    r = rec.groupby("partner_id").agg(
        p_vol_90d=("claim_id", "size"),
        p_near_thr_share=("claim_amount_inr", lambda s: ((s >= 1500) & (s < SMALL_LIMIT)).mean()),
        p_uninsp_share=("partner_inspected", lambda s: (s == "N").mean()),
        p_small_share=("claim_amount_inr", lambda s: (s < SMALL_LIMIT).mean()))
    rl = lab[lab.ts >= asof - pd.Timedelta(days=75)]
    g2 = rl.groupby("partner_id").is_fraud.sum().rename("p_fraud_recent")
    su = lab[(lab.claim_amount_inr < SMALL_LIMIT) & (lab.partner_inspected == "N")]
    g3 = su.groupby("partner_id").is_fraud.agg(["size", "sum"]).rename(columns={"size": "p_su_n", "sum": "p_su_fraud"})
    snap = g.join(r, how="outer").join(g2, how="left").join(g3, how="left")
    for c in ("p_fraud_recent", "p_su_n", "p_su_fraud"): snap[c] = snap[c].fillna(0)
    snap["p_su_rate_sm"] = (snap.p_su_fraud + 0.03 * 10) / (snap.p_su_n + 10)
    snap["p_n_prior"] = snap.p_n_prior.fillna(0)
    snap["p_fraud_prior"] = snap.p_fraud_prior.fillna(0)
    snap["p_fraud_rate_sm"] = (snap.p_fraud_prior + PRIOR * K) / (snap.p_n_prior + K)
    return snap

def make_features(df, partners, products, snap):
    d = df.copy()
    d["ts"] = pd.to_datetime(d["submitted_at"])
    d = d.merge(partners, on="partner_id", how="left").merge(products, on="sku", how="left")
    d["desc"] = d.claim_description.map(clean_desc)
    # NOTE: partner age/type deliberately NOT used - a partner is judged on its own claims record, not on being new (Service Desk ask).
    amt = d.claim_amount_inr.astype(float)
    X = pd.DataFrame(index=d.index)
    X["amount"] = amt
    X["amt_to_price"] = amt / d.list_price_inr
    X["days_since_purchase"] = d.days_since_purchase
    wd = d.warranty_months * 30.4
    X["days_over_warranty"] = d.days_since_purchase - wd
    X["out_of_warranty"] = (X.days_over_warranty > 0).astype(int)
    X["warranty_frac_used"] = d.days_since_purchase / wd
    X["photo"] = (d.photo_attached == "Y").astype(int)
    X["inspected"] = (d.partner_inspected == "Y").astype(int)
    X["note_present"] = d.inspector_note.notna().astype(int)
    X["insp_no_note"] = ((d.partner_inspected == "Y") & d.inspector_note.isna()).astype(int)
    X["note_idx"] = d.inspector_note.map({n: i for i, n in enumerate(NOTE_LEVELS)}).fillna(-1)
    X["prior_claims"] = d.customer_prior_claims
    X["small_claim"] = (amt < SMALL_LIMIT).astype(int)
    X["near_threshold"] = ((amt >= 1500) & (amt < SMALL_LIMIT)).astype(int)
    X["small_uninspected"] = ((amt < SMALL_LIMIT) & (d.partner_inspected == "N")).astype(int)
    fam = {f: i for i, f in enumerate(sorted(products.family.unique()))}
    X["family_code"] = d.family.map(fam)
    ps = d.product_serial.astype(str)
    X["serial_messy"] = (~((ps == ps.str.strip()) & (ps == ps.str.upper()) & (~ps.str.contains("-")))).astype(int)
    X["desc_code"] = d.desc.map({t: i for i, t in enumerate(DESC_LIST)}).fillna(-1)
    X["desc_odd"] = (~d.desc.isin(DESC_LIST)).astype(int)
    s = snap.reindex(d.partner_id)
    for c in SNAP_COLS:
        X[c] = s[c].values
    X["p_n_prior"] = X.p_n_prior.fillna(0)
    X["p_fraud_prior"] = X.p_fraud_prior.fillna(0)
    X["p_fraud_rate_sm"] = X.p_fraud_rate_sm.fillna(PRIOR)
    for c in ("p_fraud_recent", "p_su_n", "p_su_fraud"): X[c] = X[c].fillna(0)
    X["p_su_rate_sm"] = X.p_su_rate_sm.fillna(0.03)
    return X, d
