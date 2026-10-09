"""Kestrel warranty-claim fraud screen. Run: python app/server.py  ->  http://127.0.0.1:8000
No API key, no network calls. If the model file is missing the service still starts and says so politely."""
import os, sys, json, math, datetime as dt, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=os.path.join(ROOT, "app", "static"))
ART = os.path.join(ROOT, "artifacts")
STATE = {"ready": False, "error": None}

def load():
    try:
        import joblib
        from features import make_features, clean_desc, DESC_LIST
        b = joblib.load(os.path.join(ART, "model.joblib"))
        STATE.update(models=b["models"], meta=b["meta"], ready=True,
                     snap=pd.read_csv(os.path.join(ART, "partner_snapshot.csv"), index_col=0),
                     partners=pd.read_csv(os.path.join(ART, "partners.csv")), products=pd.read_csv(os.path.join(ART, "products.csv")))
    except Exception as e:           # start anyway, fail politely
        STATE["error"] = f"{type(e).__name__}: {e}"
load()

FIELDS = {"claim_id": str, "partner_id": str, "sku": str, "product_serial": str, "days_since_purchase": float,
          "claim_amount_inr": float, "photo_attached": str, "partner_inspected": str, "claim_description": str,
          "customer_prior_claims": float}
OPTIONAL = {"submitted_at": str, "inspector_note": str}

class Bad(Exception): pass

def parse(rec):
    if not isinstance(rec, dict): raise Bad("Send one claim as a JSON object.")
    out, missing = {}, [f for f in FIELDS if rec.get(f) in (None, "")]
    if missing: raise Bad("Missing field(s): " + ", ".join(missing))
    for f, t in FIELDS.items():
        try: out[f] = t(rec[f]) if t is not str else str(rec[f])
        except Exception: raise Bad(f"'{f}' should be a {'number' if t is float else 'text'}.")
    for f in ("photo_attached", "partner_inspected"):
        out[f] = out[f].strip().upper()[:1]
        if out[f] not in ("Y", "N"): raise Bad(f"'{f}' must be Y or N.")
    if out["claim_amount_inr"] <= 0: raise Bad("'claim_amount_inr' must be above 0.")
    if out["days_since_purchase"] < 0: raise Bad("'days_since_purchase' cannot be negative.")
    out["partner_id"] = out["partner_id"].strip().upper(); out["sku"] = out["sku"].strip().upper()
    note = rec.get("inspector_note"); out["inspector_note"] = note if isinstance(note, str) and note.strip() else np.nan
    out["submitted_at"] = str(rec.get("submitted_at") or dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    try: pd.to_datetime(out["submitted_at"])
    except Exception: raise Bad("'submitted_at' should look like 2026-07-01 14:30.")
    if out["partner_id"] not in set(STATE["partners"].partner_id): raise Bad(f"Unknown partner_id {out['partner_id']}.")
    if out["sku"] not in set(STATE["products"].sku): raise Bad(f"Unknown sku {out['sku']}.")
    return out

TEXT = {
 "small_uninspected": "Small claim (under Rs 2,000) paid without inspection - the route the May-2026 policy opened",
 "near_threshold": "Amount sits just under the Rs 2,000 auto-approve line",
 "inspected": "Partner inspection sign-off present",
 "photo": "Photo attached", "note_present": "Inspector wrote a note",
 "prior_claims": "Customer's earlier warranty claims", "days_since_purchase": "Days since purchase",
 "amount": "Claim amount", "amt_to_price": "Claim as a share of the product's list price",
 "p_fraud_rate_sm": "Partner's confirmed-fraud rate in past investigated claims",
 "p_fraud_prior": "Confirmed frauds at this partner so far", "p_fraud_recent": "Confirmed frauds at this partner in the last ~75 days",
 "p_su_rate_sm": "Fraud rate on this partner's small uninspected claims", "p_su_fraud": "Confirmed frauds on this partner's small uninspected claims",
 "p_near_thr_share": "Share of this partner's recent claims priced just under Rs 2,000",
 "p_uninsp_share": "Share of this partner's recent claims with no inspection", "p_small_share": "Share of this partner's recent claims under Rs 2,000",
 "p_vol_90d": "Partner's claim volume in the last 90 days", "p_n_prior": "Investigated claims on record for this partner",
 "p_su_n": "Investigated small-uninspected claims for this partner", "warranty_frac_used": "Share of warranty period used",
 "days_over_warranty": "Days past warranty", "out_of_warranty": "Out of warranty", "small_claim": "Claim under Rs 2,000",
 "note_idx": "Type of inspector note", "insp_no_note": "Inspected but no note written", "serial_messy": "Serial typed untidily",
 "desc_code": "Fault type", "desc_odd": "Fault text had extra wording (ignored)", "family_code": "Product family"}

GROUPS = {
 "Partner track record": ["p_fraud_rate_sm", "p_fraud_prior", "p_fraud_recent", "p_su_rate_sm", "p_su_fraud", "p_su_n", "p_n_prior"],
 "How this partner has been submitting lately": ["p_vol_90d", "p_near_thr_share", "p_uninsp_share", "p_small_share"],
 "How the claim was routed": ["small_uninspected", "near_threshold", "inspected", "small_claim", "photo", "note_present", "insp_no_note", "note_idx"],
 "The claim and customer": ["amount", "amt_to_price", "days_since_purchase", "warranty_frac_used", "days_over_warranty", "out_of_warranty",
                            "prior_claims", "desc_code", "desc_odd", "family_code", "serial_messy"]}

def facts(group, X, r):
    g = lambda c: X[c].iloc[0]
    if group == "Partner track record":
        n, f, su_n, su_f = int(g("p_n_prior")), int(g("p_fraud_prior")), int(g("p_su_n")), int(g("p_su_fraud"))
        if n == 0: return f"{r['partner_id']} has no investigated claims on record yet."
        t = f"{r['partner_id']}: {f} confirmed fraud(s) in {n} investigated claims"
        if su_n: t += f"; {su_f} of {su_n} investigated small uninspected claims were fraud"
        return t + "."
    if group == "How this partner has been submitting lately":
        v = int(g("p_vol_90d")) if pd.notna(g("p_vol_90d")) else 0
        if not v: return "No recent claims from this partner to compare against."
        return (f"Last 90 days: {v} claims, {g('p_small_share')*100:.0f}% under Rs 2,000, {g('p_near_thr_share')*100:.0f}% priced Rs 1,500-1,999, "
                f"{g('p_uninsp_share')*100:.0f}% without inspection.")
    if group == "How the claim was routed":
        bits = [("small and uninspected - would be auto-approved today" if g("small_uninspected") else ("inspected" if g("inspected") else "uninspected")),
                ("priced just under the Rs 2,000 line" if g("near_threshold") else None),
                ("photo attached" if g("photo") else "no photo")]
        return ", ".join(b for b in bits if b) + "."
    return (f"Rs {r['claim_amount_inr']:,.0f} claimed {int(r['days_since_purchase'])} days after purchase "
            f"({g('warranty_frac_used')*100:.0f}% of warranty used); customer has {int(r['customer_prior_claims'])} earlier claim(s).")

def explain(X, r):
    lr = STATE["models"]["lr"]; sc, clf = lr.steps[0][1], lr.steps[1][1]
    cols = STATE["models"]["cols"]
    z = (X[cols].fillna(0).values - sc.mean_) / sc.scale_
    contrib = dict(zip(cols, z[0] * clf.coef_[0]))
    out = []
    for grp, members in GROUPS.items():
        net = sum(contrib.get(c, 0.0) for c in members)
        out.append((abs(net), {"area": grp, "direction": "raises risk" if net > 0.05 else ("lowers risk" if net < -0.05 else "neutral"),
                               "strength": "strong" if abs(net) > 2 else ("moderate" if abs(net) > 0.7 else "slight"),
                               "reason": facts(grp, X, r)}))
    return [o for _, o in sorted(out, key=lambda t: -t[0])]

def score_record(rec):
    from features import make_features
    r = parse(rec); df = pd.DataFrame([r])
    X, d = make_features(df, STATE["partners"], STATE["products"], STATE["snap"])
    from modeling import predict
    s, p_lr, p_gb = predict(STATE["models"], X)
    s = float(s[0]); meta = STATE["meta"]; cal = meta["calibration"]
    zz = math.log(min(max(s, 1e-4), 1 - 1e-4) / (1 - min(max(s, 1e-4), 1 - 1e-4)))
    p = 1 / (1 + math.exp(-(cal["a"] * zz + cal["b"])))
    amt = r["claim_amount_inr"]
    ev = p * amt - (1 - p) * meta["goodwill_inr"]
    review = s >= meta["queue_cutoff_score"]
    pr_row = STATE["snap"].reindex([r["partner_id"]]).iloc[0]
    notes = []
    if r["claim_amount_inr"] < 2000 and r["partner_inspected"] == "N": notes.append("This claim would be auto-approved under the 1 May 2026 rule.")
    if not X.p_n_prior.iloc[0]: notes.append("No investigated claims on record for this partner yet - score leans on the claim itself.")
    return {"claim_id": r["claim_id"], "score": round(s, 4), "est_fraud_probability": round(p, 3),
            "recommendation": "REVIEW BEFORE PAYING" if review else "PAY NORMALLY",
            "why_this_recommendation": ("Score is inside the top slice the desk can actually review (about 40 claims a month)." if review else
                                        "Score is below the review line; reviewing this one would likely cost more goodwill than it saves."),
            "expected_value_of_review_inr": round(ev, 0), "claim_amount_inr": amt,
            "reasons": explain(X, r), "notes": notes,
            "partner": {"partner_id": r["partner_id"], "investigated_claims": int(pr_row.p_n_prior) if pd.notna(pr_row.p_n_prior) else 0,
                        "confirmed_frauds": int(pr_row.p_fraud_prior) if pd.notna(pr_row.p_fraud_prior) else 0},
            "model": {"version": "kestrel-fraud-v1", "history_as_of": meta["asof"], "trained_on_labelled_claims": meta["n_train"],
                      "caveat": "Ranks claims for review; it is not proof of fraud. Built on 141 confirmed frauds, history as of 1 Jul 2026."}}

@app.get("/")
def home(): return send_from_directory(app.static_folder, "index.html")

@app.get("/health")
def health(): return jsonify(ready=STATE["ready"], error=STATE["error"])

@app.get("/examples")
def examples():
    try: return jsonify(json.load(open(os.path.join(ROOT, "app", "examples.json"))))
    except Exception: return jsonify([])

@app.post("/score")
def score():
    if not STATE["ready"]:
        return jsonify(error="The model file could not be loaded.", detail=STATE["error"], fix="Run: python src/train.py (needs the data files in ./data) and restart."), 503
    try:
        rec = request.get_json(force=True, silent=False)
        return jsonify(score_record(rec))
    except Bad as e: return jsonify(error=str(e)), 400
    except Exception as e: return jsonify(error="Could not score this claim.", detail=f"{type(e).__name__}: {e}"), 422

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), debug=False)
