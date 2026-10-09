# 3-minute screen recording – shot list (no slides)
0:00-0:20  Open `evidence/EVIDENCE.md` table. "Only 1.2% of claims are fraud. Predicting nothing scores 97-99%. So I threw accuracy out and measured rupees per claim reviewed."
0:20-0:55  What I tried: random split (looked great, but re-submitted claims leak; threw away). Forward-in-time folds instead. Show `backtest_summary.png`: A and C work, B fails at 0.48 when the 1 May rule change landed. "That failure is the main finding about this model."
0:55-1:20  What I changed: dropped outlet age/type so it can't paint new partners (AUC 0.892 → 0.879, same catches); weighted post-May claims; de-duplicated 681 rows. What I threw away: GBM alone, first reasons text, unpinned deps.
1:20-2:10  Run `python app/server.py`. Browser: example 1 (REVIEW, reasons), example 3 (PAY NORMALLY), then clear partner_id to show the polite error. Show `curl` to `/score`. Rename `artifacts/model.joblib` and reload to show it fails politely.
2:10-2:45  Open `memo.pdf`: the decision, Rs 502 per claim (June) → plan on ~Rs 350, 8 outlets, 2 in 3 holds are genuine.
2:45-3:00  "What breaks it: the next policy change, and a new ring with no history. Retrain monthly."
