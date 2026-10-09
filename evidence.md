# Evidence – does it work, and how often does it not?

## 1. The KPI problem (read this first)
Only **1.2%** of investigated claims are fraud (141 of 11,146). A "model" that says *nothing is ever fraud* is right 98.8% of the time.
Same story in the folds below: 99.0% (Q1-26), 97.9% (Apr-Jun), **96.9% (June)**. A 97% accuracy bar is cleared by doing nothing, and our real model
scores the same on it (96.6–99.6%). So accuracy cannot tell Kestrel whether the model works. We report ranking quality (AUC / average precision)
and, for Finance, rupees: **fraud stopped per claim reviewed**, capped at the desk's 40 reviews/month.

## 2. How it was tested
Forward in time only: train on the past, score the next 1-3 months, with each partner's history frozen at the fold start exactly like the real test (scored 1 Jul-30 Sep from history to 30 Jun).
A random split would leak (re-submitted claims share a claim_id and identical fields) and would hide the May-2026 policy change. Review slots = 40 x months in the window.
Value of a catch = claim amount; cost of a wrongly held genuine claim = Rs 380 goodwill (policy s4). 90% intervals are bootstrap over claims.

| Fold (train → test) | Frauds in window | AUC (90% int.) | Avg precision | Reviewed | Frauds caught | Fraud Rs stopped | **Gross Rs / claim checked** (90% int.) | Net of goodwill / claim | Net also paying Rs 260/contact |
|---|---|---|---|---|---|---|---|---|---|
| A: old regime (train Oct25-Dec25, test Q1-26) | 22 | 0.90 (0.83-0.96) | 0.72 | 120 | 16 | 1,25,270 | **1,044** (527-1,625) | 715 | 455 |
| B: policy change lands mid-window (test Apr-Jun-26) | 45 | **0.48 (0.41-0.55)** | 0.06 | 120 | 4 | 18,141 | **151** (10-360) | -216 | -476 |
| C: new regime (train thru May-26, test Jun-26) | 22 | 0.88 (0.78-0.95) | 0.37 | 40 | 14 | 20,081 | **502** (296-698) | 255 (-3 to 491) | -5 |

Baselines in `backtest.csv`: random and "flag nothing" lose Rs 195-380 per claim checked; ranking by a partner's past fraud rate alone works pre-shift (AUC 0.86) but collapses after it (0.73 / 0.82, 0-3 catches).

## 3. How often it does not work
* **Two in three holds are genuine customers** (fold C: 26 of 40 holds). That is the price; each costs Rs 380 of goodwill.
* **It fails when the world changes.** Fold B (trained entirely on pre-May data, tested across the 1 May change) is a coin flip or worse (AUC 0.48). Fraud changed shape: median fraud claim Rs 3,392 before May, Rs 1,416 after; from large inspected claims at ~7 older outlets to small uninspected claims at new outlets. It recovers once it has seen about a month of the new pattern (fold C). Retrain monthly; re-test after any policy change.
* **Small numbers.** 22 frauds in a fold means wide intervals. Do not read 0.88 as 0.88.
* **Test is further out than fold C.** Fold C scores the month right after training; the real test is 1-3 months out with history frozen. Expect worse than C (see form for stated expectation).
* Partner "memory" is the strongest signal. A new ring at an outlet with no investigated history is caught only by claim-level signals (small, uninspected, just under Rs 2,000, repeat customer).
* ~11 of the 40 top claims in fold C were at new outlets that are otherwise clean. Per-claim review will sometimes hit good partners.

## 4. Ablations
Removing outlet age/type: AUC 0.892 → 0.879, same 14 frauds caught - **kept removed**, so a partner is judged on its own record, not for being new (Service Desk ask).
Removing all partner history: AUC 0.79, 6 catches (collapses). Neither model alone dominates (logistic is better in fold C, boosting in fold A), which is why the final score is a 50/50 blend; with 22 frauds per fold this is not a strong choice.

## 5. What the data showed (and what we did about each trap)
| Issue | What we found | What we did |
|---|---|---|
| Re-submitted claims | 681 duplicate claim_ids, identical except a later `submitted_at`, no label conflicts | Kept the first, dropped 681 (12,029 → 11,348) |
| "Undecided" labels | 215 blank (202 after de-dup), all in CRM. Legacy Zoho stored these as 0, so some legacy 0s are really open cases and cannot be told apart | Excluded blanks from training; legacy zeros kept (label noise, mostly in old-regime rows) |
| Serial mess | ~30% have a leading space, lower case or hyphens; fraud rate 1.2% either way; repeated serials not predictive | Normalised; no serial feature used |
| Free text | 4 descriptions had extra wording appended after the fault phrase | Kept only the leading fault phrase |
| Timestamps | Policy: legacy Zoho *resolution events* are UTC. The pack has none; `submitted_at` is IST. | Used as is; no time-of-day features |
| `source` column | Proxy for date | Not used as a feature |
| May-2026 change | Claims under Rs 2,000 skip inspection: inspected share 92% → 22%; fraud rate on small uninspected claims 3.2% after vs 0.2% before | Features for small / uninspected / just-under-2,000; post-May rows weighted 3x |
| Out of warranty | No claim is past warranty | Feature kept, carries no signal |
| Reading the brief | "Accuracy > 97%" would be met by flagging nothing | Reported rupees per claim reviewed instead |

## 6. The partner question
New outlets (onboarded in the last 12 months, 60 of ~380) are 4% of investigated claims but 24% of all fraud and **34 of the 36 frauds since 1 May**. Yet only **8 of the 46 new outlets with investigated claims have any confirmed fraud** (38 are clean). Before May, fraud sat mainly at ~7 older outlets (SP3207, SP3376, SP3095, SP3350, SP3103, SP3228, SP3292: 7-13 frauds each, none since). So the data supports "a handful of outlets, mostly new" and not "the newer partners".
Of the 120 claims at the top of the test queue, 79 come from 8 outlets (SP3129, SP3160, SP3232, SP3319, SP3252, SP3307, SP3287, SP3300).
Simple rule check, June: outlets with 2+ confirmed frauds since 1 May (3 outlets, 16 claims) held 7 of June's 22 frauds, 44% precision.
