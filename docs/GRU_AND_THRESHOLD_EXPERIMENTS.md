# GRU vs trees, and threshold experiments (FD001)

Both experiments use the shared protocol from `EVALUATION_RESULTS.md` (grouped 5-fold CV, seed 0, truncated endpoints, truth clipped at 150 for CV) and the out-of-fold predictions behind `POLICY_BACKTEST.md`. Raw numbers: `gru_comparison.json`, `threshold_experiments.json`. Backend defaults were **not** changed.

## 1. Is a GRU better than the tree models?

The GRU is a **JAX stand-in** (`scripts/exp_gru_jax.py`, experimental) because torch could not be installed in the analysis environment. It copies the promoted torch architecture (1-layer GRU, 32 hidden, 16-unit head, Huber, Adam 1e-3, batch 256, clip 5, PyTorch gate order/init). Before trusting it, it was run on the promoted setup (24 raw features, uncapped, units 1-80, 11 epochs):

| Check (3 seeds) | Promoted torch | JAX stand-in |
|---|---|---|
| Dev-validation MAE (mean) | 27.15 | 26.49 |
| Dev-validation low-RUL MAE | 5.04 | 5.15 |
| Test RMSE, seed 42 | 25.28 | 24.36 (seeds 7/42/77: 21.6 / 24.4 / 25.6) |

The stand-in reproduces the torch numbers within seed noise, so it is a fair proxy. It was then trained on the same inputs as the best tree model (14 informative sensors, RUL capped at 125; epoch count chosen on a 20% engine hold-out inside the training engines).

| Model | CV RMSE (clipped truth) | CV low-RUL MAE | Test RMSE (refit on 100 engines unless noted) | Test NASA |
|---|---|---|---|---|
| HGB, window features, capped | **17.98** | 3.43 | **14.35** | 287 |
| RF, window features, capped | 18.27 | 3.98 | 13.47 (trained on units 1-80 only) | not computed |
| GRU, 14 sensors, capped (mean of 3 seeds) | 18.58 (fold SD 1.57) | 5.03 | 15.64 (seeds 15.3-16.0; bootstrap 95% CI 12.9-17.6) | 457 |
| Promoted torch GRU (24 raw, uncapped) | not on this protocol | | 25.28 | |

**Verdict:** the trees are slightly better on CV (the GRU is within one fold SD of both) and on the test set (units 1-80 only: HGB 14.61, RF 13.47, GRU 15.34), and they are much simpler, so they stay the primary model. The bigger finding is that the promoted GRU's weakness was its inputs and target, not the architecture: capping RUL and dropping uninformative sensors takes the same network from 25.3 to about 15.6 test RMSE. The GRU is within CV noise of HGB, so there is no case for swapping it in; torch retraining would only matter if the GRU later wins on a harder subset (FD002/FD004). One caveat: the GRU's CV used one training seed per fold, the test numbers use three.

## 2. Threshold experiments (HGB OOF predictions unless stated; failure = 10x a planned job)

The cost floor (perfect foresight) is 4.87 per 1000 cycles. Every sensible policy here lands at 5.0-5.6, so differences of ~0.5 are about **one engine** failing out of 100. Read the trends, not the decimals.

- **E1, margin rule (trigger at lead + m).** The best margin is 8-11 cycles for every lead (cost 5.03 / 5.07 / 5.13 for lead 0 / 5 / 10, 0 failures). A margin of 0 is bad (cost 34-45, nearly every engine fails). The backend's fixed threshold of 10 is fine only for lead 0; at lead 5 it costs 8.08, and at lead 10 it equals margin 0 (33.8). Recommendation unchanged: set the threshold to the execution lead time plus about 10.
- **E2, persistence k x lead.** k = 2 or 3 gave the lowest cross-fold cost at leads 5 and 10 (about 5.05 vs 5.5 for k = 1), but k = 5 is worse (about 5.5-5.95) because it delays the alert. This is a one-engine difference, so use k = 2-3 as a mild noise filter, not as a proven gain.
- **E4, model robustness (lead 5).** Best in-sample threshold: HGB 14 (within 5% of best cost for 14-22), RF 13 (13-23), GRU 15 (15-25). The three models agree within 2 cycles, and cross-fold choices were stable. The threshold range is wide on the high side and sharp on the low side, so err high. The GRU showed 0 failures vs 1 per 100 for the trees, which is within noise.
- **E5, uncertainty-aware trigger (RF mean minus z x tree spread).** No clear trend: cost was 5.45, 5.44, 4.98, 5.49, 5.59 for z = 0, 0.5, 1, 1.5, 2, and the z = 1 win is again one engine. Larger z mostly just raises unused life (6.4 to 11.5 cycles). Not worth adding to the product on this evidence.

Caveats: one dataset (100 engines), illustrative costs, a single CV seed, and thresholds near the floor where one engine moves the result by 5-10%.

## 3. Fixed-threshold sweep (default 10 vs 15, regret table)

A second, fixed-in-advance sweep (`scripts/run_fd001_threshold_experiment.py`, output `threshold_experiment.json`) scores thresholds 8-30 on all 100 out-of-fold engines across 4 cost ratios x 6 lead times x persistence 1 or 3, with a paired bootstrap for 15 vs the current 10. Thresholds were fixed before running, so there is no selection step, but the lead-time grid was chosen by us.

- **15 beats 10 in 16 of 24 single-trigger scenarios, ties in 4, and loses in 4.** The 4 losses are lead 0 and 2, where 10 costs 1-3% less. At a 5-cycle lead and failure ratio 10, 10 costs 8.08 with 7 failures per 100 and 15 costs 5.09 with none; at a 10-cycle lead it is 33.8 vs 7.65.
- **Worst-case regret as the lead time is unknown** (worst cost increase over the best threshold for that scenario): if the lead is 0-5 cycles, 15 is the safest fixed threshold (worst-case 4% above best, while 10 can be 334% above). If leads up to 10 are possible, 25 is safest (10% worst case). If leads up to 15 are possible, 30 (13%). Threshold 10 is never the safest option.
- With 3-in-a-row persistence the safest thresholds rise to 20 (leads 0-5) and 25-30 beyond, because the alert is delayed by two cycles.

**What to do:** the data supports raising the default from 10 only if the real work-order lead time is at least ~2 cycles; the safe default grows with the lead time, which is a business input we do not have. Choosing the new default (and whether to wire it into `RUL_CRITICAL_THRESHOLD`) stays with the user.
