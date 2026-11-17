# FD001 Shared-Protocol Evaluation

All models below are scored on the same rows with the same code
(`ml/evaluate.py`, `scripts/run_fd001_evaluation.py`). Machine-readable output is
in `docs/evaluation_metrics.json`.

## Why this exists

Earlier reports compared models on different rows and different distributions:

- Baselines were scored on all 4,493 validation rows; the GRU on the 3,913
  windows that have 30 cycles behind them. Cycles 1–29 are the hardest rows.
- Validation metrics cover every cycle of units 81–100, where 36% of windows have
  RUL above 125. The official test endpoints have only 11% above 125 (maximum
  145), so validation MAE and test MAE are not comparable.
- The capped-versus-linear experiment promised in `EDA_FINDINGS.md` was never run.

## Protocol (fixed before any test label was scored)

- **Features:** 14 informative sensors (sensors 2, 3, 4, 7, 8, 9, 11, 12, 13, 14,
  15, 17, 20, 21). Operational settings and constant or two-valued sensors are
  dropped (`ml.features.select_informative_sensors`).
- **Representations:** `current` (values at the endpoint) or `window` (last value,
  30-cycle mean, 30-cycle slope per sensor).
- **Target:** uncapped linear RUL, or RUL capped at 125 (a convention, not tuned).
- **Selection:** grouped 5-fold CV over the 100 training engines, scored on 10
  seeded truncated endpoints per held-out engine (cut at 20–97% of life). Test
  labels play no part in selection.
- **Scoring:** predictions are clipped at zero and compared with unmodified true
  RUL on the official test endpoints. Metrics: RMSE, MAE, NASA asymmetric score
  (lower is better), and MAE where true RUL <= 30.

## Official test endpoints (100 engines, scored once)

Models trained on units 1–80 so they match the promoted GRU's training data. RF
rows are the mean of seeds 0, 1, 2.

| Model | Features | Target | RMSE | MAE | NASA | MAE at RUL <= 30 |
|---|---|---|---:|---:|---:|---:|
| Promoted GRU (seed 42, single run) | 24 raw | uncapped | 25.28 | 18.31 | not computed | 5.00 |
| Random forest | 24 raw | uncapped | 23.44 | 16.91 | 3,763 | 10.64 |
| Histogram gradient boosting | 24 raw | uncapped | 26.27 | 18.42 | 16,903 | 11.36 |
| Random forest | 14 sensors, current | capped | 17.84 | 13.17 | 698 | 10.53 |
| Random forest | 14 sensors, window | uncapped | 19.77 | 14.70 | 1,849 | 4.41 |
| Random forest | 14 sensors, window | capped | 13.47 | 10.17 | 246 | 4.17 |
| Histogram gradient boosting | 14 sensors, window | capped | 14.61 | 11.28 | 295 | 4.03 |

The configuration chosen by CV was histogram gradient boosting on window features
with the capped target. Refit on all 100 training engines it scores **RMSE 14.35,
MAE 10.76, NASA 287, low-RUL MAE 4.32**. The bootstrap 95% interval for RMSE over
the 100 test engines is **12.0–16.5**.

The GRU's NASA score is not computed because predictions from the saved model
were not regenerated for this report.

## Cross-validation (selection data)

Mean over 5 grouped folds. "Clipped" scores against truth clipped at 150 cycles;
"raw" scores against unmodified truth.

| Features | Target | Model | RMSE clipped | RMSE raw | NASA (clipped) |
|---|---|---|---:|---:|---:|
| current | uncapped | RF | 28.68 | 32.52 | 23,315 |
| current | uncapped | HGB | 29.82 | 33.38 | 36,776 |
| current | capped | RF | 24.54 | 31.44 | 3,843 |
| current | capped | HGB | 24.83 | 31.59 | 4,338 |
| window | uncapped | RF | 21.42 | 25.19 | 5,866 |
| window | uncapped | HGB | 22.67 | 25.91 | 26,197 |
| window | capped | RF | 18.27 | 26.13 | 1,036 |
| window | capped | HGB | **17.98** | 25.81 | 1,055 |

## What the numbers support

- **History helps.** Window features cut low-RUL MAE from about 10.5 to about 4
  with a forest, and cut test RMSE by 4 to 5 cycles at either target.
- **Capping helps on the official test endpoints.** Test RMSE drops from 19.77 to
  13.47 for the window forest. This is a statement about that protocol, see below.
- **The promoted GRU does not beat simple tree models on this protocol.** Its RMSE
  of 25.28 is above every window model and about equal to the current-cycle
  baselines. It matches the window models only on near-failure MAE. This is for
  an uncapped, untuned GRU on 24 raw features. A capped, cleaned-feature GRU was
  not run.

## Caveats

- **The capping advantage depends on the protocol.** On raw-truth CV the capped
  window models (25.8–26.1) are no better than the uncapped ones (25.2–25.9). The
  gain appears when truth is limited to the horizon where the test endpoints live
  (maximum 145). Capping is justified by that, not by being universally better.
- **The CV horizon was changed once.** The first CV pass scored against raw truth,
  which reaches about 290 for early cuts, and penalized capped models for RUL no
  sensor can reveal. The horizon was set to 150 after viewing CV output only,
  before any test label was scored.
- **Cap sensitivity (CV RMSE: cap 100 = 23.70, 115 = 19.52, 125 = 17.98,
  130 = 17.83, 140 = 17.48) is confounded with that horizon.** Larger caps look
  better partly because they match the 150 horizon. It does not show 125 is
  optimal. The cap stays at the pre-declared 125.
- **HGB versus RF is a toss-up.** CV favored HGB by 0.3 RMSE against a fold
  standard deviation of 1.4–1.8. On test the forest was ahead (13.47 against
  14.61). The CV-selected model is reported, not the better test number.
- **Test set size.** 100 engines gives a wide interval (12.0–16.5).
- **Not deployed.** The backend still serves the promoted GRU
  (24 features, uncapped). These models are not wired into the API.

## Reproduce

```powershell
python scripts/run_fd001_evaluation.py
pytest -q tests/test_features.py tests/test_evaluate.py
```

Set `EVAL_CACHE_DIR` to a folder outside the repository to make the run resumable.
