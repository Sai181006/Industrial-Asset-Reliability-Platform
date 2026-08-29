# FD001 GRU Model Promotion

## Promotion decision

`fd001-gru-rul` version `1.0.0` is promoted for synchronous backend MVP
integration. Promotion means the artifact and inference contract are stable
enough for application development; it does not claim production certification.

## Stability across fixed seeds

The architecture, 30-cycle window, preprocessing, target, and 11-epoch training
length were frozen before running seeds 7, 42, and 77 on units 1–80. Units
81–100 remained the development validation set.

| Metric | Mean | Sample SD | Range |
|---|---:|---:|---:|
| MAE | 27.147 | 1.081 | 26.492–28.395 |
| RMSE | 41.747 | 1.816 | 40.623–43.843 |
| MAE at RUL <= 30 | 5.043 | 0.134 | 4.890–5.136 |
| RMSE at RUL <= 30 | 6.974 | 0.352 | 6.718–7.376 |

Near-failure behavior is stable across these runs. Aggregate error varies more,
but every seed remains materially better than the random-forest development MAE
of 36.369. Seed 42 remains promoted because it was predeclared and already
packaged—not because of official test performance.

## Official FD001 test endpoints

One final 30-cycle window was constructed per official test engine. The scaler
from training units 1–80 was applied unchanged. NASA's 100 provided RUL values
were used only for this final evaluation and did not select configuration,
epochs, or seed.

| Output | MAE | RMSE | MAE at RUL <= 30 | RMSE at RUL <= 30 | Negative predictions |
|---|---:|---:|---:|---:|---:|
| Raw seed-42 GRU | 18.305 | 25.282 | 4.998 | 7.396 | 0 |
| Operational non-negative | 18.305 | 25.282 | 4.998 | 7.396 | 0 |

Clipping has no effect on these official endpoints because none is negative.

## Output policy

- `raw_rul` is the unmodified floating-point model output. It is persisted for
  audit, error analysis, and drift monitoring.
- `operational_rul = max(raw_rul, 0.0)` is the value exposed to deterministic
  risk rules and user workflows. Negative remaining life is physically invalid.
- Both values use operating cycles. No rounding occurs in the model boundary;
  presentation layers may format values without changing persisted precision.

## Versioned contract

The machine-readable manifest is `ml/artifacts/fd001_gru_manifest.json`. It
records artifact SHA-256, feature order, input shape, scaling rule, target/output
semantics, training units and seeds, metrics, and runtime versions.

Artifact SHA-256:
`8d72fd3183d02eda52765ef7f6c486b13915cafef06dd688f18fb2252bf081b1`

## Limitations

- Three seeds provide a practical stability check, not a full uncertainty study.
- FD001 remains synthetic and covers one operating condition/fault mode.
- The backend must reject missing, reordered, mixed-engine, non-consecutive, or
  incorrectly shaped windows before inference.
- Monitoring thresholds and model replacement policy remain future MLOps work.

MODEL PROMOTION STATUS: PASSED FOR BACKEND MVP
