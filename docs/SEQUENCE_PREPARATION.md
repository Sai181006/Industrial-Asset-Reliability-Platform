# FD001 Sequence Preparation

## Outcome

Leakage-safe sequence generation is implemented. The selected initial
configuration is a 30-cycle window with the existing uncapped linear-RUL target.
No sequence model has been trained yet.

## Window evidence

| Window | Sequences | Engines excluded | Median target | Maximum target | RUL <= 30 sequences |
|---:|---:|---:|---:|---:|---:|
| 20 | 18,731 | 0 | 93 | 342 | 3,100 |
| 30 | 17,731 | 0 | 88 | 332 | 3,100 |
| 50 | 15,731 | 0 | 78 | 312 | 3,100 |

Thirty cycles is the first configuration because it adds 50% more history than
20 while retaining 94.7% as many sequences. A 50-cycle window retains only
84.0% as many sequences as 20 and removes considerably more early-life target
coverage. All candidates preserve every engine and every low-RUL endpoint.

The target remains uncapped linear RUL because it directly represents operating
cycles remaining, matches the established baseline contract, and supports the
platform's future deterministic risk mapping. Target capping remains a sensitivity
experiment; it is not adopted merely to reduce early-life regression error.

## Leakage controls

- Scaling is fitted only on raw rows from units 1–80.
- The fitted scaler is applied unchanged to units 81–100.
- Windows are constructed independently inside each engine after sorting by
  cycle; no window crosses engine or split boundaries.
- Each sequence records its owner unit and end cycle for auditability.
- Arrays are stored as `float32` to bound memory without changing raw files.

## Grouped robustness

A five-fold grouped random-forest check was run across complete engines. Each
fold held out 20 engines:

| Metric | Fold mean | Fold standard deviation |
|---|---:|---:|
| MAE | 30.262 | 1.284 |
| RMSE | 42.604 | 1.934 |
| MAE at RUL <= 30 | 7.598 | 1.459 |

The modest fold dispersion indicates that the nonlinear current-cycle signal is
not unique to the original units 81–100 split. The original frozen validation
results remain the comparison contract; grouped folds are robustness evidence,
not a replacement test set.

## Next model experiment

Train one small sequence model first. A compact GRU is preferred over a broad
architecture search because it directly tests whether 30-cycle history improves
over current-cycle tree models with limited complexity. It must report overall
and low-RUL errors, negative predictions, inference cost, and the exact same
validation units. A TCN is the single fallback if the GRU does not add value.

SEQUENCE PREPARATION STATUS: COMPLETE
