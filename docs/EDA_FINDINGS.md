# FD001 EDA Findings

## Provenance and local evidence

The NASA C-MAPSS Turbofan Engine Degradation Simulation archive was retrieved
on 2026-08-28 from the official NASA PCoE repository download. The outer archive
SHA-256 is `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
Only `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` were analyzed,
from `data/raw/source/CMAPSSData/`. Raw files were not modified.

FD001 represents one sea-level operating condition and one HPC degradation
fault mode. One row is one observation of one engine at one operational cycle:
unit ID, cycle, three operational settings, and 21 sensor readings (26 columns).

## Integrity and schema

- Training: 20,631 rows, 100 engines, 26 columns.
- Test: 13,096 rows, 100 engines, 26 columns; 100 final-cycle RUL labels.
- Both trajectory files have zero missing cells and zero duplicate rows.
- Unit IDs and cycles are positive integers; every unit's cycles are contiguous
  and start at 1. Four columns are integer typed and 22 are floating point.
- The reusable loader rejects unexpected trajectory/RUL column counts and a test
  RUL count that does not match the number of test engines.

## Trajectories

Training lifetimes range from 128 to 362 cycles, with mean 206.31 and median
199 cycles (standard deviation 46.34; interquartile range 177–229.25). Unit 39
is shortest at 128 cycles, unit 26 is the selected median example at 199, and
unit 69 is longest at 362. These three trajectories were inspected in the
sensor plots.

## Features and temporal behavior

`operational_setting_3`, sensors 1, 5, 10, 16, 18, and 19 are constant in the
local training data. `operational_setting_2` has very small raw variance; sensor
6 has only two values. These are evidence-backed low-information candidates,
not automatic removals. No feature is removed during EDA.

The selected sensor plots show noisy, engine-specific, generally non-monotonic
trajectories. Nevertheless, normalized lifetime progress has substantial
whole-dataset correlations with sensors 12 (-0.702), 7 (-0.685), 21 (-0.667),
3 (0.616), 2 (0.632), 15 (0.673), 4 (0.712), and 11 (0.727). This is preliminary
evidence of degradation-related temporal structure, not proof of causality or
future predictive performance. The three settings are fixed or narrowly
distributed, as expected for FD001's single operating condition.

Feature ranges differ materially (for example, sensor 9 spans 222.86 while
sensor 15 spans 0.2599), so scaling is appropriate for gradient- and
distance-sensitive estimators. Scaling statistics must be fitted only on the
development training engines.

## RUL target and distribution

Training RUL is constructed without mutating raw data as
`max_cycle_for_unit - current_cycle`, producing RUL 0 at the final row. Linear
RUL ranges from 0 to 361, has mean 107.81 and median 103. Of 20,631 rows, 3,100
(15.0%) have RUL at or below 30. Aggregate metrics can therefore underweight
near-failure quality; later evaluation must report the low-RUL region separately.

Linear, uncapped RUL remains the initial transparent target definition. A cap is
not adopted by convention. Capped versus linear RUL is an explicit baseline-phase
experiment whose operational meaning and validation behavior must be compared.

## Leakage-safe development plan

The deterministic first split assigns complete units 1–80 to development
training and units 81–100 to validation. This split is reproducible and prevents
rows or overlapping windows from the same engine appearing on both sides. The
official test set remains untouched for final evaluation. Feature selection and
scalers must be fitted on units 1–80 only, then applied unchanged to validation
and test data. Official test RUL labels must not guide preprocessing decisions.

## Window feasibility

All 100 training engines support each candidate window:

| Window | Sequences | Engines too short | Median sequence-end RUL | Max sequence-end RUL |
|---:|---:|---:|---:|---:|
| 20 | 18,731 | 0 | 93 | 342 |
| 30 | 17,731 | 0 | 88 | 332 |
| 50 | 15,731 | 0 | 78 | 312 |

Longer windows preserve more history but reduce examples and remove more early
life targets. No window is selected yet; baseline/sequence validation and
near-failure errors must justify the choice.

## Baseline and feature plan

The next phase should compare a training-median RUL sanity predictor, linear
regression, and a tree regressor on current-cycle features. Report MAE and RMSE
overall and at RUL <= 30. Only then is a sequence model comparison justified.
Constant features may be removed in the preprocessing pipeline because they add
no FD001 variation; low-variance nonconstant features require validation evidence.

## Risks and unresolved questions

- Whether capped RUL better represents healthy early life than linear RUL.
- Which scaler is most appropriate after the development split is frozen.
- Which of the 20/30/50-cycle windows offers the best history/sample trade-off.
- Whether nonconstant low-variance features add out-of-engine validation value.
- A fixed ID-range split is transparent but may be sensitive to unit ordering;
  later baselines should check robustness with deterministic grouped folds.

EDA GATE STATUS: PASSED
