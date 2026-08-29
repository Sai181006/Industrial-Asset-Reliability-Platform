# Dataset and Model Suitability Review

## Decision

NASA C-MAPSS FD001 remains the primary dataset for the first end-to-end
Industrial Asset Reliability Platform implementation. No reviewed alternative
is unambiguously better for the present goal of demonstrating leakage-safe RUL,
multi-asset telemetry replay, model inference, alerting, and maintenance workflow
with manageable local compute.

This is not a claim that FD001 is the most realistic prognostics dataset. NASA
N-CMAPSS should be the first later robustness dataset once the FD001 pipeline is
complete. PRONOSTIA is the best later cross-asset-type dataset if real vibration
measurements and bearing-specific feature extraction become project goals.

## Evaluation criteria

Alternatives were judged on target alignment, run-to-failure trajectory count,
sensor/operating-condition structure, physical realism, availability from a
credible source, suitability for telemetry replay, leakage-safe evaluation, and
local implementation cost.

| Dataset | Evidence and strengths | Why it does not replace FD001 now | Role |
|---|---|---|---|
| NASA C-MAPSS FD001 | 100 train and 100 test engines; explicit cycle-level RUL task; multivariate trajectories; official compact archive | Synthetic, one condition and one fault mode; anonymized sensors | Primary MVP |
| NASA N-CMAPSS / Turbofan Simulation-2 | NASA describes realistic flight conditions and operation-history-dependent degradation; higher-fidelity flight trajectories and multiple modes | Millions of 1 Hz rows per sub-dataset, small unit fleets (for example DS02 commonly uses 6 train and 3 test units), substantial memory/subsampling choices; would delay the platform workflow and make validation variance harder to assess | First robustness extension |
| FEMTO/PRONOSTIA bearings | Real accelerated bearing degradation, multiple operating conditions, vibration and temperature signals, explicit RUL challenge | Only two learning bearings per operating condition in the challenge protocol; raw vibration requires domain-specific spectral/health-indicator engineering and is less direct for fleet telemetry replay | Later cross-asset validation |
| Scania APS | Real operational data from heavy trucks; 60,000 training examples and asymmetric failure costs | Anonymized, non-sequential classification data with missing values; no trajectory-level RUL target | Later classification extension |
| AI4I 2020 | Small, accessible synthetic industrial example | Six main features and a machine-failure classification target; lacks run-to-failure asset trajectories and adds no realism over C-MAPSS | Not selected |

## Source review

- NASA PCoE lists both the original C-MAPSS turbofan dataset and the newer
  Turbofan Engine Degradation Simulation-2 dataset, and describes the latter as
  a realistic run-to-failure fleet under realistic flight conditions.
- NASA's technical record states that N-CMAPSS uses real flight conditions and
  relates degradation to operation history.
- NASA PCoE and FEMTO-ST describe PRONOSTIA as real accelerated bearing-life
  experiments intended for RUL/prognostics research.
- UCI describes Scania APS as real heavy-truck operational data for highly
  imbalanced classification, and AI4I as synthetic classification/regression data.

Official sources reviewed 2026-08-28:

- https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- https://ntrs.nasa.gov/citations/20205001125
- https://members.femto-st.fr/patrick-nectoux/fr/pronostic-defaillances-roulement-pronostia
- https://archive.ics.uci.edu/dataset/414/ida2016challenge
- https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenanc

## Model decision

The current-cycle random forest is a useful nonlinear baseline, but histogram
gradient boosting is a stronger tabular candidate that can model nonlinear
interactions efficiently without adding an external dependency. It is evaluated
under the identical frozen split and raw linear-RUL contract. Sequence candidates
remain a small GRU or TCN after window evidence is complete; they must beat or
operationally complement the strongest tabular baseline rather than being assumed
better.

DATASET REVIEW STATUS: FD001 RETAINED
