# Industrial Asset Reliability Platform

An end-to-end predictive-maintenance system. It turns machine telemetry into Remaining Useful Life (RUL) estimates, risk levels, alerts and maintenance work orders. It is built on the NASA C-MAPSS FD001 turbofan dataset, with a FastAPI and SQLAlchemy backend (SQLite or PostgreSQL), role-based access control and a React (Next.js) operations console.

The project's focus is evaluation discipline: every model and policy claim below is scored on one shared, leakage-safe protocol, and the numbers in this file come from result files in [`docs/`](docs/) that the scripts in [`scripts/`](scripts/) regenerate.

## Highlights

| Area | Result |
|---|---|
| RUL accuracy (official FD001 test set, 100 engines) | **RMSE 14.35** (95% CI 12.0–16.5), MAE 10.76 |
| Versus the original 24-feature GRU, same 80 training engines | RMSE **25.28 → 14.61 (−42%)** |
| NASA asymmetric score | **295** vs 3,763 for a raw-feature random-forest baseline (−92%) |
| Maintenance policy back-test (100 engines, out-of-fold predictions) | **35% lower cost** than the best fixed-interval schedule (95% CI 19.8–48.3%), 89% lower than run-to-failure, within 13% of a perfect-foresight lower bound |
| Work-order threshold finding | Raising the critical threshold from 10 to 15 cycles cut cost from 8.08 to 5.09 per 1,000 cycles and failures from 7 to 0 per 100 engines (5-cycle execution lead) |
| Backend concurrency (PostgreSQL 17, 4 concurrent assets) | 960 telemetry requests, **0 errors**, 85 requests/s, p95 latency 64 ms |
| Engineering | 15 REST endpoints, 3 roles, 5 Alembic migrations, 260 automated tests, about 7,000 lines of Python |

The data is simulated, the maintenance costs are illustrative assumptions, and the concurrency benchmark is a local in-process run. See [Limitations](#limitations).

## How it works

```text
Telemetry cycle
      ↓
Persist, then assemble the latest 30 consecutive cycles
      ↓
RUL model: window-feature gradient boosting (default) or GRU
      ↓
Raw RUL, RUL capped to 0–125, top-3 drifting sensors
      ↓
Deterministic risk severity (CRITICAL / HIGH / MEDIUM / NORMAL)
      ↓
Active alert  →  work order at CRITICAL  →  guarded state transitions
```

- Telemetry ingestion is idempotent, and conflicting retries are rejected. Predictions are unique per asset, cycle and model version.
- Alerts and work orders follow guarded state machines. Every operator action records who made it.
- The served model's manifest freezes the input sensors, checks the artifact's SHA-256 at startup and verifies stored golden predictions, so a changed library version cannot silently change predictions.

## Machine learning results

All rows are scored once on the official FD001 test endpoints (100 engines, one final window each). Hyperparameters and features were chosen by grouped 5-fold cross-validation over training engines, with 10 seeded truncated endpoints per held-out engine. Test labels were not used for selection. Details and caveats: [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md).

| Model | Features | RUL target | Test RMSE | Test MAE | NASA score |
|---|---|---|---:|---:|---:|
| Original GRU (legacy, `RUL_MODEL=gru`) | 24 raw | uncapped | 25.28 | 18.31 | not computed |
| Random forest | 24 raw | uncapped | 23.44 | 16.91 | 3,763 |
| Random forest, window features | 14 sensors | capped at 125 | 13.47 | 10.17 | 246 |
| Gradient boosting, window features | 14 sensors | capped at 125 | 14.61 | 11.28 | 295 |
| **Same, refit on all 100 engines (served, v2.0.0)** | 14 sensors | capped at 125 | **14.35** | **10.76** | **287** |
| Experimental GRU on the same inputs (JAX, mean of 3 seeds) | 14 sensors | capped at 125 | 15.34 | 11.26 | 430 |

Rows other than the refit row train on engines 1–80. Cross-validated RMSE is 17.98 for gradient boosting, 18.27 for random forest and 18.58 for the GRU. The random forest and gradient boosting are within noise of each other, and I report the one cross-validation selected.

What drove the improvement:

- **Feature cleaning:** only 14 of 21 sensors carry degradation signal. The original model also fed it constant sensors and operating settings.
- **Capped RUL target (125):** this removes the unpredictable early-life region from the loss.
- **Window summaries:** the last value, mean and slope over 30 cycles.
- **Architecture was not the problem:** the same GRU on the cleaned inputs and capped target improves from 25.3 to 15.3 RMSE, but still does not beat the trees. See [`docs/GRU_AND_THRESHOLD_EXPERIMENTS.md`](docs/GRU_AND_THRESHOLD_EXPERIMENTS.md).

The capped target's advantage depends on the scoring protocol. On raw, uncapped truth the capped and uncapped window models score the same. The cap value 125 is a convention and was not shown to be optimal.

## Maintenance-policy back-test

Predictions are replayed cycle by cycle on all 100 training engines using out-of-fold models. Thresholds and intervals are chosen on other folds and applied to the held-out fold. Scenario, declared before running: a failure costs 10 times a planned job, and a work order takes 5 cycles to execute. Full method and sensitivity grid: [`docs/POLICY_BACKTEST.md`](docs/POLICY_BACKTEST.md).

| Policy | Cost per 1,000 operating cycles | Failures per 100 engines |
|---|---:|---:|
| Run to failure | 48.47 | 100 |
| Best fixed maintenance interval | 8.49 | 1 |
| **Model-driven threshold** | **5.51** | **1** |
| Perfect foresight (lower bound) | 4.87 | 0 |

The model-driven policy saves 35.1% versus the fixed interval (paired bootstrap 95% CI 19.8–48.3%). Across 12 scenarios (failure-cost ratios 5–50, execution lead 0–10 cycles) the saving stays between 34% and 39%.

A trigger of the execution lead plus about 10 cycles is close to optimal for every lead tried. Because the previous fixed default of 10 cycles was only safe when work orders execute almost immediately, the default is now 15. Differences between good policies amount to about one engine in 100, so read the trends and not the decimals. Experiments on persistence, model choice and uncertainty-aware triggers are in [`docs/GRU_AND_THRESHOLD_EXPERIMENTS.md`](docs/GRU_AND_THRESHOLD_EXPERIMENTS.md).

## Backend and engineering

- **API:** 15 endpoints covering assets, telemetry, predictions, alerts and work orders, with strict request validation (exactly 30 consecutive cycles, unknown fields rejected). OpenAPI docs are served at `/docs`.
- **Security:** bearer tokens stored only as SHA-256 digests and compared in constant time, with viewer, operator and admin roles. See [`docs/AUTHENTICATION_AND_RBAC.md`](docs/AUTHENTICATION_AND_RBAC.md).
- **Persistence:** SQLAlchemy models, 5 Alembic migrations tested for round trips on populated databases, atomic ingestion, per-asset row locking (`SELECT ... FOR UPDATE`) with bounded PostgreSQL transaction retries, and a unique constraint per asset, cycle and model version.
- **Concurrency testing:** isolated PostgreSQL race tests (same-cycle ingestion, concurrent decisions, workflow races) plus a reproducible benchmark. Recorded run: 4 concurrent assets × 240 cycles = 960 requests, 0 errors, 85.3 requests/s, p50 46 ms, p95 64 ms, with the original GRU. Real auth and real PostgreSQL I/O were included, but network, TLS and a production worker topology were not. Details and boundaries: [`docs/BACKEND_PERSISTENCE.md`](docs/BACKEND_PERSISTENCE.md).
- **Console:** a React (Next.js) operations console showing the priority queue, remaining life, the serving model, top drifting sensors per prediction (a drift ranking, not a diagnosis), and alert and work-order actions gated by role.
- **Replay service:** an adjustable-speed, resumable FD001 telemetry replay that drives the real API.
- **Tests:** 260 tests across data integrity, leakage-safe splits, feature parity between training and serving, model artifact integrity, API validation, migrations, idempotent ingestion, workflows and PostgreSQL concurrency.

## Repository structure

```text
backend/       FastAPI app, persistence, inference adapters, decision logic
frontend/      React operations console
ml/            Dataset loader, features, evaluation protocol, policy simulator, models, artifacts
scripts/       Data acquisition, evaluation, back-test, export, replay, benchmark commands
migrations/    Alembic migrations
deploy/        Local PostgreSQL service for concurrency tests
docs/          EDA, evaluation, back-test and engineering reports with their JSON results
tests/         Automated tests
data/          Local data locations (contents ignored; see Dataset)
```

## Getting started

Python 3.12 is recommended. PyTorch is only needed for the legacy GRU and its training scripts.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\scripts\acquire_cmapss.ps1          # downloads the NASA FD001 files into data/raw/
alembic upgrade head                   # SQLite under data/interim/ by default
```

Configure at least one principal (token digests only) and start the API. Format and role matrix: [`docs/AUTHENTICATION_AND_RBAC.md`](docs/AUTHENTICATION_AND_RBAC.md).

```powershell
$env:AUTH_PRINCIPALS_JSON = '<configured principal digest records>'
uvicorn backend.app.main:app --reload
```

Replay an FD001 engine through the running API, and run the console:

```powershell
$env:RELIABILITY_API_TOKEN = '<issued operator or admin credential>'
python scripts/replay_fd001.py --unit-id 81 --cycles-per-second 5

cd frontend; npm install; npm run dev
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RUL_MODEL` | `hgb` | `hgb` serves gradient boosting v2.0.0. `gru` serves the legacy torch GRU v1.0.0 |
| `RUL_CRITICAL_THRESHOLD` | `15` | CRITICAL at or below this RUL (set to execution lead time + about 10) |
| `RUL_HIGH_THRESHOLD` / `RUL_MEDIUM_THRESHOLD` | `30` / `70` | HIGH and MEDIUM severity bounds |
| `DATABASE_URL` | SQLite | for PostgreSQL use `postgresql+psycopg://user:password@host/db` |

Stored predictions keep the severity they were created with, so changing thresholds does not rewrite history. The served model's RUL is clipped to 0–125, so values near 125 mean "healthy" and are not a measured remaining life. Pickled models depend on the scikit-learn version. After changing it, run `python scripts/export_fd001_hgb.py` to rebuild the artifact. The service refuses to start if the stored golden predictions do not match.

## Reproduce the results

```powershell
python scripts/run_fd001_evaluation.py             # shared-protocol model grid -> docs/evaluation_metrics.json
python scripts/run_fd001_policy_backtest.py        # maintenance-policy back-test -> docs/policy_backtest.json
python scripts/run_fd001_threshold_experiments.py  # threshold experiments
python scripts/exp_gru_jax.py all                  # experimental JAX GRU comparison (needs jax)
python scripts/export_fd001_hgb.py                 # rebuild the served model and manifest
pytest -q                                          # automated tests
```

PostgreSQL concurrency tests and the benchmark need Docker:

```powershell
docker compose -f deploy/docker-compose.postgres.yml up -d --wait
$env:TEST_DATABASE_URL = 'postgresql+psycopg://reliability:reliability@127.0.0.1:5433/reliability'
python -B -m pytest -q -ra -p no:cacheprovider
python -B scripts/benchmark_postgres.py --assets 4 --cycles 240 --concurrency 4
```

## Dataset

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), subset FD001: 100 training engines (20,631 cycles), 100 test engines, one operating condition and one fault mode (HPC degradation). The data is downloaded locally and not committed.

## Limitations

- The data is simulated, with one operating condition and one fault mode. Nothing here is validated on a physical fleet.
- The maintenance costs and execution lead times are illustrative. The savings are scenario results, not forecasts.
- With 100 engines the policy comparisons rest on about one engine of difference in many cases. Intervals are reported where computed.
- The GRU comparison uses a JAX re-implementation. The deployable GRU artifact is still the original torch model.
- The concurrency benchmark is a local, single-process, in-process run. It is not a capacity or SLA claim.
- Enterprise SSO, asynchronous processing and production monitoring are not implemented. Migration `0004` cannot reconstruct decision provenance for predictions that existed before it, and retrying one returns HTTP 409.
