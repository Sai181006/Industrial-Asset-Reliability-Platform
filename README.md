# Industrial Asset Reliability Platform

An end-to-end predictive-maintenance platform that converts machine telemetry
into Remaining Useful Life (RUL) estimates, deterministic risk levels, alerts,
and maintenance work orders.

The current implementation uses the NASA C-MAPSS FD001 turbofan dataset and a
window-feature gradient-boosting model (the original GRU stays selectable) behind a
synchronous FastAPI and SQLAlchemy backend.

## What is implemented

- Official NASA C-MAPSS FD001 acquisition and validation.
- Reproducible EDA and leakage-safe engine-level splits.
- Median, linear-regression, random-forest, and gradient-boosting baselines.
- A served 30-cycle window-feature gradient-boosting RUL model (version 2.0.0, test RMSE 14.35) and the original GRU (version 1.0.0, test RMSE 25.28) as a selectable legacy model.
- Strict telemetry-window validation and versioned model inference.
- Asset, telemetry, prediction, alert, and work-order persistence.
- Deterministic risk thresholds and guarded maintenance state transitions.
- Full HTTP replay verification using an FD001 engine trajectory.
- Adjustable-speed, resumable FD001 telemetry replay with per-run summaries.
- Responsive fleet operations console for RUL, risk, alerts, and work orders.
- Default-deny bearer authentication with viewer, operator, and admin roles.
- Authenticated-actor audit fields for maintenance workflow changes.
- Role-aware console actions for alert and work-order state transitions.
- Atomic ingestion and asset-scoped PostgreSQL workflow locking with bounded transaction retries.
- Isolated PostgreSQL concurrency tests and a reproducible local multi-asset benchmark.

## Model performance

All rows are scored on the official FD001 test endpoints (100 engines) with one
shared protocol; see [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md).
Models other than the refit row are trained on units 1-80, like the GRU.

| Model | Features | RUL target | Test RMSE | Test MAE | NASA score |
|---|---|---|---:|---:|---:|
| Original GRU (legacy, `RUL_MODEL=gru`) | 24 raw | uncapped | 25.28 | 18.31 | not computed |
| Random forest | 24 raw | uncapped | 23.44 | 16.91 | 3,763 |
| Random forest, window features | 14 sensors | capped at 125 | 13.47 | 10.17 | 246 |
| Histogram gradient boosting, window features | 14 sensors | capped at 125 | 14.61 | 11.28 | 295 |
| **Same, refit on all 100 engines (selected by CV; served, version 2.0.0)** | 14 sensors | capped at 125 | **14.35** | **10.76** | **287** |
| Experimental GRU (JAX stand-in, mean of 3 seeds) | 14 sensors | capped at 125 | 15.34 | 11.26 | 430 |

The refit model's RMSE has a bootstrap 95% interval of 12.0-16.5. The original GRU
does not beat simple tree models on this protocol, so the backend now serves the tree model.
Giving the same GRU capped targets and the 14 cleaned sensors cuts its test RMSE from 25.3 to 15.3 but
still does not beat the trees (see [`docs/GRU_AND_THRESHOLD_EXPERIMENTS.md`](docs/GRU_AND_THRESHOLD_EXPERIMENTS.md)).

Earlier validation tables in `docs/GRU_RESULTS.md` and `docs/BASELINE_RESULTS.md`
use different row sets and are not comparable with this table.

## Maintenance-policy back-test

Replaying all 100 training engines with out-of-fold predictions, a model-driven
maintenance rule costs 35.1% less per operating cycle than the best fixed
interval (95% interval 19.8% to 48.3%), assuming a failure costs 10x a planned
job and work orders take 5 cycles to execute. A critical threshold of
10 cycles is only safe when work orders execute almost immediately: with a
5-cycle lead it fails 7 engines in 100. The default is now 15, the safest fixed
value for leads of 0-5 cycles (use lead time + about 10 if yours differs). These costs are illustrative assumptions
on simulated data. See [`docs/POLICY_BACKTEST.md`](docs/POLICY_BACKTEST.md).

## System flow

```text
Telemetry cycle
      ↓
Persist and assemble latest 30 consecutive cycles
      ↓
Window-HGB inference (default) or GRU (`RUL_MODEL=gru`)
      ↓
Raw RUL + non-negative operational RUL
      ↓
Deterministic risk severity
      ↓
Active alert
      ↓
Critical-condition work order
```

## Repository structure

```text
backend/       FastAPI application, persistence, inference, and decision logic
data/          Local raw/interim/processed data locations (contents ignored)
docs/          EDA, model, API, and engineering result reports
migrations/    Alembic database migrations
ml/            Dataset utilities, baselines, sequence preparation, and GRU
scripts/       Acquisition, analysis, training, and promotion commands
tests/         Data, model, API, migration, and workflow tests
```

## Setup

Python 3.12 is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download and extract the official NASA dataset:

```powershell
.\scripts\acquire_cmapss.ps1
```

Raw data is stored under `data/raw/` and is intentionally excluded from Git.

## Run the backend

Create the local database and start the API:

```powershell
alembic upgrade head
$env:AUTH_PRINCIPALS_JSON = '<configured principal digest records>'
uvicorn backend.app.main:app --reload
```

OpenAPI documentation will be available at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

Replay a historical FD001 unit through the running API at five cycles per second:

```powershell
$env:RELIABILITY_API_TOKEN = '<issued operator or admin credential>'
python scripts/replay_fd001.py --unit-id 81 --cycles-per-second 5
```

Use `--cycles-per-second 0` for an unthrottled replay. Existing assets resume
after their latest stored cycle. Operators can replay registered assets; creating
a missing asset requires an admin credential. See
[`docs/ADJUSTABLE_REPLAY_SERVICE.md`](docs/ADJUSTABLE_REPLAY_SERVICE.md).
Authentication configuration and the role matrix are documented in
[`docs/AUTHENTICATION_AND_RBAC.md`](docs/AUTHENTICATION_AND_RBAC.md).

Run the operations console in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The console connects to `http://127.0.0.1:8000` by default. Configuration and
scope are documented in
[`docs/FRONTEND_OPERATIONS_CONSOLE.md`](docs/FRONTEND_OPERATIONS_CONSOLE.md).

The default database is SQLite under `data/interim/`. To use PostgreSQL:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://user:password@localhost/reliability"
alembic upgrade head
```

## Risk thresholds

Default thresholds are configurable before API startup:

```powershell
$env:RUL_CRITICAL_THRESHOLD = "15"   # work-order lead time in cycles + about 10
$env:RUL_HIGH_THRESHOLD = "30"
$env:RUL_MEDIUM_THRESHOLD = "70"
```

The required ordering is `0 <= critical <= high <= medium`.
Stored predictions keep the severity they were created with; changing thresholds
does not reclassify history.

## Choose the served model

```powershell
$env:RUL_MODEL = "hgb"   # default: window-feature gradient boosting, version 2.0.0
$env:RUL_MODEL = "gru"   # legacy torch GRU, version 1.0.0 (needs torch)
```

The two models have different `model_version` values on purpose: predictions are
unique per asset, cycle and `model_version`. Switching models mid-stream adds
predictions under the new version, which can escalate an existing alert.
`operational_rul` from the HGB model is clipped to 0-125, so values near 125 mean
"healthy, no usable estimate", not a measured remaining life. Each HGB prediction
also lists the top 3 sensors drifting furthest toward their degraded direction
versus a healthy baseline (`drivers`). This is a drift ranking, not a diagnosis.
Pickled models depend on the scikit-learn version: after changing it, run
`python scripts/export_fd001_hgb.py` to rebuild the artifact and manifest. The
service refuses to start if the golden predictions in the manifest do not match.

## Reproduce the ML results

```powershell
python scripts/run_fd001_baselines.py
python scripts/analyze_fd001_sequences.py
python scripts/train_fd001_gru.py
python scripts/promote_fd001_gru.py
python scripts/run_fd001_evaluation.py
python scripts/run_fd001_policy_backtest.py
python scripts/export_fd001_hgb.py
```

The promoted models and its versioned inference manifest are stored in
`ml/artifacts/`.

## Run tests

```powershell
pytest -q
```

The suite covers dataset integrity, leakage-safe splits, model artifacts,
official test-window construction, API validation, migrations, idempotent
telemetry ingestion, prediction persistence, and maintenance workflows.

## PostgreSQL concurrency validation

Run from the repository root with the Python environment activated:

```powershell
docker compose -f deploy/docker-compose.postgres.yml up -d --wait
$env:TEST_DATABASE_URL = 'postgresql+psycopg://reliability:reliability@127.0.0.1:5433/reliability'
python -B -m pytest -q -ra -p no:cacheprovider
python -B scripts/benchmark_postgres.py --assets 4 --cycles 240 --concurrency 4
```

The Compose credentials are disposable local defaults, not production credentials.
PostgreSQL binds only to loopback port 5433. Tests and the benchmark migrate a
unique schema and remove only that schema afterward; they do not migrate the
application database. Without `TEST_DATABASE_URL`, PostgreSQL-specific checks
explicitly skip. The benchmark accepts only a loopback PostgreSQL address.

Stop the local service without removing its volume:

```powershell
docker compose -f deploy/docker-compose.postgres.yml stop
```

See [persistence hardening and measured results](docs/BACKEND_PERSISTENCE.md#postgresql-hardening-validation)
for pool settings, migration compatibility, and benchmark boundaries.

## Dataset

Source: [NASA Prognostics Center of Excellence Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Subset: `FD001` — one operating condition and one HPC degradation fault mode.

Expected files:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

The historical trajectories are used as replayed simulated telemetry and are
not represented as measurements collected from a physical factory.

## Current limitations

- FD001 is simulated and covers one operating condition and fault mode.
- Enterprise SSO and short-lived credential issuance are not implemented.
- The console currently supports workflow transitions but not asset administration.
- Local PostgreSQL concurrency validation is implemented; network/server load, soak testing, and production deployment remain pending.
- Asynchronous processing, durable replay orchestration, and production monitoring remain pending.
- Migration `0004` preserves historical predictions but cannot reconstruct their decision provenance. Retrying a pre-migration prediction returns HTTP 409 requiring reconciliation; no automatic reconciliation tool is provided.
