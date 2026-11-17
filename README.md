# Industrial Asset Reliability Platform

An end-to-end predictive-maintenance platform that converts machine telemetry
into Remaining Useful Life (RUL) estimates, deterministic risk levels, alerts,
and maintenance work orders.

The current implementation uses the NASA C-MAPSS FD001 turbofan dataset and a
compact GRU model behind a synchronous FastAPI and SQLAlchemy backend.

## What is implemented

- Official NASA C-MAPSS FD001 acquisition and validation.
- Reproducible EDA and leakage-safe engine-level splits.
- Median, linear-regression, random-forest, and gradient-boosting baselines.
- A promoted 30-cycle GRU RUL model.
- Strict telemetry-window validation and versioned model inference.
- Asset, telemetry, prediction, alert, and work-order persistence.
- Deterministic risk thresholds and guarded maintenance state transitions.
- Full HTTP replay verification using an FD001 engine trajectory.
- Adjustable-speed, resumable FD001 telemetry replay with per-run summaries.
- Responsive fleet operations console for RUL, risk, alerts, and work orders.
- Default-deny bearer authentication with viewer, operator, and admin roles.
- Authenticated-actor audit fields for maintenance workflow changes.
- Role-aware console actions for alert and work-order state transitions.

## Model performance

All rows are scored on the official FD001 test endpoints (100 engines) with one
shared protocol; see [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md).
Models other than the last row are trained on units 1-80, like the GRU.

| Model | Features | RUL target | Test RMSE | Test MAE | NASA score |
|---|---|---|---:|---:|---:|
| Promoted GRU (serving model) | 24 raw | uncapped | 25.28 | 18.31 | not computed |
| Random forest | 24 raw | uncapped | 23.44 | 16.91 | 3,763 |
| Random forest, window features | 14 sensors | capped at 125 | 13.47 | 10.17 | 246 |
| Histogram gradient boosting, window features | 14 sensors | capped at 125 | 14.61 | 11.28 | 295 |
| **Same, refit on all 100 engines (selected by CV)** | 14 sensors | capped at 125 | **14.35** | **10.76** | **287** |

The refit model's RMSE has a bootstrap 95% interval of 12.0-16.5. The promoted GRU
does not beat simple tree models on this protocol, and the backend still serves it.
Earlier validation tables in `docs/GRU_RESULTS.md` and `docs/BASELINE_RESULTS.md`
use different row sets and are not comparable with this table.

## System flow

## Maintenance-policy back-test

Replaying all 100 training engines with out-of-fold predictions, a model-driven
maintenance rule costs 35.1% less per operating cycle than the best fixed
interval (95% interval 19.8% to 48.3%), assuming a failure costs 10x a planned
job and work orders take 5 cycles to execute. The default critical threshold of
10 cycles is only safe when work orders execute almost immediately: with a
5-cycle lead it fails 7 engines in 100. These costs are illustrative assumptions
on simulated data. See [`docs/POLICY_BACKTEST.md`](docs/POLICY_BACKTEST.md).

```text
Telemetry cycle
      ↓
Persist and assemble latest 30 consecutive cycles
      ↓
Promoted GRU inference
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
after their latest stored cycle. See
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
$env:RUL_CRITICAL_THRESHOLD = "10"
$env:RUL_HIGH_THRESHOLD = "30"
$env:RUL_MEDIUM_THRESHOLD = "70"
```

The required ordering is `0 <= critical <= high <= medium`.

## Reproduce the ML results

```powershell
python scripts/run_fd001_baselines.py
python scripts/analyze_fd001_sequences.py
python scripts/train_fd001_gru.py
python scripts/promote_fd001_gru.py
```

The promoted model and its versioned inference manifest are stored in
python scripts/run_fd001_evaluation.py
python scripts/run_fd001_policy_backtest.py
python scripts/run_fd001_evaluation.py
`ml/artifacts/`.

## Run tests

```powershell
pytest -q
```

The suite covers dataset integrity, leakage-safe splits, model artifacts,
official test-window construction, API validation, migrations, idempotent
telemetry ingestion, prediction persistence, and maintenance workflows.

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
- PostgreSQL concurrency/load testing and asynchronous processing are pending.
