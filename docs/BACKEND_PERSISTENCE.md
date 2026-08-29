# Phase 4 Persistence Slice

## Scope completed

The synchronous backend now persists the minimum entities required to turn
individual telemetry cycles into an auditable prediction:

- `Asset`
- `TelemetryPointRecord`
- `Prediction`

The schema is SQLAlchemy 2.x and PostgreSQL-compatible. Alembic revision `0001`
creates the three tables, indexes, foreign keys, and idempotency constraints.
SQLite is used only as the default local/test database; production-style local
use should set `DATABASE_URL` to PostgreSQL before running migrations.

## API flow

1. `POST /v1/assets` explicitly registers an asset.
2. `POST /v1/assets/{asset_id}/telemetry` submits one complete telemetry cycle.
3. The `(asset_id, cycle)` pair is unique.
4. Fewer than 30 stored cycles, or a gap in the latest 30, produces no prediction.
5. Once the latest 30 cycles are consecutive, the promoted inference boundary
   calculates RUL and persists one prediction.
6. `(asset_id, window_end_cycle, model_version)` is unique, preventing duplicate
   predictions for an identical model/window boundary.

## Idempotency contract

- Repeating the same asset/cycle with identical values succeeds with
  `created: false`; it does not create another telemetry row or prediction.
- Repeating an asset/cycle with different values returns HTTP 409. Published raw
  telemetry is immutable through this ingestion path.
- Resubmitting cycle 30 returns the already persisted prediction.
- Missing assets return HTTP 404 rather than being created implicitly.

## Migration usage

Local SQLite default:

```powershell
alembic upgrade head
```

PostgreSQL:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://user:password@localhost/reliability"
alembic upgrade head
```

The migration was verified through both `upgrade head` and `downgrade base` on
an isolated database.

## Explicitly deferred

- Editing or deleting telemetry.
- Prediction-history/list endpoints.
- Risk severity and threshold configuration.
- Alerts and work orders.
- Concurrent-ingestion load testing and retry recovery.
- Authentication, streaming, frontend, and deployment topology.

## Verification

Integration tests cover asset creation, 29-cycle no-prediction behavior,
cycle-30 inference and persistence, identical retry idempotency, conflicting
retry rejection, gap suppression, missing assets, and exact database row counts.

BACKEND PERSISTENCE STATUS: COMPLETE
