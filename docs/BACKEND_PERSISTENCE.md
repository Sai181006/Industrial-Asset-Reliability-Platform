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

## PostgreSQL hardening validation

This section supersedes the earlier slice's deferred concurrency and retry scope.
The synchronous architecture and promoted FD001 model remain unchanged.

### Transactions and ordering

Telemetry insertion, inference persistence, risk classification, and workflow
creation now commit as one transaction. All API mutations affecting an existing
asset acquire its PostgreSQL row lock before reading or updating workflow state.
Different assets can proceed concurrently; inference holds the same-asset lock,
so same-asset throughput is deliberately serialized. Asset registration handles
a primary-key race as one HTTP 201 and HTTP 409 for the losing request.

Unique telemetry, prediction, active-alert, and work-order constraints remain
in place. Alert escalation uses refreshed locked state. Workflow transitions
cannot use stale state to skip or reverse a transition. Same-state resolution
preserves the original resolution timestamp. This is not an optimistic version
API: simultaneous valid same-state notes or assignment edits are last-writer-wins.

New predictions persist their decision severity and alert/work-order IDs in the
same transaction. Retrying an existing processed prediction returns those stored
IDs, including after a human resolves the workflow; it does not reopen it or
reclassify it using changed thresholds.

At most three complete transaction attempts are made for PostgreSQL serialization
failure (`40001`), deadlock (`40P01`), or lock timeout (`55P03`), with 50 ms and
100 ms backoff. Other database failures and invalidated connections are not
retried automatically. Exhaustion surfaces an error; no durable retry queue exists.
SQLite retains sequential development/test support, not PostgreSQL row-lock semantics.

### Pool configuration

Configure before API startup. Limits apply per process, not across an entire deployment.

| Variable | Default | Accepted range / unit |
|---|---:|---|
| `DB_POOL_SIZE` | 5 | 1–1000 connections |
| `DB_MAX_OVERFLOW` | 10 | 0–1000 connections |
| `DB_POOL_TIMEOUT` | 30 | 1–3600 seconds |
| `DB_POOL_RECYCLE` | 1800 | -1–604800 seconds; -1 disables recycling |
| `PG_LOCK_TIMEOUT_MS` | 5000 | 1–2147483647 milliseconds |

Connections use pre-ping. PostgreSQL lock timeout is transaction-local. The
benchmark separately sets a 15-second statement timeout and a 5-second connection
timeout; these are not application-wide defaults. No isolation level override
is configured; PostgreSQL tests used its default READ COMMITTED behavior.

### Migration and legacy-data boundary

Revision `0005` adds a nullable JSON `drivers` column to `predictions` (top drifting sensors; null for GRU predictions and history).

Revision `0004` adds `decision_processed`, `decision_alert_id`, and
`decision_work_order_id`. Existing records remain readable, but receive
`decision_processed=false`: the previous separately committed steps cannot prove
whether a decision completed or which shared alert a prediction used.

**Retrying an existing legacy prediction returns HTTP 409 with
`legacy prediction requires decision reconciliation`.** It leaves that record and
its workflows unchanged. There is no automatic backfill or reconciliation command.
Do not set the flag or infer historical links blindly. Plan explicit reconciliation
before deploying this revision over an operational database. New prediction
windows use the new atomic path normally. This compatibility limitation is not a
claim of seamless legacy recovery.

Back up the database and stop writers before applying migrations. Populated
SQLite upgrade/downgrade is tested with foreign-key enforcement enabled and
disabled. For marked SQLite batch revisions, the migration runner temporarily
disables enforcement outside a transaction, checks all references before commit,
and restores the original setting even on failure. Caller-owned active
transactions are rejected for this operation. PostgreSQL constraints are never
disabled. Downgrading removes decision provenance; it is not a lossless rollback
of the new idempotency contract.

### Executed checks

- Full suite with an explicit PostgreSQL test URL: **216 passed**, no skips.
- Without that URL: **189 passed, 27 explicitly skipped** PostgreSQL checks.
- Frontend lint, TypeScript `--noEmit --incremental false`, and production build passed.
- Remaining warnings concern dependency deprecations and Vinext route classification.
- PostgreSQL tests migrate unique schemas, compare ORM metadata, and verify the
  active-alert partial index. Separate request connections exercise registration,
  identical/conflicting telemetry, cycle-30 inference, escalation, workflow races,
  rollback after injected failures, and replay after resolution.
- Real-artifact sequential/parallel smoke runs complement deterministic model doubles.
- Populated SQLite migration tests cover upgrade, downgrade, injected batch failure,
  foreign-key-check failure, and enforcement restoration.
- The original persistence fixture remains SQLite-only; PostgreSQL behavior is
  exercised by the dedicated integration suite, not silently inferred from it.
- Backend lint/typecheck commands are not configured; test success is not a claim
  that Python static analysis was run.

### Local benchmark result

Reproduction from the repository root, using the local Compose service:

```powershell
$env:TEST_DATABASE_URL = 'postgresql+psycopg://reliability:reliability@127.0.0.1:5433/reliability'
python -B scripts/benchmark_postgres.py --assets 4 --cycles 240 --concurrency 4
```

Recorded environment: Windows 11 build 26200, Python 3.12.14, PostgreSQL 17.11
in `postgres:17-alpine`, PyTorch 2.13.0+cpu, eight Torch compute/inter-op threads,
one application process, 40 AnyIO thread tokens, four concurrent assets, default
pool settings above. FD001 training unit 81 was replayed once for each asset,
cycles 1–240 in order. The real GRU v1.0.0 was warmed once before measurement;
authentication used an ephemeral bearer credential with no authorization bypass.

| Measurement | Observed |
|---|---:|
| Telemetry requests | 960 |
| Errors | 0 |
| Measured elapsed | 11.256 seconds |
| Throughput | 85.286 requests/second |
| p50 latency | 46.472 ms |
| p95 latency | 64.431 ms |
| Maximum latency | 134.517 ms |
| Persisted assets / telemetry / predictions | 4 / 960 / 844 |
| Persisted alerts / work orders | 4 / 4 |

Counts, decision responses, stored decision linkage, same-asset relationships,
and final escalation state all matched. The generated schema was removed.
Percentiles use linear interpolation at `(n-1)*p`.

These measurements include in-process ASGI dispatch, real authentication,
PostgreSQL I/O, and inference, but **not network HTTP, TLS, an external server,
or a production worker topology**. Setup, migrations, warmup, final verification,
and cleanup are excluded from measured throughput. A frontend build ran on the
same host during this recorded run, so it is not a controlled capacity baseline.
This is a bounded local correctness/performance smoke benchmark, not a service
SLA or proof of saturation capacity. Longer soaks, network load, and failure
recovery under a deployed topology remain future validation.

### Phase boundary

The local PostgreSQL concurrency and benchmark work is implemented and verified.
Production deployment remains next, subject to review of the legacy-data migration
policy. No broker, deployment pipeline, enterprise identity system, or physical
industrial-fleet validation is implied by these results.
