# Deterministic Decision, Alert, and Work-Order Layer

## Risk mapping

Operational RUL is mapped deterministically:

| Operational RUL | Severity |
|---|---|
| `<= RUL_CRITICAL_THRESHOLD` | CRITICAL |
| `<= RUL_HIGH_THRESHOLD` | HIGH |
| `<= RUL_MEDIUM_THRESHOLD` | MEDIUM |
| above medium threshold | NORMAL |

Defaults are 10, 30, and 70 cycles. Environment overrides are accepted only
when `0 <= critical <= high <= medium`. The persisted prediction records the
resulting severity.

## Workflow creation

- NORMAL creates no workflow record.
- MEDIUM or HIGH creates one active alert if none exists.
- Severity may escalate on an existing active alert; it never automatically
  downgrades an active human workflow.
- CRITICAL creates one work order for the active alert.
- A partial unique database index permits at most one active alert per asset.
- A unique `alert_id` on work orders permits at most one work order per alert.
- Retrying the same prediction/window returns the same alert and work order.

Predictions do not automatically resolve alerts or maintenance work. Resolution
represents an operator/engineer action and must follow the API state machines.

## State machines

Alert:

```text
OPEN → ACKNOWLEDGED → RESOLVED
```

Work order:

```text
OPEN → ASSIGNED → IN_PROGRESS → RESOLVED
```

- Repeating the current state is idempotent.
- Skipping a state returns HTTP 409.
- `ASSIGNED` requires `assigned_to`.
- A critical alert cannot resolve until its work order is RESOLVED.

## Endpoints

- `GET /v1/alerts`
- `PATCH /v1/alerts/{alert_id}`
- `GET /v1/work-orders`
- `PATCH /v1/work-orders/{work_order_id}`

The telemetry-ingestion response now includes the decision severity and any
alert/work-order identifiers created or reused.

## Persistence and migration

Alembic revision `0002` adds prediction severity, alerts, work orders, foreign
keys, indexes, and uniqueness controls. Upgrade, schema-drift check, and complete
downgrade were verified.

## Subsequent integration

Authentication, role authorization, and actor audit fields are now implemented
as documented in `AUTHENTICATION_AND_RBAC.md`.

## Deferred

- Enterprise identity provider integration and technician directory.
- Configurable thresholds stored per asset or plant.
- Alert filtering, pagination, and detailed history endpoints.
- Notifications.
- Asynchronous processing and broker delivery.
- Asset administration screens.

DECISION AND WORKFLOW STATUS: COMPLETE
