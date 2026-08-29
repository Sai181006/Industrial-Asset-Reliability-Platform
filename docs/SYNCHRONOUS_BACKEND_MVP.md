# Synchronous Backend MVP

## Status

The synchronous backend MVP is complete. A client can register an asset, submit
individual telemetry cycles, obtain persisted model predictions, observe
deterministic risk, and operate alert/work-order state through read/write APIs.

## API coverage

Assets:

- create, list, detail, and rename;
- detail includes counts, latest cycle/prediction, active alert, and active work order.

Telemetry and predictions:

- submit one telemetry point;
- retrieve bounded recent telemetry;
- retrieve bounded prediction history;
- direct promoted-model 30-cycle inference remains available.

Workflow:

- filtered/list and individual alert reads;
- alert acknowledge/resolve transitions;
- filtered/list and individual work-order reads;
- work-order assign/start/resolve transitions.

## End-to-end replay evidence

FD001 training unit 81 was replayed through the HTTP ingestion endpoint as 240
individual cycles. Observed results:

- 240 telemetry rows persisted;
- predictions began at cycle 30;
- 211 predictions persisted through cycle 240;
- final prediction severity was CRITICAL;
- one active critical alert existed;
- one OPEN work order existed;
- asset detail, recent five cycles, latest five predictions, and filtered
  workflow endpoints returned consistent persisted state.

The replay uses historical NASA simulation data and must be described as replayed
telemetry, not measurements from a physical factory.

## Current endpoints

```text
POST  /v1/assets
GET   /v1/assets
GET   /v1/assets/{asset_id}
PATCH /v1/assets/{asset_id}
POST  /v1/assets/{asset_id}/telemetry
GET   /v1/assets/{asset_id}/telemetry
GET   /v1/assets/{asset_id}/predictions
POST  /v1/predictions/rul
GET   /v1/alerts
GET   /v1/alerts/{alert_id}
PATCH /v1/alerts/{alert_id}
GET   /v1/work-orders
GET   /v1/work-orders/{work_order_id}
PATCH /v1/work-orders/{work_order_id}
```

Nine OpenAPI path templates cover these method operations.

## Deferred after MVP

- Authentication and actor audit identity.
- Frontend operations console.
- Adjustable-speed replay process and run IDs.
- Notifications and real-time UI updates.
- Event broker/worker architecture.
- PostgreSQL concurrency/load tests and deployment.

SYNCHRONOUS BACKEND MVP STATUS: COMPLETE
