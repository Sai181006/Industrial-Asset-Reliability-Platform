# Phase 4 Synchronous Inference Boundary

## Scope completed

The first backend vertical slice exposes one synchronous endpoint:

`POST /v1/predictions/rul`

It validates one asset's ordered 30-cycle telemetry window, verifies and loads
the promoted `fd001-gru-rul` version `1.0.0` artifact, runs inference, and returns
both raw and non-negative operational RUL. Nothing is persisted yet.

## Request contract

- `asset_id`: non-empty string, at most 128 characters.
- `telemetry`: exactly 30 points.
- Every point contains `cycle`, three operational settings, and sensors 1–21.
- Cycles must be ascending, consecutive positive integers.
- Missing, non-finite, or additional fields are rejected.
- Feature order is taken from the promoted manifest, never client ordering.

Example shape:

```json
{
  "asset_id": "engine-001",
  "telemetry": [
    {
      "cycle": 1,
      "operational_setting_1": 0.0,
      "operational_setting_2": 0.0,
      "operational_setting_3": 100.0,
      "sensor_1": 518.67,
      "sensor_2": 641.82
    }
  ]
}
```

The abbreviated point above is illustrative; all 21 sensor fields are required
and exactly 30 complete points must be submitted.

## Response contract

```json
{
  "asset_id": "engine-001",
  "window_start_cycle": 1,
  "window_end_cycle": 30,
  "raw_rul": 123.45,
  "operational_rul": 123.45,
  "model_id": "fd001-gru-rul",
  "model_version": "1.0.0"
}
```

`operational_rul` is `max(raw_rul, 0.0)`. Raw output is preserved so later
persistence and monitoring do not lose model behavior.

## Startup integrity

The inference service refuses to initialize if:

- the manifest is not promoted for backend use,
- the model artifact SHA-256 differs from the manifest,
- artifact and manifest feature order differ.

The model service is cached once per application process; preprocessing is never
refitted at request time.

## Explicitly deferred

- Database and telemetry persistence.
- Prediction history.
- Risk thresholds, alerts, and work orders.
- Authentication/RBAC.
- Streaming, queues, and workers.
- Frontend and deployment architecture.

## Verification

Tests cover successful inference, exact model/version response, operational RUL
policy, short-window rejection, cycle-gap rejection, unknown-field rejection,
and checksum mismatch refusal.

BACKEND INFERENCE BOUNDARY STATUS: COMPLETE
