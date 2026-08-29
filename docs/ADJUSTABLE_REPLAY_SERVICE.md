# Adjustable-Speed FD001 Replay Service

## Status

The replay client streams one official FD001 training trajectory through the
backend telemetry API at a configurable rate. It uses the same HTTP boundary as
an external telemetry producer, so asset registration, validation, inference,
risk decisions, alerts, and work orders are exercised without bypassing the API.

Each invocation receives a UUID run ID and emits a JSON completion summary with
submitted/skipped cycle counts, prediction count, cycle range, and timestamps.
Existing assets resume after their latest stored cycle by default. Backend
idempotency remains the final protection against duplicate telemetry.

## Usage

Start the backend, then run:

```powershell
python scripts/replay_fd001.py --unit-id 81 --cycles-per-second 5
```

Useful options:

- `--cycles-per-second 0` replays without throttling.
- `--asset-id` and `--asset-name` override generated asset identity.
- `--base-url` targets a backend on another host or port.
- `--no-resume` submits every cycle and relies on API idempotency.
- `--raw-dir` selects an installed official C-MAPSS directory.

The source is historical NASA simulation data and must not be described as live
measurements from a physical factory.

## Operational boundary

The replay process is intentionally external to the API process. Stopping it
does not affect backend availability, and restarting the command safely resumes
an incomplete run. Durable scheduling, distributed workers, and live pause or
cancel controls remain part of the later asynchronous-processing phase.

ADJUSTABLE-SPEED REPLAY STATUS: COMPLETE
