"""Adjustable-speed FD001 telemetry replay client for the backend API."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from urllib.parse import quote
from uuid import uuid4

import httpx

from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import load_fd001


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    asset_id: str
    unit_id: int
    status: str
    first_cycle: int | None
    last_cycle: int | None
    cycles_submitted: int
    cycles_skipped: int
    predictions_received: int
    started_at: str
    completed_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_training_trajectory(raw_dir: str | Path, unit_id: int) -> list[dict[str, float | int]]:
    """Load one published FD001 training trajectory as API telemetry payloads."""
    train = load_fd001(raw_dir).train
    available = sorted(int(value) for value in train["unit_id"].unique())
    if unit_id not in available:
        raise ValueError(f"FD001 training unit {unit_id} not found; expected one of {available[0]}-{available[-1]}")
    unit = train[train["unit_id"] == unit_id].sort_values("cycle")
    return [
        {
            "cycle": int(row.cycle),
            **{feature: float(row[feature]) for feature in FEATURE_COLUMNS},
        }
        for _, row in unit.iterrows()
    ]


async def replay_trajectory(
    client: httpx.AsyncClient,
    trajectory: list[dict[str, float | int]],
    *,
    asset_id: str,
    asset_name: str,
    unit_id: int,
    cycles_per_second: float,
    resume: bool = True,
    run_id: str | None = None,
    bearer_token: str | None = None,
) -> ReplayResult:
    """Register an asset and replay cycles at a fixed rate (zero means unthrottled)."""
    if cycles_per_second < 0:
        raise ValueError("cycles_per_second must be non-negative")
    if not trajectory:
        raise ValueError("trajectory must contain at least one cycle")

    run_id = run_id or str(uuid4())
    if bearer_token:
        client.headers["Authorization"] = f"Bearer {bearer_token}"
    started = datetime.now(timezone.utc)
    asset_path = f"/v1/assets/{quote(asset_id, safe='')}"
    detail = await client.get(asset_path)
    if detail.status_code == 404:
        asset_response = await client.post("/v1/assets", json={"asset_id": asset_id, "name": asset_name})
        if asset_response.status_code == 409:
            detail = await client.get(asset_path)
            detail.raise_for_status()
        else:
            asset_response.raise_for_status()
            detail = None
    else:
        detail.raise_for_status()

    latest_cycle = 0
    if resume and detail is not None:
        latest_cycle = detail.json()["latest_cycle"] or 0

    pending = [point for point in trajectory if int(point["cycle"]) > latest_cycle]
    skipped = len(trajectory) - len(pending)
    predictions = 0
    interval = 1.0 / cycles_per_second if cycles_per_second else 0.0
    next_submission = monotonic()

    for point in pending:
        if interval:
            delay = next_submission - monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        response = await client.post(f"{asset_path}/telemetry", json=point)
        response.raise_for_status()
        if response.json().get("prediction") is not None:
            predictions += 1
        next_submission += interval

    completed = datetime.now(timezone.utc)
    return ReplayResult(
        run_id=run_id,
        asset_id=asset_id,
        unit_id=unit_id,
        status="COMPLETED",
        first_cycle=int(pending[0]["cycle"]) if pending else None,
        last_cycle=int(pending[-1]["cycle"]) if pending else None,
        cycles_submitted=len(pending),
        cycles_skipped=skipped,
        predictions_received=predictions,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
    )
