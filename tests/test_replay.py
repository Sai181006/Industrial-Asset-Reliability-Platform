import asyncio

import httpx
import pytest

from backend.app.replay import replay_trajectory


def test_replay_registers_asset_submits_cycles_and_reports_predictions():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/v1/assets":
            return httpx.Response(201, json={"asset_id": "replay", "name": "Replay"})
        cycle = len([item for item in requests if item[1].endswith("/telemetry")])
        return httpx.Response(200, json={"prediction": {"cycle": cycle} if cycle == 2 else None})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
            return await replay_trajectory(
                client,
                [{"cycle": 1}, {"cycle": 2}],
                asset_id="replay",
                asset_name="Replay",
                unit_id=7,
                cycles_per_second=0,
                run_id="run-1",
            )

    result = asyncio.run(run())
    assert result.run_id == "run-1"
    assert result.cycles_submitted == 2
    assert result.predictions_received == 1
    assert requests == [
        ("POST", "/v1/assets"),
        ("POST", "/v1/assets/replay/telemetry"),
        ("POST", "/v1/assets/replay/telemetry"),
    ]


def test_replay_resumes_after_latest_persisted_cycle():
    submitted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/assets":
            return httpx.Response(409, json={"detail": "asset already exists"})
        if request.method == "GET":
            return httpx.Response(200, json={"latest_cycle": 2})
        submitted.append(request.url.path)
        return httpx.Response(200, json={"prediction": None})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
            return await replay_trajectory(
                client,
                [{"cycle": 1}, {"cycle": 2}, {"cycle": 3}],
                asset_id="existing",
                asset_name="Existing",
                unit_id=1,
                cycles_per_second=0,
            )

    result = asyncio.run(run())
    assert result.cycles_skipped == 2
    assert result.cycles_submitted == 1
    assert result.first_cycle == 3
    assert submitted == ["/v1/assets/existing/telemetry"]


def test_replay_validates_rate_and_trajectory():
    async def run(trajectory, rate):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)), base_url="http://test") as client:
            return await replay_trajectory(
                client,
                trajectory,
                asset_id="asset",
                asset_name="Asset",
                unit_id=1,
                cycles_per_second=rate,
            )

    with pytest.raises(ValueError, match="non-negative"):
        asyncio.run(run([{"cycle": 1}], -1))
    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(run([], 1))
