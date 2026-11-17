import asyncio
import json

import httpx
import pytest

from backend.app.replay import replay_trajectory


def test_replay_registers_asset_submits_cycles_and_reports_predictions():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            assert request.url.path == "/v1/assets/replay"
            return httpx.Response(404, json={"detail": "asset not found"})
        if request.url.path == "/v1/assets":
            assert json.loads(request.content) == {"asset_id": "replay", "name": "Replay"}
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
        ("GET", "/v1/assets/replay"),
        ("POST", "/v1/assets"),
        ("POST", "/v1/assets/replay/telemetry"),
        ("POST", "/v1/assets/replay/telemetry"),
    ]


@pytest.mark.parametrize("resume, latest_cycle, expected_cycles", [(True, 2, [3]), (False, 2, [1, 2, 3]), (True, None, [1, 2, 3]), (True, 3, [])])
def test_replay_existing_operator_asset_without_creation(resume, latest_cycle, expected_cycles):
    requests = []
    submitted = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-operator-token"
        requests.append((request.method, request.url.path))
        if request.url.path == "/v1/assets":
            return httpx.Response(403, json={"detail": "admin role required"})
        if request.method == "GET":
            return httpx.Response(200, json={"latest_cycle": latest_cycle})
        submitted.append(json.loads(request.content)["cycle"])
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
                resume=resume,
                bearer_token="test-operator-token",
            )

    result = asyncio.run(run())
    assert result.cycles_skipped == 3 - len(expected_cycles)
    assert result.cycles_submitted == len(expected_cycles)
    assert result.first_cycle == (expected_cycles[0] if expected_cycles else None)
    assert result.last_cycle == (expected_cycles[-1] if expected_cycles else None)
    assert submitted == expected_cycles
    assert requests == [("GET", "/v1/assets/existing")] + [
        ("POST", "/v1/assets/existing/telemetry") for _ in expected_cycles
    ]


@pytest.mark.parametrize("resume", [True, False])
def test_replay_reads_detail_after_creation_conflict(resume):
    requests = []
    submitted = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if len(requests) == 1:
            return httpx.Response(404, json={"detail": "asset not found"})
        if request.url.path == "/v1/assets":
            return httpx.Response(409, json={"detail": "asset already exists"})
        if request.method == "GET":
            return httpx.Response(200, json={"latest_cycle": 2})
        submitted.append(json.loads(request.content)["cycle"])
        return httpx.Response(200, json={"prediction": None})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
            return await replay_trajectory(
                client,
                [{"cycle": 1}, {"cycle": 2}, {"cycle": 3}],
                asset_id="race",
                asset_name="Race",
                unit_id=1,
                cycles_per_second=0,
                resume=resume,
            )

    result = asyncio.run(run())
    expected_cycles = [3] if resume else [1, 2, 3]
    assert submitted == expected_cycles
    assert result.cycles_submitted == len(expected_cycles)
    assert result.cycles_skipped == 3 - len(expected_cycles)
    assert requests == [
        ("GET", "/v1/assets/race"),
        ("POST", "/v1/assets"),
        ("GET", "/v1/assets/race"),
    ] + [("POST", "/v1/assets/race/telemetry") for _ in expected_cycles]


@pytest.mark.parametrize("resume", [True, False])
@pytest.mark.parametrize(
    "statuses",
    [(401,), (403,), (500,), (404, 401), (404, 403), (404, 500),
     (404, 409, 401), (404, 409, 403), (404, 409, 404), (404, 409, 500)],
)
def test_replay_asset_errors_do_not_ingest(statuses, resume):
    requests = []
    expected_requests = [
        ("GET", "/v1/assets/asset"),
        ("POST", "/v1/assets"),
        ("GET", "/v1/assets/asset"),
    ][:len(statuses)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-operator-token"
        requests.append((request.method, request.url.path))
        assert requests == expected_requests[:len(requests)]
        return httpx.Response(statuses[len(requests) - 1], json={"detail": "asset request failed"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
            return await replay_trajectory(
                client,
                [{"cycle": 1}],
                asset_id="asset",
                asset_name="Asset",
                unit_id=1,
                cycles_per_second=0,
                resume=resume,
                bearer_token="test-operator-token",
            )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(run())
    assert exc_info.value.response.status_code == statuses[-1]
    assert requests == expected_requests


@pytest.mark.parametrize("asset_statuses", [(200,), (404, 201), (404, 409, 200)])
def test_replay_encodes_asset_detail_and_telemetry_paths(asset_statuses):
    asset_id = "engine /?#% ü"
    asset_path = b"/v1/assets/engine%20%2F%3F%23%25%20%C3%BC"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.raw_path))
        assert not request.url.query
        assert not request.url.fragment
        if request.url.path == "/v1/assets":
            assert json.loads(request.content) == {"asset_id": asset_id, "name": "Engine"}
        if len(requests) <= len(asset_statuses):
            return httpx.Response(asset_statuses[len(requests) - 1], json={"latest_cycle": None})
        return httpx.Response(200, json={"prediction": None})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
            return await replay_trajectory(
                client,
                [{"cycle": 1}],
                asset_id=asset_id,
                asset_name="Engine",
                unit_id=1,
                cycles_per_second=0,
            )

    result = asyncio.run(run())
    assert result.asset_id == asset_id
    assert result.cycles_submitted == 1
    assert requests == [
        ("GET", asset_path),
        ("POST", b"/v1/assets"),
        ("GET", asset_path),
    ][:len(asset_statuses)] + [("POST", asset_path + b"/telemetry")]


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
