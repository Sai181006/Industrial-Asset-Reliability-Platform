from pathlib import Path
import asyncio

import httpx
import pytest

from backend.app.inference import RULInferenceService
from backend.app.main import app
from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import load_fd001


ROOT = Path(__file__).parents[1]
RAW_DIR = ROOT / "data" / "raw" / "source" / "CMAPSSData"


@pytest.fixture(scope="module")
def payload():
    if not RAW_DIR.exists():
        pytest.skip("Official FD001 files are not installed")
    train = load_fd001(RAW_DIR).train
    unit = train[train.unit_id == 81].sort_values("cycle").iloc[:30]
    telemetry = []
    for _, row in unit.iterrows():
        point = {"cycle": int(row.cycle)}
        point.update({feature: float(row[feature]) for feature in FEATURE_COLUMNS})
        telemetry.append(point)
    return {"asset_id": "fd001-unit-81", "telemetry": telemetry}


def post(path: str, payload: dict) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(path, json=payload)

    return asyncio.run(request())


def test_prediction_endpoint_uses_promoted_contract(payload):
    response = post("/v1/predictions/rul", payload)
    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == payload["asset_id"]
    assert body["window_start_cycle"] == 1
    assert body["window_end_cycle"] == 30
    assert body["operational_rul"] == max(body["raw_rul"], 0.0)
    assert body["model_id"] == "fd001-gru-rul"
    assert body["model_version"] == "1.0.0"


def test_endpoint_rejects_wrong_window_length(payload):
    invalid = {**payload, "telemetry": payload["telemetry"][:-1]}
    response = post("/v1/predictions/rul", invalid)
    assert response.status_code == 422


def test_endpoint_rejects_cycle_gap(payload):
    telemetry = [dict(point) for point in payload["telemetry"]]
    telemetry[12]["cycle"] += 1
    response = post("/v1/predictions/rul", {**payload, "telemetry": telemetry})
    assert response.status_code == 422


def test_endpoint_rejects_unknown_sensor_field(payload):
    telemetry = [dict(point) for point in payload["telemetry"]]
    telemetry[0]["sensor_22"] = 1.0
    response = post("/v1/predictions/rul", {**payload, "telemetry": telemetry})
    assert response.status_code == 422


def test_service_rejects_manifest_checksum_mismatch(tmp_path):
    artifact = tmp_path / "bad.pt"
    artifact.write_bytes(b"not the promoted model")
    with pytest.raises(RuntimeError, match="checksum"):
        RULInferenceService(artifact_path=artifact)
