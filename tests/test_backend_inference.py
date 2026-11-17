from pathlib import Path
import asyncio

import httpx
import pytest

from backend.app.inference import HGBInferenceService, RULInferenceService
from backend.app.main import app, get_inference_service
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


@pytest.fixture()
def fresh_service(monkeypatch):
    """Re-create the cached service per test so RUL_MODEL changes take effect."""
    get_inference_service.cache_clear()
    yield monkeypatch
    get_inference_service.cache_clear()


def test_prediction_endpoint_serves_hgb_by_default(payload, fresh_service):
    fresh_service.delenv("RUL_MODEL", raising=False)
    response = post("/v1/predictions/rul", payload)
    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == payload["asset_id"]
    assert body["window_start_cycle"] == 1
    assert body["window_end_cycle"] == 30
    assert body["operational_rul"] == min(max(body["raw_rul"], 0.0), 125.0)
    assert body["model_id"] == "fd001-hgb-window-rul"
    assert body["model_version"] == "2.0.0"
    assert len(body["drivers"]) == 3


def test_gru_remains_selectable(payload, fresh_service):
    pytest.importorskip("torch")
    fresh_service.setenv("RUL_MODEL", "gru")
    response = post("/v1/predictions/rul", payload)
    assert response.status_code == 200
    body = response.json()
    assert body["operational_rul"] == max(body["raw_rul"], 0.0)
    assert body["model_id"] == "fd001-gru-rul"
    assert body["model_version"] == "1.0.0"
    assert body["drivers"] is None


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


def test_hgb_service_rejects_manifest_checksum_mismatch(tmp_path):
    artifact = tmp_path / "bad.joblib"
    artifact.write_bytes(b"not the promoted model")
    with pytest.raises(RuntimeError, match="checksum"):
        HGBInferenceService(artifact_path=artifact)
