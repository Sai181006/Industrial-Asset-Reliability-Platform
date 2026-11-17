"""Torch-free checks for the window-HGB serving path; golden windows live in its manifest."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.inference import HGB_MANIFEST_PATH, HGBInferenceService, create_inference_service
from backend.app.schemas import RULPredictionRequest
from ml.features import build_samples


@pytest.fixture(scope="module")
def service():
    return HGBInferenceService()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(HGB_MANIFEST_PATH.read_text(encoding="utf-8"))


def request_for(manifest, index):
    window = manifest["golden"]["windows"][index]
    columns = manifest["input_contract"]["request_columns"]
    telemetry = [
        {"cycle": cycle, **dict(zip(columns, map(float, row)))}
        for cycle, row in zip(window["cycles"], window["values"])
    ]
    return RULPredictionRequest(asset_id=f"golden-{index}", telemetry=telemetry)


def test_golden_predictions_match_manifest(service, manifest):
    for index, expected in enumerate(manifest["golden"]["expected_raw_rul"]):
        assert service.predict(request_for(manifest, index)).raw_rul == pytest.approx(expected, abs=1e-6)


def test_served_features_equal_training_features(service, manifest):
    """The service must see exactly the features the evaluation harness trained on."""
    from ml.features import summarize_window

    columns = manifest["input_contract"]["request_columns"]
    for index in range(len(manifest["golden"]["windows"])):
        window = manifest["golden"]["windows"][index]
        frame = pd.DataFrame(window["values"], columns=columns)
        frame.insert(0, "cycle", window["cycles"])
        frame.insert(0, "unit_id", 1)
        reference = build_samples(frame, [1], service.sensors, "window", endpoints="last").features[0]
        served = summarize_window(np.asarray(window["values"])[:, service._indices])
        np.testing.assert_array_equal(served, reference)


def test_output_is_clipped_to_the_cap_and_floor(service, manifest):
    response = service.predict(request_for(manifest, 0))
    assert 0.0 <= response.operational_rul <= 125.0
    assert response.operational_rul == min(max(response.raw_rul, 0.0), 125.0)
    assert (response.model_id, response.model_version) == ("fd001-hgb-window-rul", "2.0.0")


def test_drivers_rank_drift_toward_degradation(service, manifest):
    """Golden window 2 is the last 30 cycles of a training engine, so strong drift must show up."""
    response = service.predict(request_for(manifest, 2))
    assert response.operational_rul < 10
    assert len(response.drivers) == 3
    scores = [driver.drift_z for driver in response.drivers]
    assert scores == sorted(scores, reverse=True) and scores[0] > 2.5
    assert {d.sensor for d in response.drivers} <= set(service.sensors)


def test_drivers_are_small_for_a_healthy_window(service, manifest):
    """Replaying the baseline mean produces no drift."""
    healthy = [[service.baseline[s]["healthy_mean"] for s in service.sensors]] * 30
    assert max(abs(d.drift_z) for d in service.drivers(np.asarray(healthy))) < 1e-9


def test_tampered_golden_prediction_is_refused(tmp_path, manifest):
    broken = copy.deepcopy(manifest)
    broken["golden"]["expected_raw_rul"][0] += 5.0
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RuntimeError, match="golden"):
        HGBInferenceService(manifest_path=path)


def test_unpromoted_manifest_is_refused(tmp_path, manifest):
    broken = {**manifest, "status": "draft"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not promoted"):
        HGBInferenceService(manifest_path=path)


def test_model_choice_must_be_known(monkeypatch):
    monkeypatch.setenv("RUL_MODEL", "xgboost")
    with pytest.raises(RuntimeError, match="RUL_MODEL"):
        create_inference_service()


def test_versions_differ_between_models(manifest):
    gru = json.loads((Path(HGB_MANIFEST_PATH).parent / "fd001_gru_manifest.json").read_text(encoding="utf-8"))
    # predictions are unique on (asset, cycle, model_version); a shared version would collide
    assert manifest["version"] != gru["version"]
