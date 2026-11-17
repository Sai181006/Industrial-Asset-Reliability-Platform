"""Version-checked adapters between API telemetry and the promoted models.

``RUL_MODEL`` selects the served model: ``hgb`` (default, window-feature
gradient boosting, version 2.0.0) or ``gru`` (the original torch GRU, 1.0.0).
The torch import is lazy so the backend and its tests run without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol
import warnings

import numpy as np

from backend.app.schemas import DriftDriver, RULPredictionRequest, RULPredictionResponse
from ml.features import summarize_window, window_feature_names


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "ml" / "artifacts" / "fd001_gru.pt"
MANIFEST_PATH = ROOT / "ml" / "artifacts" / "fd001_gru_manifest.json"
HGB_ARTIFACT_PATH = ROOT / "ml" / "artifacts" / "fd001_hgb_window.joblib"
HGB_MANIFEST_PATH = ROOT / "ml" / "artifacts" / "fd001_hgb_window_manifest.json"
DRIVER_COUNT = 3
DRIVER_TAIL_CYCLES = 5  # average the newest cycles so one noisy reading cannot dominate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InferenceService(Protocol):
    manifest: dict

    def predict(self, request: RULPredictionRequest) -> RULPredictionResponse: ...


class RULInferenceService:
    def __init__(
        self,
        artifact_path: Path = ARTIFACT_PATH,
        manifest_path: Path = MANIFEST_PATH,
    ) -> None:
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["status"] != "promoted_for_backend_mvp":
            raise RuntimeError("Model manifest is not promoted for backend use")
        if _sha256(artifact_path) != self.manifest["artifact_sha256"]:
            raise RuntimeError("Model artifact checksum does not match manifest")
        from ml.models.gru import load_gru_artifact  # lazy: torch is only needed for the GRU

        self.model, self.artifact = load_gru_artifact(artifact_path)
        manifest_features = self.manifest["input_contract"]["feature_order"]
        if self.artifact["feature_columns"] != manifest_features:
            raise RuntimeError("Artifact feature order does not match manifest")

    def predict(self, request: RULPredictionRequest) -> RULPredictionResponse:
        feature_order = self.artifact["feature_columns"]
        raw_window = np.asarray(
            [
                [getattr(point, feature) for feature in feature_order]
                for point in request.telemetry
            ],
            dtype=np.float32,
        )
        from ml.models.gru import predict_raw_window

        raw_rul = predict_raw_window(self.model, self.artifact, raw_window)
        return RULPredictionResponse(
            asset_id=request.asset_id,
            window_start_cycle=request.telemetry[0].cycle,
            window_end_cycle=request.telemetry[-1].cycle,
            raw_rul=raw_rul,
            operational_rul=max(raw_rul, 0.0),
            model_id=self.manifest["model_id"],
            model_version=self.manifest["version"],
        )


class HGBInferenceService:
    """Window-feature gradient boosting trained on capped RUL (see docs/EVALUATION_RESULTS.md)."""

    def __init__(
        self,
        artifact_path: Path = HGB_ARTIFACT_PATH,
        manifest_path: Path = HGB_MANIFEST_PATH,
    ) -> None:
        import joblib

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["status"] != "promoted_for_backend":
            raise RuntimeError("Model manifest is not promoted for backend use")
        if _sha256(artifact_path) != self.manifest["artifact_sha256"]:
            raise RuntimeError("Model artifact checksum does not match manifest")
        contract = self.manifest["input_contract"]
        self.sensors: list[str] = list(contract["sensors"])
        self.window_length: int = int(contract["window_length"])
        if contract["feature_names"] != window_feature_names(self.sensors):
            raise RuntimeError("Manifest feature names do not match the feature definition")
        self.rul_cap = float(self.manifest["output_contract"]["rul_cap"])
        self.baseline = self.manifest["drift_baseline"]["sensors"]
        if set(self.baseline) != set(self.sensors):
            raise RuntimeError("Drift baseline sensors do not match the input contract")
        self.request_columns: list[str] = list(contract["request_columns"])
        self._indices = [self.request_columns.index(sensor) for sensor in self.sensors]
        self.model = joblib.load(artifact_path)  # trusted: gated by the checksum above
        self._check_environment()
        self._check_golden_predictions()

    def _check_environment(self) -> None:
        import sklearn

        built = self.manifest["environment"]["scikit_learn"]
        if built.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            warnings.warn(
                f"Model built with scikit-learn {built} but running {sklearn.__version__}; "
                "rebuild with scripts/export_fd001_hgb.py if the golden check fails.",
                stacklevel=3,
            )

    def _raw_rul(self, window: np.ndarray) -> float:
        """``window`` holds only the contract sensors, in contract order, as float64."""
        return float(self.model.predict(summarize_window(window)[None, :])[0])

    def _check_golden_predictions(self) -> None:
        golden = self.manifest["golden"]
        columns = self.manifest["input_contract"]["request_columns"]
        for window, expected in zip(golden["windows"], golden["expected_raw_rul"]):
            values = np.asarray(window["values"], dtype=np.float64)
            actual = self._raw_rul(values[:, [columns.index(s) for s in self.sensors]])
            if abs(actual - expected) > golden["tolerance"]:
                raise RuntimeError("Model artifact fails its golden predictions; rebuild it in this environment")

    def drivers(self, window: np.ndarray) -> list[DriftDriver]:
        """Sensors drifting furthest toward their degraded direction versus healthy engines.

        This is a drift ranking against a healthy baseline, not a causal explanation.
        """
        recent = window[-DRIVER_TAIL_CYCLES:].mean(axis=0)
        scores = []
        for position, sensor in enumerate(self.sensors):
            stats = self.baseline[sensor]
            spread = stats["healthy_std"] or 1.0
            scores.append((stats["degradation_sign"] * (recent[position] - stats["healthy_mean"]) / spread, sensor))
        scores.sort(reverse=True)
        return [DriftDriver(sensor=sensor, drift_z=float(score)) for score, sensor in scores[:DRIVER_COUNT]]

    def predict(self, request: RULPredictionRequest) -> RULPredictionResponse:
        full = np.asarray(
            [[getattr(point, column) for column in self.request_columns] for point in request.telemetry],
            dtype=np.float64,
        )
        window = full[:, self._indices]
        raw_rul = self._raw_rul(window)
        return RULPredictionResponse(
            asset_id=request.asset_id,
            window_start_cycle=request.telemetry[0].cycle,
            window_end_cycle=request.telemetry[-1].cycle,
            raw_rul=raw_rul,
            operational_rul=float(min(max(raw_rul, 0.0), self.rul_cap)),
            model_id=self.manifest["model_id"],
            model_version=self.manifest["version"],
            drivers=self.drivers(window),
        )


def create_inference_service() -> InferenceService:
    choice = os.environ.get("RUL_MODEL", "hgb").strip().lower()
    if choice == "hgb":
        return HGBInferenceService()
    if choice == "gru":
        return RULInferenceService()
    raise RuntimeError("RUL_MODEL must be 'hgb' or 'gru'")
