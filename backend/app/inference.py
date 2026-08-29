"""Version-checked adapter between API telemetry and the promoted GRU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from backend.app.schemas import RULPredictionRequest, RULPredictionResponse
from ml.models.gru import load_gru_artifact, predict_raw_window


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "ml" / "artifacts" / "fd001_gru.pt"
MANIFEST_PATH = ROOT / "ml" / "artifacts" / "fd001_gru_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

