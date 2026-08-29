from pathlib import Path

import numpy as np
import pytest
import torch

from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import load_fd001
from ml.models.gru import GRURULRegressor, load_gru_artifact, predict_raw_window, predict_rul


def test_gru_output_shape_and_denormalization():
    torch.manual_seed(42)
    model = GRURULRegressor(input_size=24, hidden_size=8, dropout=0.0)
    features = torch.zeros(5, 30, 24)
    normalized = model(features)
    prediction = predict_rul(model, features, target_mean=100.0, target_std=50.0)
    assert normalized.shape == (5,)
    assert prediction.shape == (5,)
    torch.testing.assert_close(prediction, normalized * 50.0 + 100.0)


def test_saved_artifact_supports_raw_window_inference():
    root = Path(__file__).parents[1]
    artifact_path = root / "ml" / "artifacts" / "fd001_gru.pt"
    raw_dir = root / "data" / "raw" / "source" / "CMAPSSData"
    if not artifact_path.exists() or not raw_dir.exists():
        pytest.skip("Trained GRU artifact or FD001 data is unavailable")
    model, artifact = load_gru_artifact(artifact_path)
    train = load_fd001(raw_dir).train
    raw_window = train[train.unit_id == 81].sort_values("cycle")[FEATURE_COLUMNS].iloc[:30].to_numpy()
    prediction = predict_raw_window(model, artifact, raw_window)
    assert np.isfinite(prediction)


def test_raw_window_inference_rejects_wrong_shape():
    root = Path(__file__).parents[1]
    artifact_path = root / "ml" / "artifacts" / "fd001_gru.pt"
    if not artifact_path.exists():
        pytest.skip("Trained GRU artifact is unavailable")
    model, artifact = load_gru_artifact(artifact_path)
    with pytest.raises(ValueError, match="Expected raw window shape"):
        predict_raw_window(model, artifact, np.zeros((29, 24), dtype=np.float32))

