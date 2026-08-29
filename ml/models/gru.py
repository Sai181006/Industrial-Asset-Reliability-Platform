"""Compact GRU regressor for FD001 sequence RUL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GRURULRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(features)
        return self.head(hidden[-1]).squeeze(-1)


def predict_rul(
    model: GRURULRegressor,
    features: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> torch.Tensor:
    """Return RUL cycles from the model's standardized target output."""
    model.eval()
    with torch.inference_mode():
        return model(features) * target_std + target_mean


def load_gru_artifact(path: str | Path) -> tuple[GRURULRegressor, dict]:
    """Restore the trained model and its complete preprocessing metadata."""
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = GRURULRegressor(**artifact["model_config"])
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model, artifact


def predict_raw_window(model: GRURULRegressor, artifact: dict, window: np.ndarray) -> float:
    """Predict cycles of RUL from one raw, ordered feature window."""
    window = np.asarray(window, dtype=np.float32)
    expected = (artifact["window_length"], len(artifact["feature_columns"]))
    if window.shape != expected:
        raise ValueError(f"Expected raw window shape {expected}, found {window.shape}")
    mean = np.asarray(artifact["input_scaler_mean"], dtype=np.float32)
    scale = np.asarray(artifact["input_scaler_scale"], dtype=np.float32)
    scaled = (window - mean) / scale
    prediction = predict_rul(
        model,
        torch.from_numpy(scaled).unsqueeze(0),
        artifact["target_mean"],
        artifact["target_std"],
    )
    return float(prediction.item())
