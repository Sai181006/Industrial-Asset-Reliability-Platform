"""Select, train, evaluate, and persist one compact FD001 GRU."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import load_fd001
from ml.models.gru import GRURULRegressor, predict_rul
from ml.sequence import prepare_development_sequences


SEED = 42
BATCH_SIZE = 256
MAX_SELECTION_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3


def set_deterministic(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)


def loader(
    features: np.ndarray, targets: np.ndarray, shuffle: bool, seed: int = SEED
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


def train_epochs(
    model: GRURULRegressor,
    train_loader: DataLoader,
    epochs: int,
    validation: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.HuberLoss(delta=1.0)
    train_history, validation_history = [], []
    best_loss, stale = float("inf"), 0
    for _ in range(epochs):
        model.train()
        losses = []
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_history.append(float(np.mean(losses)))
        if validation is not None:
            model.eval()
            with torch.inference_mode():
                validation_loss = float(loss_function(model(validation[0]), validation[1]))
            validation_history.append(validation_loss)
            if validation_loss < best_loss - 1e-4:
                best_loss, stale = validation_loss, 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    return train_history, validation_history


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    low = targets <= 30
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "low_rul_mae": float(mean_absolute_error(targets[low], predictions[low])),
        "low_rul_rmse": float(mean_squared_error(targets[low], predictions[low]) ** 0.5),
        "negative_predictions": int((predictions < 0).sum()),
    }


def main() -> None:
    set_deterministic()
    raw = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData").train
    sequences = prepare_development_sequences(raw, window_length=30)

    # Units 65–80 select epoch count; units 81–100 remain untouched until final evaluation.
    selection_train = sequences.train.unit_ids <= 64
    selection_holdout = sequences.train.unit_ids >= 65
    selection_mean = float(sequences.train.targets[selection_train].mean())
    selection_std = float(sequences.train.targets[selection_train].std())
    selection_targets = (
        sequences.train.targets[selection_train] - selection_mean
    ) / selection_std
    holdout_targets = (
        sequences.train.targets[selection_holdout] - selection_mean
    ) / selection_std

    selection_model = GRURULRegressor(len(FEATURE_COLUMNS))
    selection_train_history, selection_validation_history = train_epochs(
        selection_model,
        loader(sequences.train.features[selection_train], selection_targets, True),
        MAX_SELECTION_EPOCHS,
        (
            torch.from_numpy(sequences.train.features[selection_holdout]),
            torch.from_numpy(holdout_targets),
        ),
    )
    selected_epochs = int(np.argmin(selection_validation_history) + 1)

    # Reinitialize and train all units 1–80 for the selected epoch count.
    set_deterministic()
    target_mean = float(sequences.train.targets.mean())
    target_std = float(sequences.train.targets.std())
    standardized_targets = (sequences.train.targets - target_mean) / target_std
    model = GRURULRegressor(len(FEATURE_COLUMNS))
    full_history, _ = train_epochs(
        model,
        loader(sequences.train.features, standardized_targets, True),
        selected_epochs,
    )

    validation_tensor = torch.from_numpy(sequences.validation.features)
    predictions = predict_rul(model, validation_tensor, target_mean, target_std).numpy()
    result_metrics = metrics(sequences.validation.targets, predictions)

    single = validation_tensor[:1]
    for _ in range(20):
        predict_rul(model, single, target_mean, target_std)
    start = time.perf_counter()
    for _ in range(200):
        predict_rul(model, single, target_mean, target_std)
    single_latency_ms = (time.perf_counter() - start) * 1000 / 200

    artifact_dir = ROOT / "ml" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "fd001_gru.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"input_size": len(FEATURE_COLUMNS), "hidden_size": 32, "dropout": 0.1},
            "feature_columns": FEATURE_COLUMNS,
            "window_length": 30,
            "target_definition": "uncapped_linear_rul",
            "target_mean": target_mean,
            "target_std": target_std,
            "input_scaler_mean": sequences.scaler.mean_.tolist(),
            "input_scaler_scale": sequences.scaler.scale_.tolist(),
        },
        artifact_path,
    )

    output = {
        "configuration": {
            "seed": SEED,
            "window_length": 30,
            "hidden_size": 32,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "HuberLoss(delta=1.0) on standardized target",
            "selection_training_units": list(range(1, 65)),
            "epoch_selection_units": list(range(65, 81)),
            "final_training_units": list(range(1, 81)),
            "validation_units": list(range(81, 101)),
            "selected_epochs": selected_epochs,
        },
        "metrics": result_metrics,
        "single_sequence_latency_ms": single_latency_ms,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "artifact": str(artifact_path.relative_to(ROOT)),
        "selection_train_loss": selection_train_history,
        "selection_holdout_loss": selection_validation_history,
        "full_train_loss": full_history,
    }
    (ROOT / "docs" / "gru_metrics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(selection_train_history) + 1), selection_train_history, label="units 1-64")
    plt.plot(range(1, len(selection_validation_history) + 1), selection_validation_history, label="units 65-80")
    plt.axvline(selected_epochs, color="black", linestyle="--", alpha=0.5, label=f"selected epoch {selected_epochs}")
    plt.xlabel("epoch"); plt.ylabel("standardized Huber loss"); plt.title("GRU epoch selection")
    plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / "docs" / "gru_training_curve.png", dpi=160)
    plt.close()

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
