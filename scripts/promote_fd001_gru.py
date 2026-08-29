"""Run fixed promotion checks without tuning on the official FD001 test labels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import load_fd001
from ml.models.gru import GRURULRegressor, predict_rul
from ml.sequence import prepare_development_sequences
from scripts.train_fd001_gru import loader, metrics, set_deterministic, train_epochs


SEEDS = (7, 42, 77)
FIXED_EPOCHS = 11


def final_test_windows(test: pd.DataFrame, scaler) -> tuple[np.ndarray, np.ndarray]:
    """Create exactly one final 30-cycle window per official test engine."""
    windows, units = [], []
    for unit_id, unit in test.groupby("unit_id", sort=True):
        ordered = unit.sort_values("cycle")
        if len(ordered) < 30:
            raise ValueError(f"Test unit {unit_id} has fewer than 30 cycles")
        raw = ordered[FEATURE_COLUMNS].iloc[-30:]
        windows.append(scaler.transform(raw).astype(np.float32))
        units.append(int(unit_id))
    return np.stack(windows), np.asarray(units, dtype=np.int32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_seed(sequences, seed: int) -> tuple[GRURULRegressor, dict]:
    set_deterministic(seed)
    target_mean = float(sequences.train.targets.mean())
    target_std = float(sequences.train.targets.std())
    standardized = (sequences.train.targets - target_mean) / target_std
    model = GRURULRegressor(len(FEATURE_COLUMNS))
    train_epochs(
        model,
        loader(sequences.train.features, standardized, True, seed=seed),
        FIXED_EPOCHS,
    )
    prediction = predict_rul(
        model,
        torch.from_numpy(sequences.validation.features),
        target_mean,
        target_std,
    ).numpy()
    return model, {
        "seed": seed,
        **metrics(sequences.validation.targets, prediction),
    }


def main() -> None:
    fd001 = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData")
    sequences = prepare_development_sequences(fd001.train, window_length=30)
    seed_results = []
    seed_42_model = None
    for seed in SEEDS:
        model, result = train_seed(sequences, seed)
        seed_results.append(result)
        if seed == 42:
            seed_42_model = model

    if seed_42_model is None:
        raise AssertionError("The predeclared promotion seed 42 was not evaluated")

    test_windows, test_units = final_test_windows(fd001.test, sequences.scaler)
    expected_units = np.arange(1, 101, dtype=np.int32)
    if not np.array_equal(test_units, expected_units):
        raise AssertionError("Official test units are not ordered 1..100")
    target_mean = float(sequences.train.targets.mean())
    target_std = float(sequences.train.targets.std())
    raw_test_prediction = predict_rul(
        seed_42_model, torch.from_numpy(test_windows), target_mean, target_std
    ).numpy()
    operational_test_prediction = np.maximum(raw_test_prediction, 0.0)
    official_targets = fd001.test_rul.to_numpy(dtype=np.float32)

    seed_frame = pd.DataFrame(seed_results)
    stability = {
        metric: {
            "mean": float(seed_frame[metric].mean()),
            "sample_std": float(seed_frame[metric].std(ddof=1)),
            "min": float(seed_frame[metric].min()),
            "max": float(seed_frame[metric].max()),
        }
        for metric in ("mae", "rmse", "low_rul_mae", "low_rul_rmse")
    }
    raw_test_metrics = metrics(official_targets, raw_test_prediction)
    operational_test_metrics = metrics(official_targets, operational_test_prediction)

    # The already-created seed-42 artifact remains the promoted artifact; verify
    # this script's seed-42 state matches it rather than selecting on test results.
    artifact_path = ROOT / "ml" / "artifacts" / "fd001_gru.pt"
    saved = torch.load(artifact_path, map_location="cpu", weights_only=False)
    for key, value in seed_42_model.state_dict().items():
        torch.testing.assert_close(value, saved["model_state"][key])

    manifest = {
        "model_id": "fd001-gru-rul",
        "version": "1.0.0",
        "status": "promoted_for_backend_mvp",
        "created_date": "2026-08-28",
        "artifact": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(artifact_path),
        "dataset": {
            "name": "NASA C-MAPSS FD001",
            "training_units": "1-80",
            "development_validation_units": "81-100",
            "official_test_units": "1-100 final observed endpoint",
        },
        "input_contract": {
            "shape": [30, len(FEATURE_COLUMNS)],
            "feature_order": FEATURE_COLUMNS,
            "dtype": "float32",
            "cycle_order": "ascending, consecutive within one engine",
            "scaling": "artifact StandardScaler statistics; never refit at inference",
        },
        "output_contract": {
            "raw_rul": "unclipped model output retained for monitoring",
            "operational_rul": "max(raw_rul, 0.0) used by deterministic decision logic",
            "unit": "operating cycles",
        },
        "training": {"seeds_checked": list(SEEDS), "promoted_seed": 42, "epochs": FIXED_EPOCHS},
        "development_seed_results": seed_results,
        "development_stability": stability,
        "official_test_seed_42": {
            "raw": raw_test_metrics,
            "operational_non_negative": operational_test_metrics,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    manifest_path = ROOT / "ml" / "artifacts" / "fd001_gru_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "docs" / "promotion_metrics.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
