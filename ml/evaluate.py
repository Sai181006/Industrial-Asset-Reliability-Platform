"""One evaluation protocol for every FD001 RUL model.

Headline numbers are computed on the official test endpoints (one final window
per test engine). Model selection uses only grouped cross-validation on the
training engines, scored on seeded truncated endpoints, never on test labels.
Predictions are clipped at zero, matching the operational contract of the
promoted model, and compared against the uncapped true RUL.
"""

from __future__ import annotations

import numpy as np


LOW_RUL_THRESHOLD = 30


def nasa_score(true_rul: np.ndarray, predicted_rul: np.ndarray) -> float:
    """Asymmetric PHM08 score: late predictions are penalized more. Lower is better."""
    error = np.asarray(predicted_rul, dtype=np.float64) - np.asarray(true_rul, dtype=np.float64)
    return float(np.sum(np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)))


def endpoint_metrics(
    true_rul: np.ndarray,
    raw_prediction: np.ndarray,
    low_threshold: int = LOW_RUL_THRESHOLD,
    truth_cap: float | None = None,
) -> dict[str, float | int]:
    """Score clipped (operational) predictions against true RUL.

    ``truth_cap`` optionally clips the *evaluation* truth at a horizon beyond
    which sensors carry no information. It is used for cross-validation only;
    the official test endpoints are always scored against unmodified truth.
    """
    true_rul = np.asarray(true_rul, dtype=np.float64)
    if truth_cap is not None:
        true_rul = np.minimum(true_rul, truth_cap)
    raw_prediction = np.asarray(raw_prediction, dtype=np.float64)
    if true_rul.shape != raw_prediction.shape or true_rul.ndim != 1 or len(true_rul) == 0:
        raise ValueError("true and predicted RUL must be non-empty 1-D arrays of equal length")
    prediction = np.maximum(raw_prediction, 0.0)
    error = prediction - true_rul
    low = true_rul <= low_threshold
    return {
        "n": int(len(true_rul)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_score": nasa_score(true_rul, prediction),
        "low_rul_mae": float(np.mean(np.abs(error[low]))) if low.any() else float("nan"),
        "low_rul_count": int(low.sum()),
        "negative_raw_predictions": int((raw_prediction < 0).sum()),
    }


def bootstrap_rmse_interval(
    true_rul: np.ndarray,
    raw_prediction: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over engines (one row per engine at the test endpoint)."""
    true_rul = np.asarray(true_rul, dtype=np.float64)
    error = np.maximum(np.asarray(raw_prediction, dtype=np.float64), 0.0) - true_rul
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(error), size=(n_boot, len(error)))
    rmse = np.sqrt((error[draws] ** 2).mean(axis=1))
    tail = (1.0 - level) / 2.0
    return float(np.quantile(rmse, tail)), float(np.quantile(rmse, 1.0 - tail))


def grouped_folds(
    unit_ids: list[int], n_splits: int = 5, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Deterministic engine-level folds: no engine appears on both sides."""
    units = sorted({int(unit) for unit in unit_ids})
    if not 2 <= n_splits <= len(units):
        raise ValueError("n_splits must be between 2 and the number of units")
    shuffled = np.random.default_rng(seed).permutation(units)
    folds = []
    for held_out in np.array_split(shuffled, n_splits):
        held = sorted(int(unit) for unit in held_out)
        held_set = set(held)
        folds.append(([unit for unit in units if unit not in held_set], held))
    return folds
