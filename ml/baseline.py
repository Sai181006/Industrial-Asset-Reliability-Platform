"""Leakage-safe current-cycle baselines for FD001 linear RUL."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.cmapss import SENSOR_COLUMNS, SETTING_COLUMNS, add_linear_train_rul


FEATURE_COLUMNS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]
TRAIN_UNIT_IDS = tuple(range(1, 81))
VALIDATION_UNIT_IDS = tuple(range(81, 101))
LOW_RUL_THRESHOLD = 30


@dataclass(frozen=True)
class DevelopmentSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


def make_development_split(frame: pd.DataFrame) -> DevelopmentSplit:
    """Freeze the documented unit-level 1–80 / 81–100 development split."""
    units = set(frame["unit_id"].unique())
    expected = set(TRAIN_UNIT_IDS) | set(VALIDATION_UNIT_IDS)
    if units != expected:
        raise ValueError(f"Expected FD001 units 1..100; found {sorted(units)}")
    with_rul = add_linear_train_rul(frame)
    train = with_rul[with_rul["unit_id"].isin(TRAIN_UNIT_IDS)].copy()
    validation = with_rul[with_rul["unit_id"].isin(VALIDATION_UNIT_IDS)].copy()
    if set(train["unit_id"]) & set(validation["unit_id"]):
        raise AssertionError("Unit leakage detected in development split")
    return DevelopmentSplit(train=train, validation=validation)


def _metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=float)
    low_mask = y_true.to_numpy() <= LOW_RUL_THRESHOLD
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "low_rul_mae": float(mean_absolute_error(y_true[low_mask], prediction[low_mask])),
        "low_rul_rmse": float(mean_squared_error(y_true[low_mask], prediction[low_mask]) ** 0.5),
        "negative_predictions": int((prediction < 0).sum()),
    }


def evaluate_baselines(frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit only on units 1–80 and evaluate once on complete units 81–100."""
    split = make_development_split(frame)
    x_train, y_train = split.train[FEATURE_COLUMNS], split.train["rul"]
    x_validation, y_validation = split.validation[FEATURE_COLUMNS], split.validation["rul"]

    median_value = float(y_train.median())
    median_prediction = np.full(len(y_validation), median_value)

    linear = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )
    linear.fit(x_train, y_train)

    tree = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=1,
        random_state=42,
    )
    tree.fit(x_train, y_train)

    gradient_boosting = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    gradient_boosting.fit(x_train, y_train)

    results = {
        "training_median": _metrics(y_validation, median_prediction),
        "linear_regression": _metrics(y_validation, linear.predict(x_validation)),
        "random_forest": _metrics(y_validation, tree.predict(x_validation)),
        "hist_gradient_boosting": _metrics(
            y_validation, gradient_boosting.predict(x_validation)
        ),
    }
    metadata = {
        "target": "linear_rul=max_cycle_for_unit-cycle",
        "features": FEATURE_COLUMNS,
        "training_units": list(TRAIN_UNIT_IDS),
        "validation_units": list(VALIDATION_UNIT_IDS),
        "training_rows": len(split.train),
        "validation_rows": len(split.validation),
        "low_rul_threshold": LOW_RUL_THRESHOLD,
        "training_median_rul": median_value,
        "scaler_fit_rows": len(x_train),
    }
    return results, metadata
