"""Leakage-safe sequence preparation for FD001."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.baseline import FEATURE_COLUMNS, TRAIN_UNIT_IDS, VALIDATION_UNIT_IDS
from ml.data.cmapss import add_linear_train_rul


@dataclass(frozen=True)
class SequenceSet:
    features: np.ndarray
    targets: np.ndarray
    unit_ids: np.ndarray
    end_cycles: np.ndarray


@dataclass(frozen=True)
class PreparedSequences:
    train: SequenceSet
    validation: SequenceSet
    scaler: StandardScaler
    window_length: int


def sequence_feasibility(frame: pd.DataFrame, window_length: int) -> dict[str, float | int]:
    """Summarize sequence availability without allocating the 3-D tensor."""
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    with_rul = add_linear_train_rul(frame)
    lengths = with_rul.groupby("unit_id").size()
    counts = (lengths - window_length + 1).clip(lower=0)
    targets = with_rul.groupby("unit_id")["rul"].apply(
        lambda values: values.iloc[window_length - 1 :] if len(values) >= window_length else values.iloc[:0]
    )
    target_values = targets.to_numpy()
    return {
        "window_length": window_length,
        "sequences": int(counts.sum()),
        "engines_too_short": int((lengths < window_length).sum()),
        "target_min": int(target_values.min()),
        "target_median": float(np.median(target_values)),
        "target_max": int(target_values.max()),
        "low_rul_sequences": int((target_values <= 30).sum()),
    }


def _window_units(
    frame: pd.DataFrame, unit_ids: tuple[int, ...], window_length: int
) -> SequenceSet:
    windows: list[np.ndarray] = []
    targets: list[float] = []
    owners: list[int] = []
    end_cycles: list[int] = []
    for unit_id in unit_ids:
        unit = frame[frame["unit_id"] == unit_id].sort_values("cycle")
        values = unit[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        rul = unit["rul"].to_numpy(dtype=np.float32)
        cycles = unit["cycle"].to_numpy(dtype=np.int32)
        for end in range(window_length - 1, len(unit)):
            windows.append(values[end - window_length + 1 : end + 1])
            targets.append(float(rul[end]))
            owners.append(unit_id)
            end_cycles.append(int(cycles[end]))
    if not windows:
        raise ValueError("No sequences can be created for the requested units/window")
    return SequenceSet(
        features=np.stack(windows),
        targets=np.asarray(targets, dtype=np.float32),
        unit_ids=np.asarray(owners, dtype=np.int32),
        end_cycles=np.asarray(end_cycles, dtype=np.int32),
    )


def prepare_development_sequences(
    frame: pd.DataFrame, window_length: int = 30
) -> PreparedSequences:
    """Fit scaling on units 1–80 only, then create isolated train/validation windows."""
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    units = set(frame["unit_id"].unique())
    if units != set(range(1, 101)):
        raise ValueError("Expected complete FD001 training units 1..100")
    with_rul = add_linear_train_rul(frame)
    train_mask = with_rul["unit_id"].isin(TRAIN_UNIT_IDS)
    scaler = StandardScaler().fit(with_rul.loc[train_mask, FEATURE_COLUMNS])
    scaled = with_rul.astype({column: "float64" for column in FEATURE_COLUMNS})
    scaled.loc[:, FEATURE_COLUMNS] = scaler.transform(with_rul[FEATURE_COLUMNS])
    train = _window_units(scaled, TRAIN_UNIT_IDS, window_length)
    validation = _window_units(scaled, VALIDATION_UNIT_IDS, window_length)
    if set(train.unit_ids) & set(validation.unit_ids):
        raise AssertionError("Unit leakage detected after windowing")
    return PreparedSequences(
        train=train,
        validation=validation,
        scaler=scaler,
        window_length=window_length,
    )
