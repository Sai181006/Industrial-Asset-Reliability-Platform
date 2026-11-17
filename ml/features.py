"""Cleaned features, capped RUL target, and window summaries for FD001.

These helpers are additive: ``ml.baseline`` and ``ml.sequence`` keep the
24-feature, uncapped contract of the promoted GRU artifact. Everything here
supports the shared evaluation protocol in ``ml.evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ml.data.cmapss import SENSOR_COLUMNS, add_linear_train_rul


RUL_CAP = 125  # Pre-declared convention for piecewise-linear RUL; not tuned on test data.
WINDOW_LENGTH = 30
REPRESENTATIONS = ("current", "window")


def select_informative_sensors(train: pd.DataFrame) -> list[str]:
    """Return sensors with more than two distinct training values.

    Operational settings are excluded entirely: FD001 has a single operating
    condition, so their variation is measurement noise. Constant and two-valued
    sensors carry no usable degradation signal.
    """
    missing = [column for column in SENSOR_COLUMNS if column not in train.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    return [column for column in SENSOR_COLUMNS if train[column].nunique() > 2]


def add_rul_targets(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Return a copy with linear ``rul`` and piecewise-linear ``rul_capped``."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    result = add_linear_train_rul(train)
    result["rul_capped"] = result["rul"].clip(upper=cap)
    return result


def summarize_window(window: np.ndarray) -> np.ndarray:
    """Collapse a (length, features) window into [last, mean, slope] features.

    Slope is the least-squares trend per cycle, so a rising sensor is positive.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[0] < 2:
        raise ValueError("window must be 2-D with at least two cycles")
    offsets = np.arange(window.shape[0], dtype=np.float64) - (window.shape[0] - 1) / 2
    slope = (offsets[:, None] * (window - window.mean(axis=0))).sum(axis=0) / (offsets**2).sum()
    return np.concatenate([window[-1], window.mean(axis=0), slope])


def window_feature_names(columns: Sequence[str]) -> list[str]:
    return [f"{column}__{kind}" for kind in ("last", "mean", "slope") for column in columns]


@dataclass(frozen=True)
class SampleSet:
    features: np.ndarray
    rul: np.ndarray  # uncapped true RUL; NaN when the frame has no RUL column
    rul_capped: np.ndarray  # NaN when the frame has no RUL column
    unit_ids: np.ndarray
    end_cycles: np.ndarray


Endpoints = str | Mapping[int, Sequence[int]]


def sample_truncation_points(
    lengths: Mapping[int, int],
    per_unit: int = 10,
    low: float = 0.2,
    high: float = 0.97,
    window_length: int = WINDOW_LENGTH,
    seed: int = 0,
) -> dict[int, list[int]]:
    """Draw seeded cut points that mimic how the official test set was truncated.

    Each value is a zero-based row index of the last observed cycle, so a unit
    cut after ``k`` observed cycles maps to index ``k - 1``.
    """
    if not 0 < low <= high <= 1:
        raise ValueError("require 0 < low <= high <= 1")
    rng = np.random.default_rng(seed)
    points: dict[int, list[int]] = {}
    for unit_id in sorted(lengths):
        length = int(lengths[unit_id])
        first = max(window_length, int(np.ceil(low * length)))
        last = max(first, int(np.floor(high * length)))
        if last > length:
            raise ValueError(f"Unit {unit_id} has fewer than {window_length} cycles")
        cuts = rng.integers(first, last + 1, size=per_unit)
        points[int(unit_id)] = sorted(int(cut) - 1 for cut in cuts)
    return points


def build_samples(
    frame: pd.DataFrame,
    unit_ids: Sequence[int],
    columns: Sequence[str],
    representation: str,
    endpoints: Endpoints = "all",
    window_length: int = WINDOW_LENGTH,
) -> SampleSet:
    """Build one sample per endpoint with identical rows for every representation.

    ``endpoints`` is ``"all"`` (every row with a full window behind it),
    ``"last"`` (the final observed row of each unit, as in the official test
    protocol), or a mapping of unit id to zero-based row indices.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    has_rul = "rul" in frame.columns
    features, rul, capped, owners, cycles = [], [], [], [], []
    for unit_id in unit_ids:
        unit = frame[frame["unit_id"] == unit_id].sort_values("cycle")
        if len(unit) < window_length:
            raise ValueError(f"Unit {unit_id} has fewer than {window_length} cycles")
        values = unit[list(columns)].to_numpy(dtype=np.float64)
        if endpoints == "all":
            ends = range(window_length - 1, len(unit))
        elif endpoints == "last":
            ends = [len(unit) - 1]
        elif isinstance(endpoints, Mapping):
            ends = endpoints[int(unit_id)]
        else:
            raise ValueError("endpoints must be 'all', 'last', or a mapping")
        for end in ends:
            if end < window_length - 1 or end >= len(unit):
                raise ValueError(f"Endpoint {end} is invalid for unit {unit_id}")
            if representation == "current":
                features.append(values[end])
            else:
                features.append(summarize_window(values[end - window_length + 1 : end + 1]))
            owners.append(int(unit_id))
            cycles.append(int(unit["cycle"].iloc[end]))
            rul.append(float(unit["rul"].iloc[end]) if has_rul else np.nan)
            capped.append(
                float(unit["rul_capped"].iloc[end]) if "rul_capped" in unit else np.nan
            )
    if not features:
        raise ValueError("No samples were created")
    return SampleSet(
        features=np.asarray(features, dtype=np.float64),
        rul=np.asarray(rul, dtype=np.float64),
        rul_capped=np.asarray(capped, dtype=np.float64),
        unit_ids=np.asarray(owners, dtype=np.int32),
        end_cycles=np.asarray(cycles, dtype=np.int32),
    )


def build_sequences(
    frame: pd.DataFrame,
    unit_ids: Sequence[int],
    columns: Sequence[str],
    endpoints: Endpoints = "all",
    window_length: int = WINDOW_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw (samples, window, features) sequences plus RUL, unit ids and end cycles.

    Uses the same endpoint rules as ``build_samples``. RUL is NaN when the frame
    has no ``rul`` column (the official test frame).
    """
    has_rul = "rul" in frame.columns
    windows, rul, owners, cycles = [], [], [], []
    for unit_id in unit_ids:
        unit = frame[frame["unit_id"] == unit_id].sort_values("cycle")
        if len(unit) < window_length:
            raise ValueError(f"Unit {unit_id} has fewer than {window_length} cycles")
        values = unit[list(columns)].to_numpy(dtype=np.float32)
        if endpoints == "all":
            ends = range(window_length - 1, len(unit))
        elif endpoints == "last":
            ends = [len(unit) - 1]
        elif isinstance(endpoints, Mapping):
            ends = endpoints[int(unit_id)]
        else:
            raise ValueError("endpoints must be 'all', 'last', or a mapping")
        for end in ends:
            if end < window_length - 1 or end >= len(unit):
                raise ValueError(f"Endpoint {end} is invalid for unit {unit_id}")
            windows.append(values[end - window_length + 1 : end + 1])
            rul.append(float(unit["rul"].iloc[end]) if has_rul else np.nan)
            owners.append(int(unit_id))
            cycles.append(int(unit["cycle"].iloc[end]))
    if not windows:
        raise ValueError("No sequences were created")
    return (
        np.stack(windows),
        np.asarray(rul, dtype=np.float64),
        np.asarray(owners, dtype=np.int32),
        np.asarray(cycles, dtype=np.int32),
    )
