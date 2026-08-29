"""Non-destructive loader for the NASA C-MAPSS FD001 files."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SETTING_COLUMNS = [f"operational_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ["unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
EXPECTED_COLUMN_COUNT = 26


@dataclass(frozen=True)
class FD001Data:
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series


def _read_trajectory(path: Path) -> pd.DataFrame:
    """Read one whitespace-delimited trajectory file without preprocessing."""
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns in {path.name}, "
            f"found {frame.shape[1]}"
        )
    frame.columns = CMAPSS_COLUMNS
    return frame


def load_fd001(raw_dir: str | Path) -> FD001Data:
    """Load FD001 train, test, and final-cycle test RUL values as published."""
    raw_dir = Path(raw_dir)
    train = _read_trajectory(raw_dir / "train_FD001.txt")
    test = _read_trajectory(raw_dir / "test_FD001.txt")
    rul_path = raw_dir / "RUL_FD001.txt"
    if not rul_path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {rul_path}")
    test_rul_frame = pd.read_csv(rul_path, sep=r"\s+", header=None)
    if test_rul_frame.shape[1] != 1:
        raise ValueError(
            f"Expected 1 column in {rul_path.name}, found {test_rul_frame.shape[1]}"
        )
    test_rul = test_rul_frame.iloc[:, 0].rename("rul_after_last_observed_cycle")
    if len(test_rul) != test["unit_id"].nunique():
        raise ValueError("Test RUL count does not match the number of test units")
    return FD001Data(train=train, test=test, test_rul=test_rul)


def add_linear_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with linear training RUL; never mutate the loaded raw frame."""
    required = {"unit_id", "cycle"}
    if not required.issubset(train.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(train.columns))}")
    result = train.copy()
    failure_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = failure_cycle - result["cycle"]
    return result

