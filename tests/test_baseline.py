from pathlib import Path

import numpy as np
import pytest

from ml.baseline import (
    FEATURE_COLUMNS,
    TRAIN_UNIT_IDS,
    VALIDATION_UNIT_IDS,
    evaluate_baselines,
    make_development_split,
)
from ml.data.cmapss import load_fd001


RAW_DIR = Path(__file__).parents[1] / "data" / "raw" / "source" / "CMAPSSData"


@pytest.fixture(scope="module")
def train_frame():
    if not RAW_DIR.exists():
        pytest.skip("Official FD001 files are not installed")
    return load_fd001(RAW_DIR).train


def test_split_keeps_complete_units_disjoint(train_frame):
    split = make_development_split(train_frame)
    assert set(split.train.unit_id.unique()) == set(TRAIN_UNIT_IDS)
    assert set(split.validation.unit_id.unique()) == set(VALIDATION_UNIT_IDS)
    assert not set(split.train.unit_id) & set(split.validation.unit_id)
    assert len(split.train) + len(split.validation) == len(train_frame)


def test_rul_is_constructed_per_unit(train_frame):
    split = make_development_split(train_frame)
    for part in (split.train, split.validation):
        final_rows = part.loc[part.groupby("unit_id")["cycle"].idxmax()]
        assert (final_rows.rul == 0).all()


def test_baseline_evaluation_is_reproducible_and_finite(train_frame):
    first, first_metadata = evaluate_baselines(train_frame)
    second, second_metadata = evaluate_baselines(train_frame)
    assert first_metadata == second_metadata
    assert first.keys() == second.keys()
    assert "hist_gradient_boosting" in first
    for model in first:
        assert first[model] == pytest.approx(second[model])
        assert all(np.isfinite(value) for value in first[model].values())
    assert first_metadata["features"] == FEATURE_COLUMNS
    assert first_metadata["scaler_fit_rows"] == first_metadata["training_rows"]


def test_split_rejects_incomplete_unit_population(train_frame):
    with pytest.raises(ValueError, match="Expected FD001 units"):
        make_development_split(train_frame[train_frame.unit_id != 100])
