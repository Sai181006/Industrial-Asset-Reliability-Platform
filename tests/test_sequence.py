from pathlib import Path

import numpy as np
import pytest

from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import load_fd001
from ml.sequence import prepare_development_sequences, sequence_feasibility


RAW_DIR = Path(__file__).parents[1] / "data" / "raw" / "source" / "CMAPSSData"


@pytest.fixture(scope="module")
def train_frame():
    if not RAW_DIR.exists():
        pytest.skip("Official FD001 files are not installed")
    return load_fd001(RAW_DIR).train


def test_candidate_feasibility_matches_observed_counts(train_frame):
    assert sequence_feasibility(train_frame, 20)["sequences"] == 18_731
    assert sequence_feasibility(train_frame, 30)["sequences"] == 17_731
    assert sequence_feasibility(train_frame, 50)["sequences"] == 15_731


def test_window_generation_is_leakage_safe_and_aligned(train_frame):
    prepared = prepare_development_sequences(train_frame, window_length=30)
    assert prepared.train.features.shape[1:] == (30, len(FEATURE_COLUMNS))
    assert prepared.validation.features.shape[1:] == (30, len(FEATURE_COLUMNS))
    assert not set(prepared.train.unit_ids) & set(prepared.validation.unit_ids)
    assert len(prepared.train.features) == len(prepared.train.targets)
    assert len(prepared.validation.features) == len(prepared.validation.targets)
    assert (prepared.train.end_cycles >= 30).all()
    assert (prepared.validation.end_cycles >= 30).all()


def test_scaler_is_fit_on_training_rows_only(train_frame):
    prepared = prepare_development_sequences(train_frame, window_length=30)
    training_rows = train_frame[train_frame.unit_id <= 80][FEATURE_COLUMNS]
    np.testing.assert_allclose(prepared.scaler.mean_, training_rows.mean().to_numpy())


def test_invalid_window_rejected(train_frame):
    with pytest.raises(ValueError, match="positive"):
        sequence_feasibility(train_frame, 0)

