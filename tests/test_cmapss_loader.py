from pathlib import Path

import pandas as pd
import pytest

from ml.data.cmapss import CMAPSS_COLUMNS, EXPECTED_COLUMN_COUNT, load_fd001


RAW_DIR = Path(__file__).parents[1] / "data" / "raw" / "source" / "CMAPSSData"


@pytest.fixture(scope="module")
def fd001():
    if not RAW_DIR.exists():
        pytest.skip("Official FD001 files are not installed; run scripts/acquire_cmapss.ps1")
    return load_fd001(RAW_DIR)


def test_schema_is_exact(fd001):
    assert fd001.train.shape[1] == EXPECTED_COLUMN_COUNT
    assert fd001.test.shape[1] == EXPECTED_COLUMN_COUNT
    assert fd001.train.columns.tolist() == CMAPSS_COLUMNS
    assert fd001.test.columns.tolist() == CMAPSS_COLUMNS


def test_ids_and_cycles_are_valid(fd001):
    for frame in (fd001.train, fd001.test):
        assert pd.api.types.is_integer_dtype(frame["unit_id"])
        assert pd.api.types.is_integer_dtype(frame["cycle"])
        assert (frame["unit_id"] > 0).all()
        assert (frame["cycle"] > 0).all()
        assert frame.groupby("unit_id")["cycle"].apply(
            lambda cycles: cycles.tolist() == list(range(1, len(cycles) + 1))
        ).all()


def test_loader_is_reproducible(fd001):
    again = load_fd001(RAW_DIR)
    pd.testing.assert_frame_equal(fd001.train, again.train)
    pd.testing.assert_frame_equal(fd001.test, again.test)
    pd.testing.assert_series_equal(fd001.test_rul, again.test_rul)


def test_published_fd001_counts(fd001):
    assert fd001.train["unit_id"].nunique() == 100
    assert fd001.test["unit_id"].nunique() == 100
    assert len(fd001.test_rul) == 100

