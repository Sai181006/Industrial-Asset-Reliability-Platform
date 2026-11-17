import numpy as np
import pytest

from ml.evaluate import bootstrap_rmse_interval, endpoint_metrics, grouped_folds, nasa_score


def test_nasa_score_is_zero_for_perfect_and_penalizes_late_more():
    true = np.array([50.0])
    assert nasa_score(true, true) == 0.0
    early = nasa_score(true, np.array([40.0]))  # predicted 10 cycles too early
    late = nasa_score(true, np.array([60.0]))  # predicted 10 cycles too late
    assert early == pytest.approx(np.exp(10 / 13) - 1)
    assert late == pytest.approx(np.exp(10 / 10) - 1)
    assert late > early


def test_endpoint_metrics_clips_negatives_and_counts_them():
    true = np.array([5.0, 20.0, 100.0])
    raw = np.array([-3.0, 25.0, 90.0])
    metrics = endpoint_metrics(true, raw)
    assert metrics["negative_raw_predictions"] == 1
    # Clipped predictions are [0, 25, 90] -> errors [-5, 5, -10].
    assert metrics["mae"] == pytest.approx((5 + 5 + 10) / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt((25 + 25 + 100) / 3))
    assert metrics["low_rul_count"] == 2
    assert metrics["low_rul_mae"] == pytest.approx(5.0)


def test_endpoint_metrics_truth_cap_only_clips_evaluation_truth():
    true = np.array([300.0])
    raw = np.array([125.0])
    assert endpoint_metrics(true, raw)["mae"] == pytest.approx(175.0)
    assert endpoint_metrics(true, raw, truth_cap=150)["mae"] == pytest.approx(25.0)


def test_endpoint_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        endpoint_metrics(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        endpoint_metrics(np.array([]), np.array([]))


def test_grouped_folds_are_disjoint_complete_and_deterministic():
    units = list(range(1, 101))
    folds = grouped_folds(units, n_splits=5, seed=0)
    assert folds == grouped_folds(units, n_splits=5, seed=0)
    held_all = []
    for train, held in folds:
        assert not set(train) & set(held)
        assert sorted(train + held) == units
        assert len(held) == 20
        held_all.extend(held)
    assert sorted(held_all) == units


def test_grouped_folds_validates_split_count():
    with pytest.raises(ValueError):
        grouped_folds([1, 2, 3], n_splits=1)
    with pytest.raises(ValueError):
        grouped_folds([1, 2, 3], n_splits=4)


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(1)
    true = rng.uniform(5, 140, size=100)
    raw = true + rng.normal(0, 12, size=100)
    low, high = bootstrap_rmse_interval(true, raw, n_boot=500, seed=0)
    rmse = endpoint_metrics(true, raw)["rmse"]
    assert low < rmse < high
    assert (low, high) == bootstrap_rmse_interval(true, raw, n_boot=500, seed=0)
