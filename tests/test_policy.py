import pytest

from ml.policy import (
    Outcome,
    fixed_interval_outcome,
    perfect_foresight_outcome,
    run_to_failure_outcome,
    select_by_cost,
    summarize_outcomes,
    threshold_outcome,
    trigger_cycle,
)


def test_trigger_cycle_first_crossing_and_never():
    cycles = [30, 31, 32, 33, 34]
    predictions = [90, 40, 55, 20, 10]
    assert trigger_cycle(cycles, predictions, 50) == 31
    assert trigger_cycle(cycles, predictions, 5) is None


def test_trigger_persistence_requires_consecutive_predictions():
    cycles = [30, 31, 32, 33, 34, 35]
    predictions = [10, 90, 10, 10, 90, 10]
    assert trigger_cycle(cycles, predictions, 20, persistence=1) == 30
    assert trigger_cycle(cycles, predictions, 20, persistence=2) == 33
    assert trigger_cycle(cycles, predictions, 20, persistence=3) is None
    with pytest.raises(ValueError):
        trigger_cycle(cycles, predictions, 20, persistence=0)


def test_threshold_outcome_planned_when_lead_fits_before_failure():
    outcome = threshold_outcome(trigger=80, lifetime=100, lead=5)
    assert outcome == Outcome(age=85, failed=False, unused_life=15)


def test_threshold_outcome_fails_when_lead_overruns_failure():
    assert threshold_outcome(trigger=95, lifetime=100, lead=5).failed  # executes at 100 = failure
    assert threshold_outcome(trigger=96, lifetime=100, lead=5) == Outcome(100, True, 0)
    assert threshold_outcome(trigger=94, lifetime=100, lead=5).failed is False
    assert threshold_outcome(trigger=None, lifetime=100, lead=0) == Outcome(100, True, 0)
    with pytest.raises(ValueError):
        threshold_outcome(10, 100, lead=-1)


def test_fixed_interval_outcomes():
    assert fixed_interval_outcome(lifetime=200, interval=150) == Outcome(150, False, 50)
    assert fixed_interval_outcome(lifetime=150, interval=150) == Outcome(150, True, 0)
    assert fixed_interval_outcome(lifetime=120, interval=150) == Outcome(120, True, 0)
    with pytest.raises(ValueError):
        fixed_interval_outcome(100, 0)


def test_reference_policies():
    assert run_to_failure_outcome(100) == Outcome(100, True, 0)
    assert perfect_foresight_outcome(100) == Outcome(99, False, 1)


def test_cost_rate_is_per_operating_cycle_not_per_engine():
    early = [Outcome(50, False, 50)] * 2  # replaced early, twice as often
    late = [Outcome(100, False, 0)] * 2
    early_rate = summarize_outcomes(early, 1.0, 10.0)["cost_per_1000_cycles"]
    late_rate = summarize_outcomes(late, 1.0, 10.0)["cost_per_1000_cycles"]
    assert early_rate == pytest.approx(20.0)
    assert late_rate == pytest.approx(10.0)


def test_summarize_outcomes_counts_and_rejects_bad_input():
    outcomes = [Outcome(90, False, 10), Outcome(100, True, 0), Outcome(80, False, 20)]
    summary = summarize_outcomes(outcomes, planned_cost=1.0, failure_cost=10.0)
    assert summary["failures"] == 1 and summary["planned"] == 2
    assert summary["mean_unused_life"] == pytest.approx(15.0)
    assert summary["total_cost"] == pytest.approx(12.0)
    assert summary["cost_per_1000_cycles"] == pytest.approx(1000 * 12 / 270)
    with pytest.raises(ValueError):
        summarize_outcomes([])
    with pytest.raises(ValueError):
        summarize_outcomes(outcomes, planned_cost=0)


def test_select_by_cost_picks_cheapest_and_breaks_ties_low():
    lifetimes = [100, 100, 100]
    by_interval = {t: [fixed_interval_outcome(life, t) for life in lifetimes] for t in (60, 90, 99, 100, 120)}
    # 99 is the latest interval that never fails; 100 and 120 fail every time.
    assert select_by_cost(by_interval, planned_cost=1.0, failure_cost=10.0) == 99
    same = {5: [Outcome(100, False, 0)], 7: [Outcome(100, False, 0)]}
    assert select_by_cost(same, 1.0, 10.0) == 5
    with pytest.raises(ValueError):
        select_by_cost({}, 1.0, 10.0)
