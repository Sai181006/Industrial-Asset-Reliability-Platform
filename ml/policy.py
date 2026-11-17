"""Maintenance-policy simulation on run-to-failure RUL data.

Each engine's last observed cycle is its failure. A policy decides when planned
maintenance happens. Planned maintenance and failure both renew the asset, so
policies are compared by renewal-reward cost rate: total cost divided by total
operating cycles. Per-engine cost would make replacing early look free.

The cost and lead-time inputs are illustrative assumptions, not field data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Outcome:
    age: int  # operating cycles consumed before the asset is renewed
    failed: bool
    unused_life: int  # life discarded by planned maintenance; 0 when the asset failed


def trigger_cycle(
    cycles: Sequence[int], predicted_rul: Sequence[float], threshold: float, persistence: int = 1
) -> int | None:
    """First cycle where ``persistence`` consecutive predictions are <= threshold."""
    if persistence < 1:
        raise ValueError("persistence must be at least 1")
    run = 0
    for cycle, value in zip(cycles, predicted_rul):
        run = run + 1 if value <= threshold else 0
        if run >= persistence:
            return int(cycle)
    return None


def threshold_outcome(trigger: int | None, lifetime: int, lead: int) -> Outcome:
    """Maintenance executes ``lead`` cycles after the trigger; too late means failure."""
    if lead < 0:
        raise ValueError("lead must be non-negative")
    if trigger is not None and trigger + lead < lifetime:
        age = trigger + lead
        return Outcome(age=age, failed=False, unused_life=lifetime - age)
    return Outcome(age=lifetime, failed=True, unused_life=0)


def fixed_interval_outcome(lifetime: int, interval: int) -> Outcome:
    if interval < 1:
        raise ValueError("interval must be positive")
    if interval < lifetime:
        return Outcome(age=interval, failed=False, unused_life=lifetime - interval)
    return Outcome(age=lifetime, failed=True, unused_life=0)


def run_to_failure_outcome(lifetime: int) -> Outcome:
    return Outcome(age=lifetime, failed=True, unused_life=0)


def perfect_foresight_outcome(lifetime: int) -> Outcome:
    """Lower bound: maintain on the last cycle before failure."""
    return Outcome(age=lifetime - 1, failed=False, unused_life=1)


def summarize_outcomes(
    outcomes: Sequence[Outcome], planned_cost: float = 1.0, failure_cost: float = 10.0
) -> dict[str, float | int | None]:
    if not outcomes:
        raise ValueError("no outcomes to summarize")
    if planned_cost <= 0 or failure_cost <= 0:
        raise ValueError("costs must be positive")
    failures = sum(outcome.failed for outcome in outcomes)
    planned = [outcome for outcome in outcomes if not outcome.failed]
    total_age = sum(outcome.age for outcome in outcomes)
    total_cost = planned_cost * len(planned) + failure_cost * failures
    return {
        "engines": len(outcomes),
        "failures": int(failures),
        "failures_per_100": 100.0 * failures / len(outcomes),
        "planned": len(planned),
        "mean_unused_life": float(np.mean([o.unused_life for o in planned])) if planned else None,
        "total_cost": float(total_cost),
        "total_operating_cycles": int(total_age),
        "cost_per_1000_cycles": 1000.0 * total_cost / total_age,
    }


def select_by_cost(
    outcomes_by_param: Mapping[float, Sequence[Outcome]], planned_cost: float, failure_cost: float
) -> float:
    """Return the parameter with the lowest cost rate; ties go to the smallest value."""
    if not outcomes_by_param:
        raise ValueError("no candidate parameters")
    best_param, best_rate = None, float("inf")
    for param in sorted(outcomes_by_param):
        rate = summarize_outcomes(outcomes_by_param[param], planned_cost, failure_cost)[
            "cost_per_1000_cycles"
        ]
        if rate < best_rate - 1e-12:
            best_param, best_rate = param, rate
    return best_param
