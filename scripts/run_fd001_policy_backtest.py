"""Maintenance-policy back-test on out-of-fold FD001 predictions.

Predictions come from the CV-selected model (14 sensors, window features, RUL
capped at 125, histogram gradient boosting) under grouped 5-fold CV, so every
engine is scored by a model that never saw it. Policies are replayed cycle by
cycle on all 100 training engines and compared by cost per operating cycle.

Primary scenario, declared before running: failure costs 10x a planned job,
work orders take 5 cycles to execute, one prediction below the threshold fires.
Thresholds and intervals are chosen on the *other* folds' engines, then applied
to the held-out fold. Cost and lead-time inputs are illustrative assumptions.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mplcfg"))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.cmapss import load_fd001
from ml.evaluate import grouped_folds
from ml.features import RUL_CAP, add_rul_targets, build_samples, select_informative_sensors
from ml.policy import (
    fixed_interval_outcome,
    perfect_foresight_outcome,
    run_to_failure_outcome,
    select_by_cost,
    summarize_outcomes,
    threshold_outcome,
    trigger_cycle,
)
from scripts.run_fd001_evaluation import CV_SEED, cached, make_model

RATIOS = (5, 10, 20, 50)  # failure cost relative to a planned job (planned = 1)
LEADS = (0, 5, 10)  # cycles between the trigger and executed maintenance
PERSISTENCE = (1, 3)  # consecutive sub-threshold predictions required
THRESHOLDS = tuple(range(0, 101))
INTERVALS = tuple(range(30, 363))
PRIMARY = {"ratio": 10, "lead": 5, "persistence": 1}
BACKEND_DEFAULT_THRESHOLD = 10  # RUL_CRITICAL_THRESHOLD: where work orders are created today


def oof_predictions(frame, sensors, units):
    predictions = {}
    for train_units, held_units in grouped_folds(units, n_splits=5, seed=CV_SEED):
        train = build_samples(frame, train_units, sensors, "window")
        model = make_model("hgb", seed=CV_SEED).fit(train.features, np.minimum(train.rul, RUL_CAP))
        held = build_samples(frame, held_units, sensors, "window")
        raw = np.maximum(model.predict(held.features), 0.0)
        for unit in held_units:
            mask = held.unit_ids == unit
            predictions[str(unit)] = {"cycles": held.end_cycles[mask].tolist(), "rul": raw[mask].tolist()}
    return predictions


def cross_fold(folds, outcome_fn, params, planned_cost, failure_cost):
    """Pick the parameter on the other folds' engines, apply it to the held-out fold."""
    pooled, chosen = {}, []
    for other_units, held_units in folds:
        best = select_by_cost(
            {p: [outcome_fn(u, p) for u in other_units] for p in params}, planned_cost, failure_cost
        )
        chosen.append(best)
        pooled.update({u: outcome_fn(u, best) for u in held_units})
    return pooled, chosen


def paired_bootstrap_savings(model, baseline, units, planned_cost, failure_cost, n_boot=2000, seed=0):
    """95% interval for percent cost saved versus the baseline, resampling engines."""
    def arrays(outcomes):
        age = np.array([outcomes[u].age for u in units], dtype=float)
        cost = np.array([failure_cost if outcomes[u].failed else planned_cost for u in units], dtype=float)
        return age, cost
    (m_age, m_cost), (b_age, b_cost) = arrays(model), arrays(baseline)
    draws = np.random.default_rng(seed).integers(0, len(units), size=(n_boot, len(units)))
    model_rate = m_cost[draws].sum(axis=1) / m_age[draws].sum(axis=1)
    base_rate = b_cost[draws].sum(axis=1) / b_age[draws].sum(axis=1)
    savings = 100.0 * (1.0 - model_rate / base_rate)
    return [float(np.quantile(savings, 0.025)), float(np.quantile(savings, 0.975))]


def main() -> None:
    started = time.time()
    data = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData")
    frame = add_rul_targets(data.train)
    sensors = select_informative_sensors(data.train)
    units = list(range(1, 101))
    life = {int(u): int(n) for u, n in frame.groupby("unit_id")["cycle"].max().items()}
    folds = grouped_folds(units, n_splits=5, seed=CV_SEED)

    raw = cached("oof_predictions", lambda: oof_predictions(frame, sensors, units))
    cycles = {int(u): v["cycles"] for u, v in raw.items()}
    rul = {int(u): v["rul"] for u, v in raw.items()}
    assert sorted(cycles) == units, "missing out-of-fold predictions"

    @lru_cache(maxsize=None)
    def trigger(unit, threshold, persistence):
        return trigger_cycle(cycles[unit], rul[unit], threshold, persistence)

    def model_outcome(lead, persistence):
        return lambda u, tau: threshold_outcome(trigger(u, tau, persistence), life[u], lead)

    fixed_outcome = lambda u, interval: fixed_interval_outcome(life[u], interval)
    summarize = lambda outcomes, c_f: summarize_outcomes(list(outcomes), 1.0, float(c_f))

    scenarios, fixed_cache = {}, {}
    for ratio in RATIOS:
        pooled, chosen = cross_fold(folds, fixed_outcome, INTERVALS, 1.0, float(ratio))
        best = select_by_cost({t: [fixed_outcome(u, t) for u in units] for t in INTERVALS}, 1.0, float(ratio))
        fixed_cache[ratio] = (pooled, chosen, best)
    for ratio in RATIOS:
        fixed_pooled, fixed_chosen, fixed_best = fixed_cache[ratio]
        for lead in LEADS:
            for persistence in PERSISTENCE:
                outcome_fn = model_outcome(lead, persistence)
                pooled, chosen = cross_fold(folds, outcome_fn, THRESHOLDS, 1.0, float(ratio))
                best_tau = select_by_cost({t: [outcome_fn(u, t) for u in units] for t in THRESHOLDS}, 1.0, float(ratio))
                policies = {
                    "run_to_failure": {"summary": summarize([run_to_failure_outcome(life[u]) for u in units], ratio)},
                    "fixed_cross_fold": {"summary": summarize(fixed_pooled.values(), ratio), "chosen": fixed_chosen},
                    "fixed_in_sample": {"summary": summarize([fixed_outcome(u, fixed_best) for u in units], ratio), "chosen": fixed_best},
                    "model_default_tau10": {"summary": summarize([outcome_fn(u, BACKEND_DEFAULT_THRESHOLD) for u in units], ratio), "chosen": BACKEND_DEFAULT_THRESHOLD},
                    "model_cross_fold": {"summary": summarize(pooled.values(), ratio), "chosen": chosen},
                    "model_in_sample": {"summary": summarize([outcome_fn(u, best_tau) for u in units], ratio), "chosen": best_tau},
                    "perfect_foresight": {"summary": summarize([perfect_foresight_outcome(life[u]) for u in units], ratio)},
                }
                saving = 100.0 * (1.0 - policies["model_cross_fold"]["summary"]["cost_per_1000_cycles"] / policies["fixed_cross_fold"]["summary"]["cost_per_1000_cycles"])
                scenarios[f"ratio{ratio}_lead{lead}_k{persistence}"] = {
                    "ratio": ratio, "lead": lead, "persistence": persistence,
                    "model_saving_vs_fixed_pct": saving, "policies": policies,
                    "_pooled": (pooled, fixed_pooled),
                }
    key = f"ratio{PRIMARY['ratio']}_lead{PRIMARY['lead']}_k{PRIMARY['persistence']}"
    pooled, fixed_pooled = scenarios[key]["_pooled"]
    primary_ci = paired_bootstrap_savings(pooled, fixed_pooled, units, 1.0, float(PRIMARY["ratio"]))
    outcome_fn = model_outcome(PRIMARY["lead"], PRIMARY["persistence"])
    curve = [
        {"threshold": t, **{k: summarize([outcome_fn(u, t) for u in units], PRIMARY["ratio"])[k] for k in ("cost_per_1000_cycles", "failures_per_100", "mean_unused_life")}}
        for t in THRESHOLDS
    ]
    for scenario in scenarios.values():
        scenario.pop("_pooled")

    output = {
        "assumptions": {
            "model": "14 sensors, window features, RUL capped at 125, histogram gradient boosting, grouped 5-fold out-of-fold",
            "planned_cost": 1.0, "failure_cost_ratios": list(RATIOS), "lead_cycles": list(LEADS),
            "persistence": list(PERSISTENCE), "primary": PRIMARY, "cost_metric": "cost per 1000 operating cycles (renewal-reward)",
            "selection": "threshold/interval chosen on the other folds' engines, applied to the held-out fold",
            "engines": len(units), "backend_default_critical_threshold": BACKEND_DEFAULT_THRESHOLD,
        },
        "primary_key": key, "primary_saving_vs_fixed_ci95": primary_ci,
        "scenarios": scenarios, "cost_curve_primary": curve,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (ROOT / "docs" / "policy_backtest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = scenarios[key]["policies"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([c["threshold"] for c in curve], [c["cost_per_1000_cycles"] for c in curve], color="#1f77b4", label="Model-driven, by threshold")
    ax.axhline(primary["fixed_in_sample"]["summary"]["cost_per_1000_cycles"], color="#d62728", linestyle="--", label="Best fixed interval (tuned on same fleet)")
    ax.axhline(primary["run_to_failure"]["summary"]["cost_per_1000_cycles"], color="#7f7f7f", linestyle=":", label="Run to failure")
    ax.axhline(primary["perfect_foresight"]["summary"]["cost_per_1000_cycles"], color="#2ca02c", linestyle=":", label="Perfect foresight")
    ax.axvline(BACKEND_DEFAULT_THRESHOLD, color="black", alpha=0.35)
    ax.annotate("backend default (10)", (BACKEND_DEFAULT_THRESHOLD, ax.get_ylim()[1] * 0.9), rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Maintenance threshold on predicted RUL (cycles)")
    ax.set_ylabel("Cost per 1000 operating cycles")
    ax.set_title(f"Policy back-test, failure = {PRIMARY['ratio']}x planned job, {PRIMARY['lead']}-cycle lead time")
    ax.set_ylim(0, min(ax.get_ylim()[1], primary["run_to_failure"]["summary"]["cost_per_1000_cycles"] * 1.1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "policy_backtest.png", dpi=150)
    plt.close(fig)
    print(f"done in {time.time() - started:.0f}s; primary saving vs fixed = {scenarios[key]['model_saving_vs_fixed_pct']:.1f}% (CI {primary_ci[0]:.1f} to {primary_ci[1]:.1f})", flush=True)


if __name__ == "__main__":
    main()
