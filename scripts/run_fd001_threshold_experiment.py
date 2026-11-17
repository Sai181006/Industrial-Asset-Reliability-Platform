"""Compare maintenance thresholds on out-of-fold FD001 predictions.

Candidates are fixed before running, so scoring them on all 100 engines involves
no selection. Headline question: today's work-order threshold (10 cycles) against
15, which is roughly lead time + 10 for a 5-cycle lead. A regret table then asks
which fixed threshold is the safest default when the real lead time is unknown.
Costs and lead times are illustrative assumptions on simulated data.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.cmapss import load_fd001
from ml.evaluate import grouped_folds
from ml.features import add_rul_targets, select_informative_sensors
from ml.policy import summarize_outcomes, threshold_outcome, trigger_cycle
from scripts.run_fd001_evaluation import CV_SEED, cached
from scripts.run_fd001_policy_backtest import oof_predictions

CANDIDATES = (8, 10, 12, 15, 20, 25, 30)
DEFAULT_THRESHOLD, PROPOSED_THRESHOLD = 10, 15
RATIOS = (5, 10, 20, 50)
LEADS = (0, 2, 5, 8, 10, 15)
PERSISTENCE = (1, 3)
GRID = tuple(range(0, 101))
LEAD_RANGES = {"leads 0-5": (0, 2, 5), "leads 0-10": (0, 2, 5, 8, 10), "leads 0-15": LEADS}


def paired_bootstrap(a_age, a_cost, b_age, b_cost, n_boot=2000, seed=0):
    """Percent cost saved by policy A relative to B, resampling engines."""
    draws = np.random.default_rng(seed).integers(0, len(a_age), size=(n_boot, len(a_age)))
    saving = 100.0 * (1.0 - (a_cost[draws].sum(1) / a_age[draws].sum(1)) / (b_cost[draws].sum(1) / b_age[draws].sum(1)))
    return saving


def main() -> None:
    data = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData")
    frame = add_rul_targets(data.train)
    sensors = select_informative_sensors(data.train)
    units = list(range(1, 101))
    life = {int(u): int(n) for u, n in frame.groupby("unit_id")["cycle"].max().items()}
    raw = cached("oof_predictions", lambda: oof_predictions(frame, sensors, units))
    cycles = {int(u): v["cycles"] for u, v in raw.items()}
    rul = {int(u): v["rul"] for u, v in raw.items()}

    @lru_cache(maxsize=None)
    def trigger(unit, threshold, persistence):
        return trigger_cycle(cycles[unit], rul[unit], threshold, persistence)

    def outcomes(threshold, lead, persistence):
        return [threshold_outcome(trigger(u, threshold, persistence), life[u], lead) for u in units]

    def engine_arrays(outs, ratio):
        age = np.array([o.age for o in outs], dtype=float)
        cost = np.array([float(ratio) if o.failed else 1.0 for o in outs])
        return age, cost

    rows, headline, regret_rows = [], [], []
    curves = {}
    for persistence in PERSISTENCE:
        for ratio in RATIOS:
            for lead in LEADS:
                grid_cost = {t: summarize_outcomes(outcomes(t, lead, persistence), 1.0, float(ratio))["cost_per_1000_cycles"] for t in GRID}
                best_t = min(GRID, key=lambda t: (grid_cost[t], t))
                curves[(persistence, ratio, lead)] = (grid_cost, best_t)
                for t in CANDIDATES:
                    s = summarize_outcomes(outcomes(t, lead, persistence), 1.0, float(ratio))
                    rows.append({
                        "persistence": persistence, "ratio": ratio, "lead": lead, "threshold": t,
                        "cost_per_1000_cycles": s["cost_per_1000_cycles"], "failures_per_100": s["failures_per_100"],
                        "mean_unused_life": s["mean_unused_life"], "best_grid_threshold": best_t,
                        "regret_abs": grid_cost[t] - grid_cost[best_t], "regret_pct": 100.0 * (grid_cost[t] / grid_cost[best_t] - 1.0),
                    })
                d_out, p_out = outcomes(DEFAULT_THRESHOLD, lead, persistence), outcomes(PROPOSED_THRESHOLD, lead, persistence)
                d_age, d_cost = engine_arrays(d_out, ratio)
                p_age, p_cost = engine_arrays(p_out, ratio)
                saving = paired_bootstrap(p_age, p_cost, d_age, d_cost)
                lo, hi = float(np.quantile(saving, 0.025)), float(np.quantile(saving, 0.975))
                cost_d, cost_p = grid_cost[DEFAULT_THRESHOLD], grid_cost[PROPOSED_THRESHOLD]
                winner = "15" if lo > 0 else ("10" if hi < 0 else "tie")
                headline.append({
                    "persistence": persistence, "ratio": ratio, "lead": lead,
                    "cost_10": cost_d, "cost_15": cost_p,
                    "failures_10": summarize_outcomes(d_out, 1.0, float(ratio))["failures_per_100"],
                    "failures_15": summarize_outcomes(p_out, 1.0, float(ratio))["failures_per_100"],
                    "saving_15_vs_10_pct": 100.0 * (1.0 - cost_p / cost_d), "ci95": [lo, hi], "winner": winner,
                })

    # Safest fixed default when the lead time is unknown: smallest worst-case relative regret.
    regret_summary = {}
    for persistence in PERSISTENCE:
        for name, leads in LEAD_RANGES.items():
            per_t = {}
            for t in CANDIDATES:
                sel = [r["regret_pct"] for r in rows if r["persistence"] == persistence and r["threshold"] == t and r["lead"] in leads]
                per_t[t] = {"max_regret_pct": max(sel), "mean_regret_pct": float(np.mean(sel))}
            minimax = min(CANDIDATES, key=lambda t: (per_t[t]["max_regret_pct"], t))
            regret_summary[f"k{persistence} {name}"] = {"by_threshold": per_t, "minimax_threshold": minimax}

    wins = {w: sum(1 for h in headline if h["winner"] == w and h["persistence"] == 1) for w in ("10", "15", "tie")}
    output = {
        "design": {
            "candidates": CANDIDATES, "default": DEFAULT_THRESHOLD, "proposed": PROPOSED_THRESHOLD,
            "ratios": RATIOS, "leads": LEADS, "persistence": PERSISTENCE,
            "scoring": "all 100 engines on out-of-fold predictions; thresholds fixed in advance; cost per 1000 operating cycles",
            "regret": "cost minus the best threshold on a 0-100 grid for the same scenario (in-sample reference)",
        },
        "headline_10_vs_15": headline, "headline_winner_counts_k1": wins,
        "regret_summary": regret_summary, "candidate_rows": rows,
    }
    (ROOT / "docs" / "threshold_experiment.json").write_text(json.dumps(output, indent=2, default=list) + "\n", encoding="utf-8")
    print("winner counts (single trigger, 24 scenarios):", wins)
    for key, value in regret_summary.items():
        print(f"{key}: minimax threshold = {value['minimax_threshold']}")


if __name__ == "__main__":
    main()
