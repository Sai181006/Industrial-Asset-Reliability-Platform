"""Extra threshold experiments on out-of-fold FD001 predictions (back-test framework only).

E1 margin rule tau = lead + m; E2 persistence k x lead; E4 model robustness
(HGB vs RF vs GRU); E5 uncertainty-aware trigger (RF mean - z * tree std).
Scenario: failure costs 10x a planned job unless stated. Backend defaults are not changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.cmapss import load_fd001
from ml.evaluate import grouped_folds
from ml.features import RUL_CAP, add_rul_targets, build_samples, select_informative_sensors
from ml.policy import select_by_cost, summarize_outcomes, threshold_outcome, trigger_cycle
from scripts.run_fd001_evaluation import CV_SEED, cached, make_model
from scripts.run_fd001_policy_backtest import cross_fold

UNITS = list(range(1, 101))
RATIO, LEAD = 10.0, 5
TAUS = tuple(range(0, 101))


def rf_oof(frame, sensors):
    """Per-endpoint OOF mean and across-tree std from the project RF on capped window features."""
    out = {}
    for train_units, held_units in grouped_folds(UNITS, n_splits=5, seed=CV_SEED):
        train = build_samples(frame, train_units, sensors, "window")
        model = make_model("rf", seed=CV_SEED).fit(train.features, np.minimum(train.rul, RUL_CAP))
        held = build_samples(frame, held_units, sensors, "window")
        trees = np.stack([t.predict(held.features) for t in model.estimators_])
        for u in held_units:
            m = held.unit_ids == u
            out[str(u)] = {"cycles": held.end_cycles[m].tolist(), "mean": np.maximum(trees[:, m].mean(0), 0).tolist(), "std": trees[:, m].std(0).tolist()}
    return out


def main() -> None:
    data = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData")
    frame, sensors = add_rul_targets(data.train), select_informative_sensors(data.train)
    life = {int(u): int(n) for u, n in frame.groupby("unit_id")["cycle"].max().items()}
    folds = grouped_folds(UNITS, n_splits=5, seed=CV_SEED)
    rf = cached("oof_rf_std", lambda: rf_oof(frame, sensors))
    sources = {}
    for name, key in (("hgb", "oof_predictions"), ("gru", "oof_gru")):
        raw = cached(key, lambda: None)
        sources[name] = {int(u): (v["cycles"], v["rul"]) for u, v in raw.items()}
    sources["rf"] = {int(u): (v["cycles"], v["mean"]) for u, v in rf.items()}
    summarize = lambda outs, ratio=RATIO: summarize_outcomes(list(outs), 1.0, float(ratio))

    def outcome_fn(series, lead, k):
        cache = {}
        def fn(u, tau):
            if (u, tau) not in cache:
                cache[(u, tau)] = threshold_outcome(trigger_cycle(series[u][0], series[u][1], tau, k), life[u], lead)
            return cache[(u, tau)]
        return fn

    def evaluate(fn, params, ratio=RATIO):
        pooled, chosen = cross_fold(folds, fn, params, 1.0, float(ratio))
        curve = {p: summarize([fn(u, p) for u in UNITS], ratio) for p in params}
        best = min(params, key=lambda p: (curve[p]["cost_per_1000_cycles"], p))
        best_cost = curve[best]["cost_per_1000_cycles"]
        near = [p for p in params if curve[p]["cost_per_1000_cycles"] <= 1.05 * best_cost]
        return {"cross_fold": summarize(pooled.values(), ratio), "chosen_per_fold": chosen, "best_in_sample_tau": best,
                "best_in_sample": curve[best], "tau_within_5pct_of_best": [min(near), max(near)], "curve": curve}

    out = {"scenario": {"failure_cost_ratio": RATIO, "lead": LEAD, "note": "OOF predictions; tau/m chosen cross-fold; costs illustrative"}}

    # E1: margin rule, HGB. tau = lead + m, for several leads, ratio 10
    e1 = {}
    for lead in (0, 5, 10):
        fn = outcome_fn(sources["hgb"], lead, 1)
        margins = list(range(0, 31))
        curve = {m: summarize([fn(u, lead + m) for u in UNITS]) for m in margins}
        best_m = min(margins, key=lambda m: (curve[m]["cost_per_1000_cycles"], m))
        e1[f"lead{lead}"] = {"best_margin": best_m, "best": curve[best_m], "margin0": curve[0],
                             "tau10_default": summarize([fn(u, 10) for u in UNITS]),
                             "cost_by_margin": {m: round(curve[m]["cost_per_1000_cycles"], 2) for m in margins},
                             "failures_by_margin": {m: round(curve[m]["failures_per_100"], 1) for m in margins}}
    out["E1_margin_rule_hgb"] = e1

    # E2: persistence x lead, HGB, cross-fold tau
    out["E2_persistence_hgb"] = {
        f"lead{lead}_k{k}": {kk: v for kk, v in evaluate(outcome_fn(sources["hgb"], lead, k), TAUS).items() if kk != "curve"}
        for lead in (0, 5, 10) for k in (1, 2, 3, 5)
    }

    # E4: model robustness at the primary scenario and ratio 20
    out["E4_model_robustness"] = {
        f"{name}_ratio{ratio}": {kk: v for kk, v in evaluate(outcome_fn(sources[name], LEAD, 1), TAUS, ratio).items() if kk != "curve"}
        for name in ("hgb", "rf", "gru") for ratio in (10, 20)
    }

    # E5: uncertainty-aware trigger with RF: fire when mean - z*std < tau
    e5 = {}
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        series = {int(u): (v["cycles"], (np.maximum(np.array(v["mean"]) - z * np.array(v["std"]), 0)).tolist()) for u, v in rf.items()}
        e5[f"z{z}"] = {kk: v for kk, v in evaluate(outcome_fn(series, LEAD, 1), TAUS).items() if kk != "curve"}
    out["E5_uncertainty_trigger_rf"] = e5

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o.item() if hasattr(o, "item") else o
    (ROOT / "docs" / "threshold_experiments.json").write_text(json.dumps(clean(out), indent=2) + "\n", encoding="utf-8")

    print("E1 (HGB, ratio10): lead, best margin, best cost, margin0 cost, tau10 cost, failures@best")
    for k, v in e1.items():
        print(k, v["best_margin"], round(v["best"]["cost_per_1000_cycles"], 2), round(v["margin0"]["cost_per_1000_cycles"], 2), round(v["tau10_default"]["cost_per_1000_cycles"], 2), round(v["best"]["failures_per_100"], 1))
    print("E2 cross-fold cost/1000 (lead5):", {k: round(v["cross_fold"]["cost_per_1000_cycles"], 2) for k, v in out["E2_persistence_hgb"].items() if "lead5" in k})
    print("E2 lead0/lead10:", {k: round(v["cross_fold"]["cost_per_1000_cycles"], 2) for k, v in out["E2_persistence_hgb"].items() if "lead5" not in k})
    for k, v in out["E4_model_robustness"].items():
        print("E4", k, round(v["cross_fold"]["cost_per_1000_cycles"], 2), "fail", round(v["cross_fold"]["failures_per_100"], 1), "best tau", v["best_in_sample_tau"], "range", v["tau_within_5pct_of_best"], "chosen", v["chosen_per_fold"])
    for k, v in e5.items():
        print("E5", k, round(v["cross_fold"]["cost_per_1000_cycles"], 2), "fail", round(v["cross_fold"]["failures_per_100"], 1), "unused", round(v["cross_fold"]["mean_unused_life"] or 0, 1), "best tau", v["best_in_sample_tau"])


if __name__ == "__main__":
    main()
