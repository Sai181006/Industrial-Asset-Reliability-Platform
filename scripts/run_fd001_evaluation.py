"""Shared-protocol FD001 evaluation: cleaned features x target cap x model.

Protocol (declared before any test label is read):
  * Cap is fixed at 125 by convention; it is not tuned on test data.
  * Candidates are ranked only by grouped 5-fold CV on the 100 training engines,
    scored on seeded truncated endpoints (mimicking the official test cuts).
  * Official test endpoints are scored once, after the ranking is fixed.
  * Predictions are clipped at zero and compared against uncapped true RUL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.baseline import FEATURE_COLUMNS, TRAIN_UNIT_IDS
from ml.data.cmapss import load_fd001
from ml.evaluate import bootstrap_rmse_interval, endpoint_metrics, grouped_folds
from ml.features import (
    RUL_CAP,
    WINDOW_LENGTH,
    add_rul_targets,
    build_samples,
    sample_truncation_points,
    select_informative_sensors,
)

CV_SEED = 0
# CV truncation can leave true RUL near 290; no model can know that from sensors,
# and capped training would be penalized for it. CV therefore scores against truth
# clipped at this horizon (raw-truth numbers are reported as secondary). The change
# was made after seeing CV output only, before any test label was scored.
EVAL_HORIZON = 150
CAP_SENSITIVITY = (100, 115, 125, 130, 140)
RF_SEEDS = (0, 1, 2)
MODELS = ("rf", "hgb")
REPRESENTATIONS = ("current", "window")
TARGETS = ("uncapped", "capped")


CACHE_DIR = Path(os.environ["EVAL_CACHE_DIR"]) if os.environ.get("EVAL_CACHE_DIR") else None


def cached(name: str, compute):
    """Optional resume cache for slow, fully seeded steps (off unless EVAL_CACHE_DIR is set)."""
    if CACHE_DIR is None:
        return compute()
    path = CACHE_DIR / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    result = compute()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def make_model(name: str, seed: int):
    if name == "rf":  # same hyperparameters as the existing baseline
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features="sqrt", n_jobs=2, random_state=seed
        )
    return HistGradientBoostingRegressor(  # same hyperparameters as the existing baseline
        learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )


def target_of(samples, target: str, cap: int = RUL_CAP):
    return samples.rul if target == "uncapped" else np.minimum(samples.rul, cap)


def cross_validate(frame, units, columns, representation, target, model_name, cap=RUL_CAP):
    """Grouped CV over training engines; held-out scored on truncated endpoints."""
    lengths = frame.groupby("unit_id")["cycle"].max().to_dict()
    cuts = sample_truncation_points(lengths, per_unit=10, seed=CV_SEED)
    fold_metrics = []
    for train_units, held_units in grouped_folds(list(units), n_splits=5, seed=CV_SEED):
        train = build_samples(frame, train_units, columns, representation)
        held = build_samples(
            frame, held_units, columns, representation,
            endpoints={u: cuts[u] for u in held_units},
        )
        model = make_model(model_name, seed=CV_SEED).fit(train.features, target_of(train, target, cap))
        raw = model.predict(held.features)
        primary = endpoint_metrics(held.rul, raw, truth_cap=EVAL_HORIZON)
        primary["rmse_raw_truth"] = endpoint_metrics(held.rul, raw)["rmse"]
        fold_metrics.append(primary)
    return {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in ("rmse", "mae", "nasa_score", "low_rul_mae", "rmse_raw_truth")
    } | {"rmse_fold_sd": float(np.std([m["rmse"] for m in fold_metrics], ddof=1))}


def test_scores(frame, test_frame, test_rul, units, columns, representation, target, model_name, seeds=(0,)):
    """Fit on ``units`` and score the official test endpoints once per seed."""
    train = build_samples(frame, units, columns, representation)
    test = build_samples(test_frame, range(1, 101), columns, representation, endpoints="last")
    per_seed, predictions = [], []
    for seed in seeds:
        model = make_model(model_name, seed=seed).fit(train.features, target_of(train, target))
        raw = model.predict(test.features)
        predictions.append(raw)
        per_seed.append(endpoint_metrics(test_rul, raw))
    summary = {
        key: float(np.mean([m[key] for m in per_seed])) for key in ("rmse", "mae", "nasa_score", "low_rul_mae")
    }
    summary["rmse_seed_min"] = float(min(m["rmse"] for m in per_seed))
    summary["rmse_seed_max"] = float(max(m["rmse"] for m in per_seed))
    summary["negative_raw_predictions"] = int(per_seed[0]["negative_raw_predictions"])
    return summary, predictions[0]


def run_test(name: str, *args, **kwargs):
    """``test_scores`` behind the resume cache; returns (summary, seed-0 prediction)."""
    def compute():
        summary, prediction = test_scores(*args, **kwargs)
        return {"summary": summary, "prediction": prediction.tolist()}

    payload = cached(name, compute)
    return payload["summary"], np.asarray(payload["prediction"])


def main() -> None:
    started = time.time()
    data = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData")
    frame = add_rul_targets(data.train)
    test_rul = data.test_rul.to_numpy(dtype=np.float64)
    sensors = select_informative_sensors(data.train)
    assert len(sensors) == 14, f"expected 14 informative sensors, found {sensors}"
    all_units = list(range(1, 101))

    # 1) Selection: grouped CV on training engines only.
    grid = []
    for representation in REPRESENTATIONS:
        for target in TARGETS:
            for model_name in MODELS:
                cv = cached(
                    f"cv_{representation}_{target}_{model_name}",
                    lambda: cross_validate(frame, all_units, sensors, representation, target, model_name),
                )
                grid.append({"representation": representation, "target": target, "model": model_name, "cv": cv})
                print(f"CV {representation:7s} {target:8s} {model_name:3s} rmse={cv['rmse']:.2f} (raw-truth {cv['rmse_raw_truth']:.2f}) nasa={cv['nasa_score']:.0f}", flush=True)
    ranked = sorted(grid, key=lambda row: (row["cv"]["rmse"], row["cv"]["nasa_score"]))
    best = ranked[0]
    print("SELECTED:", best["representation"], best["target"], best["model"], flush=True)

    # 2) Cap sensitivity for the selected family (CV only; the cap stays at 125).
    sensitivity = {}
    for cap in CAP_SENSITIVITY:
        cv = cached(
            f"cap_{best['representation']}_{best['model']}_{cap}",
            lambda: cross_validate(frame, all_units, sensors, best["representation"], "capped", best["model"], cap=cap),
        )
        sensitivity[str(cap)] = {"rmse": cv["rmse"], "nasa_score": cv["nasa_score"]}
        print(f"CAP {cap} cv rmse={cv['rmse']:.2f}", flush=True)

    # 3) Official test endpoints, scored once. Everything fit on units 1-80 for
    #    comparability with the promoted GRU, which also saw only units 1-80.
    train_units = list(TRAIN_UNIT_IDS)
    test_rows = []
    for row in grid:
        scores, _ = run_test(
            f"test80_{row['representation']}_{row['target']}_{row['model']}",
            frame, data.test, test_rul, train_units, sensors,
            row["representation"], row["target"], row["model"],
            seeds=RF_SEEDS if row["model"] == "rf" else (0,),
        )
        test_rows.append({**{k: row[k] for k in ("representation", "target", "model")}, "features": "14 sensors", "test": scores})
    for model_name in MODELS:  # reference: the existing baseline's 24 raw features, uncapped
        scores, _ = run_test(
            f"test80_ref_{model_name}",
            frame, data.test, test_rul, train_units, FEATURE_COLUMNS,
            "current", "uncapped", model_name, seeds=RF_SEEDS if model_name == "rf" else (0,),
        )
        test_rows.append({"representation": "current", "target": "uncapped", "model": model_name, "features": "24 raw (existing baseline)", "test": scores})

    # 4) Selected configuration refit on all 100 training engines.
    refit, refit_prediction = run_test(
        "test100_selected",
        frame, data.test, test_rul, all_units, sensors,
        best["representation"], best["target"], best["model"],
        seeds=RF_SEEDS if best["model"] == "rf" else (0,),
    )
    low, high = bootstrap_rmse_interval(test_rul, refit_prediction)
    refit["rmse_bootstrap_95ci_seed0"] = [low, high]

    output = {
        "protocol": {
            "cap": RUL_CAP, "window_length": WINDOW_LENGTH, "informative_sensors": sensors,
            "selection": "grouped 5-fold CV over units 1-100, 10 seeded truncated endpoints per held-out engine, truth clipped at 150",
            "cv_eval_horizon": EVAL_HORIZON,
            "test_scored_once_after_selection": True, "predictions_clipped_at_zero": True,
            "reference_gru_from_docs": {"source": "docs/MODEL_PROMOTION.md", "test_rmse": 25.282, "test_mae": 18.305, "low_rul_mae": 4.998, "trained_on_units": "1-80", "target": "uncapped", "nasa_score": None},
        },
        "cv_grid": grid,
        "selected": {k: best[k] for k in ("representation", "target", "model")},
        "cap_sensitivity_cv": sensitivity,
        "test_units_1_80": test_rows,
        "selected_refit_units_1_100": refit,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (ROOT / "docs" / "evaluation_metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
