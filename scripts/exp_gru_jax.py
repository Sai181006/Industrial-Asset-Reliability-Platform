"""EXPERIMENTAL: JAX stand-in for the torch GRU, on the shared evaluation protocol.

torch could not be installed in the analysis environment, so this reproduces the
architecture of ``ml/models/gru.py`` (1-layer GRU with 32 hidden units, 16-unit
ReLU head, dropout 0.1, Huber loss on a standardized target, Adam 1e-3, batch 256,
gradient clip 5, PyTorch gate order and initialization) in JAX. It answers "would
a capped, cleaned-feature GRU beat the tree models?". It does not produce a
deployable artifact: retrain in torch if the GRU wins.

Stages (each resumable through EVAL_CACHE_DIR): proxy [seeds], cv, test, refit, finalize.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.baseline import FEATURE_COLUMNS, TRAIN_UNIT_IDS
from ml.data.cmapss import load_fd001
from ml.evaluate import bootstrap_rmse_interval, endpoint_metrics, grouped_folds
from ml.features import (
    RUL_CAP, add_rul_targets, build_sequences, sample_truncation_points, select_informative_sensors,
)
from scripts.run_fd001_evaluation import CV_SEED, EVAL_HORIZON, cached

HIDDEN, HEAD, DROPOUT = 32, 16, 0.1
BATCH, LR, CLIP = 256, 1e-3, 5.0
MAX_EPOCHS, PATIENCE = 40, 5
PROXY_SEEDS = (7, 42, 77)
CAPPED_SEEDS = (7, 42, 77)
GRU_CV_SEED = 42
PROMOTED = {  # docs/promotion_metrics.json and docs/MODEL_PROMOTION.md (torch, uncapped, 24 raw features)
    "dev_validation_mae": {"seed7": 28.395, "seed42": 26.492, "seed77": 26.55, "mean": 27.147},
    "dev_validation_rmse_mean": 41.747, "dev_validation_low_rul_mae_mean": 5.043,
    "test_seed42": {"rmse": 25.282, "mae": 18.305, "low_rul_mae": 4.998},
}


def init_params(seed: int, input_size: int) -> dict:
    rng = np.random.RandomState(seed)
    k = 1.0 / np.sqrt(HIDDEN)
    kh = 1.0 / np.sqrt(HEAD)
    u = lambda shape, bound: rng.uniform(-bound, bound, shape).astype(np.float32)
    return {
        "w_ih": u((3 * HIDDEN, input_size), k), "w_hh": u((3 * HIDDEN, HIDDEN), k),
        "b_ih": u((3 * HIDDEN,), k), "b_hh": u((3 * HIDDEN,), k),
        "w1": u((HEAD, HIDDEN), k), "b1": u((HEAD,), k), "w2": u((1, HEAD), kh), "b2": u((1,), kh),
    }


def forward(params, x, key=None, rate=0.0):
    def step(h, x_t):
        gi = x_t @ params["w_ih"].T + params["b_ih"]
        gh = h @ params["w_hh"].T + params["b_hh"]
        i_r, i_z, i_n = jnp.split(gi, 3, axis=-1)
        h_r, h_z, h_n = jnp.split(gh, 3, axis=-1)
        r, z = jax.nn.sigmoid(i_r + h_r), jax.nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h, None

    h, _ = jax.lax.scan(step, jnp.zeros((x.shape[0], HIDDEN), x.dtype), jnp.swapaxes(x, 0, 1))
    if key is not None and rate > 0:
        h = jnp.where(jax.random.bernoulli(key, 1.0 - rate, h.shape), h / (1.0 - rate), 0.0)
    return (jax.nn.relu(h @ params["w1"].T + params["b1"]) @ params["w2"].T + params["b2"])[:, 0]


def huber(prediction, target):
    error = jnp.abs(prediction - target)
    return jnp.mean(jnp.where(error <= 1.0, 0.5 * error**2, error - 0.5))


@jax.jit
def train_step(params, m, v, t, xb, yb, key):
    loss, grads = jax.value_and_grad(lambda p: huber(forward(p, xb, key, DROPOUT), yb))(params)
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, CLIP / (norm + 1e-6)), grads)
    t = t + 1.0
    m = jax.tree_util.tree_map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
    v = jax.tree_util.tree_map(lambda a, g: 0.999 * a + 0.001 * g * g, v, grads)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - LR * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-8), params, m, v
    )
    return params, m, v, t, loss


_forward_eval = jax.jit(lambda params, x: forward(params, x))


def train_model(x, y, epochs, seed, val=None):
    """Train from scratch; with ``val`` apply patience-based early stopping and return the best epoch."""
    params = jax.tree_util.tree_map(jnp.asarray, init_params(seed, x.shape[2]))
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    t, key, rng = jnp.float32(0.0), jax.random.PRNGKey(seed), np.random.RandomState(seed)
    xj, yj = jnp.asarray(x), jnp.asarray(y, dtype=jnp.float32)
    history, best, stale = [], float("inf"), 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), BATCH):
            index = order[start : start + BATCH]
            if len(index) < BATCH:  # wrap around so every batch has one shape (one compilation)
                index = np.concatenate([index, order[: BATCH - len(index)]])
            key, sub = jax.random.split(key)
            params, m, v, t, _ = train_step(params, m, v, t, xj[index], yj[index], sub)
        if val is not None:
            loss = float(huber(_forward_eval(params, jnp.asarray(val[0])), jnp.asarray(val[1], dtype=jnp.float32)))
            history.append(loss)
            if loss < best - 1e-4:
                best, stale = loss, 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    return params, history


def fit_gru(frame, train_units, columns, seed, cap, fixed_epochs=None):
    """Scaler and epoch choice use training engines only; mirrors the promoted protocol."""
    units = sorted(train_units)
    rows = frame[frame["unit_id"].isin(units)][list(columns)].to_numpy(dtype=np.float64)
    mean, std = rows.mean(axis=0), rows.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    windows, rul, owners, _ = build_sequences(frame, units, columns)
    x = ((windows - mean) / std).astype(np.float32)
    y = np.minimum(rul, cap) if cap else rul
    if fixed_epochs is None:  # hold out the last 20% of engines to choose the epoch count
        split = int(round(0.8 * len(units)))
        fit_mask = np.isin(owners, units[:split])
        mu, sd = y[fit_mask].mean(), y[fit_mask].std()
        _, history = train_model(
            x[fit_mask], (y[fit_mask] - mu) / sd, MAX_EPOCHS, seed,
            val=(x[~fit_mask], (y[~fit_mask] - mu) / sd),
        )
        epochs = int(np.argmin(history)) + 1
    else:
        epochs = fixed_epochs
    t_mean, t_std = float(y.mean()), float(y.std())
    params, _ = train_model(x, (y - t_mean) / t_std, epochs, seed)
    return {"params": params, "mean": mean, "std": std, "t_mean": t_mean, "t_std": t_std, "epochs": epochs}


def predict(model, windows):
    x = ((windows - model["mean"]) / model["std"]).astype(np.float32)
    out = np.concatenate([np.asarray(_forward_eval(model["params"], jnp.asarray(x[i : i + 4096]))) for i in range(0, len(x), 4096)])
    return out.astype(np.float64) * model["t_std"] + model["t_mean"]


def mean_metrics(rows, keys=("rmse", "mae", "nasa_score", "low_rul_mae")):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {
        "rmse_seed_min": float(min(r["rmse"] for r in rows)), "rmse_seed_max": float(max(r["rmse"] for r in rows)),
    }


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    started = time.time()
    data = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData")
    frame, test_rul = add_rul_targets(data.train), data.test_rul.to_numpy(dtype=np.float64)
    sensors = select_informative_sensors(data.train)
    all_units, dev_train = list(range(1, 101)), list(TRAIN_UNIT_IDS)
    lengths = frame.groupby("unit_id")["cycle"].max().to_dict()
    test_windows = lambda columns: build_sequences(data.test, range(1, 101), columns, endpoints="last")[0]

    def proxy(seed):  # promoted setup: 24 raw features, uncapped, units 1-80, fixed 11 epochs
        model = fit_gru(frame, dev_train, FEATURE_COLUMNS, seed, cap=None, fixed_epochs=11)
        dev_x, dev_y, _, _ = build_sequences(frame, range(81, 101), FEATURE_COLUMNS)
        return {"seed": seed, "dev_validation": endpoint_metrics(dev_y, predict(model, dev_x)),
                "test": endpoint_metrics(test_rul, predict(model, test_windows(FEATURE_COLUMNS)))}

    def cv_fold(i, train_units, held_units):
        cuts = sample_truncation_points(lengths, per_unit=10, seed=CV_SEED)
        model = fit_gru(frame, train_units, sensors, GRU_CV_SEED, cap=RUL_CAP)
        windows, _, owners, cycles = build_sequences(frame, held_units, sensors)
        raw = predict(model, windows)
        oof = {str(u): {"cycles": cycles[owners == u].tolist(), "rul": np.maximum(raw[owners == u], 0).tolist()} for u in held_units}
        tw, tr, _, _ = build_sequences(frame, held_units, sensors, endpoints={u: cuts[u] for u in held_units})
        pred = predict(model, tw)
        metrics = endpoint_metrics(tr, pred, truth_cap=EVAL_HORIZON) | {"rmse_raw_truth": endpoint_metrics(tr, pred)["rmse"]}
        return {"metrics": metrics, "epochs": model["epochs"], "oof": oof}

    def capped_test(seed, units):
        model = fit_gru(frame, units, sensors, seed, cap=RUL_CAP)
        pred = predict(model, test_windows(sensors))
        return {"metrics": endpoint_metrics(test_rul, pred), "prediction": pred.tolist(), "epochs": model["epochs"]}

    if stage in ("proxy", "all"):
        seeds = tuple(int(s) for s in sys.argv[2:]) or PROXY_SEEDS
        for seed in seeds:
            r = cached(f"gru_proxy_{seed}", lambda: proxy(seed))
            print(f"proxy seed {seed}: dev MAE {r['dev_validation']['mae']:.2f} RMSE {r['dev_validation']['rmse']:.2f} lowMAE {r['dev_validation']['low_rul_mae']:.2f} | test RMSE {r['test']['rmse']:.2f} MAE {r['test']['mae']:.2f} ({time.time()-started:.0f}s)", flush=True)
    if stage in ("cv", "all"):
        for i, (train_units, held_units) in enumerate(grouped_folds(all_units, 5, CV_SEED)):
            r = cached(f"gru_cv_fold{i}", lambda: cv_fold(i, train_units, held_units))
            print(f"cv fold {i}: rmse(clipped) {r['metrics']['rmse']:.2f} epochs {r['epochs']} ({time.time()-started:.0f}s)", flush=True)
    if stage in ("test", "all"):
        for seed in CAPPED_SEEDS:
            r = cached(f"gru_capped_test80_{seed}", lambda: capped_test(seed, dev_train))
            print(f"test(1-80) seed {seed}: rmse {r['metrics']['rmse']:.2f} nasa {r['metrics']['nasa_score']:.0f} epochs {r['epochs']} ({time.time()-started:.0f}s)", flush=True)
    if stage in ("refit", "all"):
        for seed in CAPPED_SEEDS:
            r = cached(f"gru_capped_test100_{seed}", lambda: capped_test(seed, all_units))
            print(f"refit(1-100) seed {seed}: rmse {r['metrics']['rmse']:.2f} nasa {r['metrics']['nasa_score']:.0f} epochs {r['epochs']} ({time.time()-started:.0f}s)", flush=True)
    if stage in ("finalize", "all"):
        proxies = [cached(f"gru_proxy_{s}", lambda: None) for s in PROXY_SEEDS]
        folds = [cached(f"gru_cv_fold{i}", lambda: None) for i in range(5)]
        t80 = [cached(f"gru_capped_test80_{s}", lambda: None) for s in CAPPED_SEEDS]
        t100 = [cached(f"gru_capped_test100_{s}", lambda: None) for s in CAPPED_SEEDS]
        if any(x is None for x in proxies + folds + t80 + t100):
            raise SystemExit("finalize needs every stage cached; run proxy, cv, test and refit first")
        keys = ("rmse", "mae", "nasa_score", "low_rul_mae", "rmse_raw_truth")
        cv = {k: float(np.mean([f["metrics"][k] for f in folds])) for k in keys} | {"rmse_fold_sd": float(np.std([f["metrics"]["rmse"] for f in folds], ddof=1))}
        oof = {}
        for f in folds:
            oof.update(f["oof"])
        cached("oof_gru", lambda: oof)
        ci = bootstrap_rmse_interval(test_rul, np.asarray(t100[0]["prediction"]))
        output = {
            "note": "JAX stand-in for the torch GRU (see scripts/exp_gru_jax.py); not a deployable artifact.",
            "architecture": {"hidden": HIDDEN, "head": HEAD, "dropout": DROPOUT, "batch": BATCH, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
            "proxy_check": {"promoted_torch": PROMOTED, "jax_seeds": proxies,
                            "jax_dev_validation_mean": {k: float(np.mean([p["dev_validation"][k] for p in proxies])) for k in ("mae", "rmse", "low_rul_mae")},
                            "jax_test_seed42": next(p["test"] for p in proxies if p["seed"] == 42)},
            "cv_grouped_5fold_truncated_endpoints": cv | {"epochs_selected": [f["epochs"] for f in folds]},
            "test_units_1_80_capped": mean_metrics([r["metrics"] for r in t80]) | {"epochs_selected": [r["epochs"] for r in t80]},
            "test_units_1_100_capped_refit": mean_metrics([r["metrics"] for r in t100]) | {"rmse_bootstrap_95ci_seed7": list(ci), "epochs_selected": [r["epochs"] for r in t100]},
        }
        (ROOT / "docs" / "gru_comparison.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print("finalized docs/gru_comparison.json", flush=True)


if __name__ == "__main__":
    main()
