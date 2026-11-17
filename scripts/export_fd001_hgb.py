"""Train and export the selected FD001 window-HGB for the backend (model version 2.0.0).

Selection happened earlier by grouped CV (see docs/EVALUATION_RESULTS.md); this
script only refits that configuration on all 100 training engines and records an
input contract, drift-baseline statistics and golden predictions in a manifest.
Run it in the environment that will serve the model, because pickled estimators
are tied to the scikit-learn version.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import date

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.cmapss import load_fd001
from ml.evaluate import endpoint_metrics
from ml.features import (
    RUL_CAP, WINDOW_LENGTH, add_rul_targets, build_samples, select_informative_sensors,
    summarize_window, window_feature_names,
)
from scripts.run_fd001_evaluation import make_model

ARTIFACT = ROOT / "ml" / "artifacts" / "fd001_hgb_window.joblib"
MANIFEST = ROOT / "ml" / "artifacts" / "fd001_hgb_window_manifest.json"
MODEL_ID, VERSION, SEED = "fd001-hgb-window-rul", "2.0.0", 0
HEALTHY_CYCLES = 30  # drift baseline: first cycles of every training engine
GOLDEN_TEST_UNITS = (1, 2)  # official test windows (healthy-looking)
GOLDEN_FAILURE_UNIT = 1  # last 30 cycles of this training engine: a run-to-failure window


def main() -> None:
    data = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData")
    frame = add_rul_targets(data.train)
    sensors = select_informative_sensors(data.train)  # frozen into the manifest, never re-derived at serve time
    units = list(range(1, 101))
    train = build_samples(frame, units, sensors, "window")
    model = make_model("hgb", seed=SEED).fit(train.features, np.minimum(train.rul, RUL_CAP))
    joblib.dump(model, ARTIFACT, compress=3)

    test = build_samples(data.test, units, sensors, "window", endpoints="last")
    test_rul = data.test_rul.to_numpy(dtype=np.float64)
    metrics = endpoint_metrics(test_rul, model.predict(test.features))

    # Drift baseline and degradation direction, from training engines only.
    first = frame[frame["cycle"] <= HEALTHY_CYCLES][sensors]
    last = frame[frame["rul"] <= HEALTHY_CYCLES][sensors]
    healthy_mean, healthy_std = first.mean(), first.std(ddof=1)
    sign = np.sign(last.mean() - healthy_mean)

    full_columns = [c for c in data.test.columns if c.startswith(("operational_setting_", "sensor_"))]
    golden_windows, golden_raw = [], []
    sources = [(data.test, unit) for unit in GOLDEN_TEST_UNITS] + [(data.train, GOLDEN_FAILURE_UNIT)]
    for source, unit in sources:
        rows = source[source["unit_id"] == unit].sort_values("cycle").iloc[-WINDOW_LENGTH:]
        golden_windows.append({"cycles": rows["cycle"].astype(int).tolist(), "values": rows[full_columns].to_numpy(dtype=float).tolist()})
        golden_raw.append(float(model.predict(summarize_window(rows[sensors].to_numpy())[None, :])[0]))

    manifest = {
        "model_id": MODEL_ID, "version": VERSION, "status": "promoted_for_backend",
        "created_date": date.today().isoformat(),
        "artifact": "ml/artifacts/fd001_hgb_window.joblib",
        "artifact_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        "selection": "grouped 5-fold CV over training engines; see docs/EVALUATION_RESULTS.md",
        "training": {"units": "1-100 (all training engines)", "target": f"RUL capped at {RUL_CAP}", "hyperparameters": make_model("hgb", SEED).get_params(), "seed": SEED},
        "input_contract": {
            "window_length": WINDOW_LENGTH, "sensors": sensors, "feature_names": window_feature_names(sensors),
            "request_columns": full_columns, "dtype": "float64",
            "note": "Only the listed sensors are read; operational settings and uninformative sensors are ignored.",
        },
        "output_contract": {"raw_rul": "model output, uncapped below", "rul_cap": RUL_CAP, "operational_rul": f"clip(raw_rul, 0, {RUL_CAP})",
                            "note": f"values near {RUL_CAP} mean 'healthy, no usable estimate', not a measured remaining life"},
        "official_test_metrics": {k: float(metrics[k]) for k in ("rmse", "mae", "nasa_score", "low_rul_mae")},
        "drift_baseline": {
            "healthy_definition": f"first {HEALTHY_CYCLES} cycles of every training engine",
            "degradation_definition": f"sign of (mean over last {HEALTHY_CYCLES} cycles before failure minus healthy mean)",
            "sensors": {s: {"healthy_mean": float(healthy_mean[s]), "healthy_std": float(healthy_std[s]), "degradation_sign": int(sign[s])} for s in sensors},
        },
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "golden": {"tolerance": 1e-4, "windows": golden_windows, "expected_raw_rul": golden_raw},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ARTIFACT.name} and {MANIFEST.name}; official test RMSE {metrics['rmse']:.3f} NASA {metrics['nasa_score']:.0f}")


if __name__ == "__main__":
    main()
