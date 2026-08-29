"""Persist sequence feasibility and grouped baseline robustness evidence."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import add_linear_train_rul, load_fd001
from ml.sequence import sequence_feasibility


def grouped_robustness(frame: pd.DataFrame) -> list[dict[str, float | int]]:
    labelled = add_linear_train_rul(frame)
    splitter = GroupKFold(n_splits=5)
    rows = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(labelled, groups=labelled.unit_id), start=1
    ):
        train, validation = labelled.iloc[train_index], labelled.iloc[validation_index]
        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=1,
            random_state=42,
        )
        model.fit(train[FEATURE_COLUMNS], train.rul)
        prediction = model.predict(validation[FEATURE_COLUMNS])
        low = validation.rul.to_numpy() <= 30
        rows.append(
            {
                "fold": fold,
                "validation_units": int(validation.unit_id.nunique()),
                "mae": float(mean_absolute_error(validation.rul, prediction)),
                "rmse": float(mean_squared_error(validation.rul, prediction) ** 0.5),
                "low_rul_mae": float(mean_absolute_error(validation.rul[low], prediction[low])),
            }
        )
    return rows


def main() -> None:
    train = load_fd001(ROOT / "data" / "raw" / "source" / "CMAPSSData").train
    windows = [sequence_feasibility(train, value) for value in (20, 30, 50)]
    folds = grouped_robustness(train)
    summary = {
        "target_decision": "uncapped linear RUL retained for direct cycle semantics",
        "window_decision": 30,
        "window_reason": "all engines retained; 30 cycles adds 50% history over 20 while retaining 94.7% as many sequences; 50 loses substantially more samples",
        "windows": windows,
        "grouped_random_forest_folds": folds,
        "grouped_random_forest_summary": {
            key: {"mean": float(np.mean([row[key] for row in folds])), "std": float(np.std([row[key] for row in folds], ddof=1))}
            for key in ("mae", "rmse", "low_rul_mae")
        },
    }
    path = ROOT / "docs" / "sequence_preparation.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(windows).to_string(index=False))
    print(pd.DataFrame(folds).to_string(index=False))
    print(json.dumps(summary["grouped_random_forest_summary"], indent=2))


if __name__ == "__main__":
    main()
