"""Run and persist the frozen FD001 development-baseline evaluation."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.baseline import evaluate_baselines
from ml.data.cmapss import load_fd001


def main() -> None:
    raw_dir = ROOT / "data" / "raw" / "source" / "CMAPSSData"
    train = load_fd001(raw_dir).train
    metrics, metadata = evaluate_baselines(train)

    output = {"metadata": metadata, "metrics": metrics}
    metrics_path = ROOT / "docs" / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    table = pd.DataFrame(metrics).T
    ax = table[["mae", "rmse", "low_rul_mae", "low_rul_rmse"]].plot.bar(
        figsize=(10, 5), rot=0, title="FD001 validation baseline errors"
    )
    ax.set_ylabel("cycles")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROOT / "docs" / "baseline_metrics.png", dpi=160)
    plt.close()

    print(table.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
