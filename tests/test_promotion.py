import json
from pathlib import Path

import numpy as np

from ml.data.cmapss import load_fd001
from ml.sequence import prepare_development_sequences
from scripts.promote_fd001_gru import final_test_windows


ROOT = Path(__file__).parents[1]
RAW_DIR = ROOT / "data" / "raw" / "source" / "CMAPSSData"


def test_official_test_window_contract():
    if not RAW_DIR.exists():
        return
    fd001 = load_fd001(RAW_DIR)
    sequences = prepare_development_sequences(fd001.train, 30)
    windows, units = final_test_windows(fd001.test, sequences.scaler)
    assert windows.shape == (100, 30, 24)
    np.testing.assert_array_equal(units, np.arange(1, 101))
    assert len(fd001.test_rul) == len(windows)


def test_promotion_manifest_contract_when_available():
    path = ROOT / "ml" / "artifacts" / "fd001_gru_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "promoted_for_backend_mvp"
    assert manifest["input_contract"]["shape"] == [30, 24]
    assert manifest["training"]["promoted_seed"] == 42
    assert manifest["output_contract"]["operational_rul"].startswith("max(")

