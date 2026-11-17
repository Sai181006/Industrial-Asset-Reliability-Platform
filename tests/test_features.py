from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.cmapss import SENSOR_COLUMNS, load_fd001
from ml.features import (
    RUL_CAP,
    add_rul_targets,
    build_samples,
    build_sequences,
    sample_truncation_points,
    select_informative_sensors,
    summarize_window,
    window_feature_names,
)


RAW_DIR = Path(__file__).parents[1] / "data" / "raw" / "source" / "CMAPSSData"
EXPECTED_SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]


def synthetic_frame(lengths=(40, 45)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for unit_id, length in enumerate(lengths, start=1):
        for cycle in range(1, length + 1):
            row = {"unit_id": unit_id, "cycle": cycle}
            row.update({column: rng.normal() + cycle * 0.01 for column in SENSOR_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_select_informative_sensors_drops_constant_and_two_valued():
    frame = synthetic_frame()
    frame["sensor_1"] = 5.0
    frame["sensor_6"] = np.where(frame["cycle"] % 2 == 0, 1.0, 2.0)
    selected = select_informative_sensors(frame)
    assert "sensor_1" not in selected and "sensor_6" not in selected
    assert len(selected) == len(SENSOR_COLUMNS) - 2


def test_real_fd001_selection_is_the_documented_fourteen():
    if not RAW_DIR.exists():
        pytest.skip("Official FD001 files are not installed")
    assert select_informative_sensors(load_fd001(RAW_DIR).train) == EXPECTED_SENSORS


def test_rul_targets_cap_without_mutating_input():
    frame = synthetic_frame((200,))
    original = frame.copy()
    result = add_rul_targets(frame, cap=125)
    assert frame.equals(original)
    assert result["rul"].iloc[0] == 199 and result["rul_capped"].iloc[0] == 125
    assert result["rul"].iloc[-1] == 0 and result["rul_capped"].iloc[-1] == 0
    assert RUL_CAP == 125


def test_summarize_window_on_linear_ramp():
    ramp = np.arange(30, dtype=float)[:, None] * np.array([[1.0, -2.0]])
    last, mean, slope = np.split(summarize_window(ramp), 3)
    np.testing.assert_allclose(last, [29, -58])
    np.testing.assert_allclose(mean, [14.5, -29])
    np.testing.assert_allclose(slope, [1.0, -2.0])
    assert len(window_feature_names(["a", "b"])) == 6


def test_summarize_window_rejects_bad_shapes():
    with pytest.raises(ValueError):
        summarize_window(np.zeros(30))
    with pytest.raises(ValueError):
        summarize_window(np.zeros((1, 3)))


def test_truncation_points_are_seeded_and_in_bounds():
    lengths = {1: 100, 2: 200}
    first = sample_truncation_points(lengths, per_unit=10, seed=3)
    assert first == sample_truncation_points(lengths, per_unit=10, seed=3)
    assert first != sample_truncation_points(lengths, per_unit=10, seed=4)
    for unit_id, ends in first.items():
        assert len(ends) == 10
        assert all(29 <= end < lengths[unit_id] for end in ends)
        assert all(end + 1 <= 0.97 * lengths[unit_id] for end in ends)


def test_build_samples_uses_identical_rows_for_both_representations():
    frame = add_rul_targets(synthetic_frame())
    columns = SENSOR_COLUMNS[:4]
    current = build_samples(frame, [1, 2], columns, "current")
    window = build_samples(frame, [1, 2], columns, "window")
    assert current.features.shape == (len(current.rul), 4)
    assert window.features.shape == (len(window.rul), 12)
    np.testing.assert_array_equal(current.unit_ids, window.unit_ids)
    np.testing.assert_array_equal(current.end_cycles, window.end_cycles)
    assert current.end_cycles.min() == 30  # no window starts before cycle 1


def test_build_samples_last_endpoint_and_unit_isolation():
    frame = add_rul_targets(synthetic_frame())
    last = build_samples(frame, [1, 2], SENSOR_COLUMNS[:3], "window", endpoints="last")
    assert list(last.unit_ids) == [1, 2] and list(last.end_cycles) == [40, 45]
    assert list(last.rul) == [0.0, 0.0]
    # A window never mixes engines: its 'last' feature equals that engine's own final row.
    unit1_last = frame[frame.unit_id == 1].sort_values("cycle")[SENSOR_COLUMNS[:3]].iloc[-1].to_numpy()
    np.testing.assert_allclose(last.features[0, :3], unit1_last)


def test_build_samples_validates_inputs():
    frame = add_rul_targets(synthetic_frame((20,)))
    with pytest.raises(ValueError):
        build_samples(frame, [1], SENSOR_COLUMNS[:2], "current")  # fewer than 30 cycles
    with pytest.raises(ValueError):
        build_samples(add_rul_targets(synthetic_frame()), [1], SENSOR_COLUMNS[:2], "bogus")
    with pytest.raises(ValueError):
        build_samples(add_rul_targets(synthetic_frame()), [1], SENSOR_COLUMNS[:2], "window", endpoints={1: [5]})


def test_frames_without_rul_yield_nan_targets():
    frame = synthetic_frame()
    result = build_samples(frame, [1], SENSOR_COLUMNS[:2], "current", endpoints="last")
    assert np.isnan(result.rul).all() and np.isnan(result.rul_capped).all()


def test_build_sequences_matches_samples_and_window_slices():
    frame = add_rul_targets(synthetic_frame())
    columns = SENSOR_COLUMNS[:3]
    windows, rul, owners, cycles = build_sequences(frame, [1, 2], columns)
    samples = build_samples(frame, [1, 2], columns, "current")
    assert windows.shape == (len(rul), 30, 3)
    np.testing.assert_array_equal(owners, samples.unit_ids)
    np.testing.assert_array_equal(cycles, samples.end_cycles)
    np.testing.assert_allclose(rul, samples.rul)
    np.testing.assert_allclose(windows[:, -1, :], samples.features, rtol=1e-6)  # last row is the endpoint
    unit1 = frame[frame.unit_id == 1].sort_values("cycle")[columns].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(windows[0], unit1[:30])  # first window never leaves its engine


def test_build_sequences_last_and_mapping_endpoints_and_errors():
    frame = add_rul_targets(synthetic_frame())
    windows, rul, owners, cycles = build_sequences(frame, [1, 2], SENSOR_COLUMNS[:2], endpoints="last")
    assert windows.shape == (2, 30, 2) and list(cycles) == [40, 45] and list(rul) == [0.0, 0.0]
    mapped, _, mapped_units, _ = build_sequences(frame, [2], SENSOR_COLUMNS[:2], endpoints={2: [29, 40]})
    assert mapped.shape == (2, 30, 2) and list(mapped_units) == [2, 2]
    with pytest.raises(ValueError):
        build_sequences(frame, [1], SENSOR_COLUMNS[:2], endpoints={1: [10]})
    no_rul = synthetic_frame()
    assert np.isnan(build_sequences(no_rul, [1], SENSOR_COLUMNS[:2], endpoints="last")[1]).all()
