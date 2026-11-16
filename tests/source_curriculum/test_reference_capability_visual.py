from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts/tools"))
from reference_capability_visual_replay import frame_indices, compare_arrays


def test_terminal_frame_included_once_and_reset_frames_excluded():
    assert frame_indices(1) == [0]
    assert frame_indices(6) == [0, 5]
    assert frame_indices(8) == [0, 5, 7]
    assert frame_indices(28, 1) == list(range(28))


def test_reproduction_detects_shape_nonfinite_and_small_state_changes():
    old = {"x": np.array([1., 2.]), "flag": np.array([True, False])}
    assert all(r["exact"] for r in compare_arrays(old, old).values())
    changed = {"x": np.array([1., 2.00001]), "flag": np.array([True, True])}
    assert not any(r["exact"] for r in compare_arrays(old, changed).values())
    assert not compare_arrays({"x": np.array([np.nan])}, {"x": np.array([0.])})["x"]["finite"]
    assert not compare_arrays({"x": np.zeros(2)}, {"x": np.zeros(3)})["x"]["exact"]
