"""Different initial stair skills must have separate retention floors."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
from source_block_b2_protocol import decision


def fixture(up, down):
    block = dict(initial_state_hashes_by_column=["same"] * 20,
                 columns=[dict(complete=32) for _ in range(20)])
    block["columns"][4]["complete"] = up
    block["columns"][5]["complete"] = down
    flat = dict(metrics=dict(rms_xy_error=.05, rms_yaw_error=.07, falls_per_episode=.02))
    return block, flat


def test_low_skill_baseline_can_acquire_upstairs_but_retains_descent():
    baseline, flat = fixture(0, 19)
    block, _ = fixture(2, 17)
    assert decision(block, flat, baseline)["continue_training"]
    block["columns"][5]["complete"] = 16
    assert not decision(block, flat, baseline)["continue_training"]


def test_strong_upstairs_floor_and_independent_flat_gate():
    baseline, flat = fixture(32, 0)
    block, _ = fixture(29, 15)
    assert not decision(block, flat, baseline)["continue_training"]
    block["columns"][4]["complete"] = 30
    assert decision(block, flat, baseline)["continue_training"]
    flat["metrics"]["rms_yaw_error"] = .1
    assert not decision(block, flat, baseline)["continue_training"]


def test_changed_initial_states_are_not_a_paired_result():
    baseline, flat = fixture(32, 0)
    block = copy.deepcopy(baseline)
    block["initial_state_hashes_by_column"][0] = "changed"
    with pytest.raises(ValueError, match="initial states"):
        decision(block, flat, baseline)
