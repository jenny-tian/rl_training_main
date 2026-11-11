import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
from source_block_b3_protocol import CASES, FIELDS, categories, first_episode_masks, schedule, summarize


def test_command_boundaries_and_directions():
    assert len(CASES) == 16
    for case in CASES:
        assert schedule(case, 1.98) == (0, 0, 0, None)
    assert schedule("turn_low_right", 2)[2] == -.15
    assert schedule("stop_left", 7.98)[2] == .5
    assert schedule("stop_left", 8)[2] == 0
    assert schedule("ramp_left", 9)[2] == .25
    assert schedule("ramp_left", 10)[2] == 0
    assert schedule("reverse_right", 8)[2] == .5
    assert schedule("heading_right", 2)[3] == -1.2
    assert schedule("lateral_left", 8)[1] == 0


def test_reset_samples_never_enter_metrics():
    ended = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=bool)
    first, valid = first_episode_masks(ended)
    assert first[:, 0].tolist() == [True, True, False, False]
    assert valid[:, 0].tolist() == [True, False, False, False]
    trace = np.zeros((4, 2, len(FIELDS) + 1), dtype=np.float32)
    trace[~valid, 5] = 1000  # reset/terminal velocity must not contaminate RMS
    trace[..., len(FIELDS)] = 1.5  # reward rate; no second dt conversion
    report = summarize(trace, ended, ended, {"all": np.ones(2, bool)}, ["yaw"], .02, "native")
    row = report["rows"][0]
    assert row["samples"] == 4
    assert row["rms_yaw"] == 0
    assert row["reward_rates"]["yaw"] == 1.5
    assert row["seconds"] == pytest.approx(.08)
    assert report["episodes"]["all"]["failures"] == 2


def test_actual_installed_standing_reward_conflicts_with_pure_turn():
    if not Path("/home/tjx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py").is_file():
        pytest.skip("installed Isaac Lab source is required for this host audit")
    from source_block_b3_reward_reproducer import reproduce
    report = reproduce()
    assert report["status"] == "passed"
    assert report["original_weighted_rate"][1] < 0
    assert report["candidate_weighted_rate"][1] == 0
    assert report["original_weighted_rate"][0] == report["candidate_weighted_rate"][0]


def test_native_command_categories_partition_all_samples():
    commands = np.array([[0, 0, 0], [0, 0, .1], [0, 0, -.5], [.3, 0, 0],
                         [0, -.3, 0], [.2, .3, 0], [.2, .3, 1e-8], [.3, 0, .2]])
    masks = categories(commands)
    assert np.array_equal(sum(masks.values()), np.ones(len(commands)))
    assert masks["mixed"][5:8].all()
