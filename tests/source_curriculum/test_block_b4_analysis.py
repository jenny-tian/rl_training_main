"""Paired response comparisons must exclude failed/reset intervals."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
from source_block_b3_protocol import CASES, FIELDS, REPEATS
from source_block_b4_analyze import paired_metric


def test_common_survivor_mask_excludes_large_reset_values():
    a = np.zeros((1000, 512, len(FIELDS)), dtype=np.float32)
    b = a.copy()
    valid = np.ones((1000, 512), dtype=bool)
    av = valid.copy()
    part = slice(CASES.index("turn_right") * REPEATS, (CASES.index("turn_right") + 1) * REPEATS)
    a[:, part, 2], b[:, part, 2] = -.5, -.5
    a[:, part, 5], b[:, part, 5] = -.4, -.3
    av[300:, part] = False
    a[300:, part, 5] = 1000
    result = paired_metric((a, av), (b, valid), "turn_right")
    assert result["paired_starts"] == 32
    assert result["mean_difference"] == pytest.approx(-.1, abs=1e-6)
    assert result["bootstrap_95_high"] < 0


def test_heading_and_stop_measure_the_intended_response():
    a = np.zeros((1000, 512, len(FIELDS)), dtype=np.float32)
    b = a.copy()
    valid = np.ones((1000, 512), dtype=bool)
    a[..., 6], b[..., 6] = .3, -.1
    a[..., 5], b[..., 5] = -.05, .2
    heading = paired_metric((a, valid), (b, valid), "heading_right")
    stop = paired_metric((a, valid), (b, valid), "stop_right")
    assert heading["metric"] == "mean_abs_heading_error" and heading["window"] == "14-20s"
    assert heading["mean_difference"] == pytest.approx(.2, abs=1e-6)
    assert stop["metric"] == "mean_abs_yaw_rate" and stop["window"] == "10-14s"
    assert stop["mean_difference"] == pytest.approx(-.15, abs=1e-6)


def test_no_survivors_is_missing_data_not_zero_error():
    trace = np.zeros((1000, 512, len(FIELDS)), dtype=np.float32)
    valid = np.zeros((1000, 512), dtype=bool)
    result = paired_metric((trace, valid), (trace, ~valid), "reverse_left")
    assert result["paired_starts"] == 0
    assert result["mean_difference"] is None


def test_q4_pairing_audit_records_material_changes_without_consuming_rng():
    from types import SimpleNamespace
    import torch
    from stage1_flat_retention import initial_state_hashes
    material = torch.ones(2, 4, 3)
    data = SimpleNamespace(root_state_w=torch.zeros(2, 13), joint_pos=torch.zeros(2, 12),
                           joint_vel=torch.zeros(2, 12))
    robot = SimpleNamespace(data=data, root_physx_view=SimpleNamespace(get_material_properties=lambda: material))
    base = SimpleNamespace(scene={"robot": robot}, episode_length_buf=torch.zeros(2, dtype=torch.long),
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(2, 3)))
    wrapped = SimpleNamespace(unwrapped=base)
    rng = torch.get_rng_state().clone()
    before = initial_state_hashes(wrapped)
    material[0, 0, 0] = .9
    after = initial_state_hashes(wrapped)
    assert [k for k in before if before[k] != after[k]] == ["materials"]
    assert torch.equal(rng, torch.get_rng_state())
