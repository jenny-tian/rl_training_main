"""A successful seed cannot hide another seed's Q4 or stair failure."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
from source_block_b4_protocol import Q4_LIMITS, candidate_gate, differences


def fixture():
    flat = dict(status="passed", completed_steps=3000, metrics=dict(Q4_LIMITS))
    block = dict(initial_state_hashes_by_column=["paired"] * 20,
                 columns=[dict(complete=23)] * 20)
    return flat, copy.deepcopy(block), copy.deepcopy(block)


def test_gate_keeps_exact_q4_boundaries_and_never_autoextends():
    result = candidate_gate(*fixture())
    assert result["eligible_for_discussion"]
    assert not result["automatic_extension"]
    for key, limit in Q4_LIMITS.items():
        flat, candidate, control = fixture()
        flat["metrics"][key] = limit + 1e-8
        assert f"Q4/{key}" in candidate_gate(flat, candidate, control)["reasons"]


def test_stair_absolute_floor_and_paired_control_tolerance():
    flat, candidate, control = fixture()
    candidate["columns"][4] = dict(complete=21)
    candidate["columns"][5] = dict(complete=21)
    assert candidate_gate(flat, candidate, control)["eligible_for_discussion"]
    candidate["columns"][4] = dict(complete=20)
    candidate["columns"][5] = dict(complete=20)
    reasons = candidate_gate(flat, candidate, control)["reasons"]
    assert "upstairs_below_21_of_32" in reasons
    assert "column_4_more_than_2_below_control" in reasons
    assert "column_5_more_than_2_below_control" in reasons


def test_unpaired_or_incomplete_evaluation_rejected():
    flat, candidate, control = fixture()
    candidate["initial_state_hashes_by_column"][3] = "different"
    with pytest.raises(ValueError, match="unpaired"):
        candidate_gate(flat, candidate, control)
    flat, candidate, control = fixture()
    flat["completed_steps"] = 2999
    with pytest.raises(ValueError, match="incomplete"):
        candidate_gate(flat, candidate, control)


def test_config_diff_detects_unintended_nested_changes():
    a = dict(rewards=dict(stand=dict(func="old", weight=-.2)), commands=[0, .5])
    b = copy.deepcopy(a)
    b["rewards"]["stand"]["func"] = "new"
    assert [d["path"] for d in differences(a, b)] == ["/rewards/stand/func"]
    b["commands"][1] = .8
    assert [d["path"] for d in differences(a, b)] == ["/rewards/stand/func", "/commands/1"]
