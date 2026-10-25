"""Ensure only explicitly registered terrain instances can use the bounded runner."""

from pathlib import Path
import sys
import math
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

from scripts.tools import source_family_runtime_smoke as runtime_smoke  # noqa: E402
from scripts.tools import run_l2_stair_multiseed_diagnostic as multiseed  # noqa: E402


def test_l2_ascending_is_the_only_new_l2_runtime_cell():
    assert runtime_smoke.CELL_IDS == (
        "slope_L1_uphill",
        "slope_L1_downhill",
        "multi_step_stairs_L1_ascending",
        "multi_step_stairs_L1_descending",
        "multi_step_stairs_L2_ascending",
    )

    with pytest.raises(ValueError, match="bounded runtime diagnostic allowlist"):
        runtime_smoke.select_cell("multi_step_stairs_L2_descending")


def test_l2_ascending_resolves_the_registered_geometry():
    _manifest, cell = runtime_smoke.select_cell("multi_step_stairs_L2_ascending")
    geometry = cell["resolved_parameters"]["geometry"]

    assert cell["episodes"] == 5
    assert geometry["step_count"] == 5
    assert geometry["step_height_m"] == pytest.approx(0.035)
    assert geometry["tread_depth_m"] == pytest.approx(0.5)


def test_terminal_phase_trigonometry_accepts_python_scalars():
    assert runtime_smoke._phase_sin_cos(math.pi / 2.0) == pytest.approx([1.0, 0.0])

    with pytest.raises(ValueError, match="finite scalar"):
        runtime_smoke._phase_sin_cos(float("nan"))


def test_termination_diagnostic_separates_configured_and_triggered_terms():
    class _Manager:
        active_terms = ["illegal_contact", "bad_orientation_2"]

        def get_term(self, name):
            return {
                "illegal_contact": torch.tensor([False]),
                "bad_orientation_2": torch.tensor([True]),
            }[name]

        def get_term_cfg(self, name):
            if name == "illegal_contact":
                return SimpleNamespace(
                    time_out=False,
                    params={
                        "threshold": 5.0,
                        "consecutive_steps": 3,
                        "sensor_cfg": SimpleNamespace(
                            name="contact_forces",
                            body_names=[
                                "TORSO", "FL_HIP", "FL_THIGH", "FL_SHANK",
                                "FR_HIP", "FR_THIGH", "FR_SHANK",
                                "HL_HIP", "HL_THIGH", "HL_SHANK",
                                "HR_HIP", "HR_THIGH", "HR_SHANK",
                            ],
                            body_ids=list(range(13)),
                        ),
                    },
                )
            return SimpleNamespace(time_out=False, params={})

    diagnostic = runtime_smoke._termination_diagnostic(
        SimpleNamespace(termination_manager=_Manager())
    )

    assert diagnostic["configured_termination_terms"] == [
        "illegal_contact", "bad_orientation_2"
    ]
    assert diagnostic["triggered_termination_terms"] == ["bad_orientation_2"]
    illegal = diagnostic["termination_term_configurations"]["illegal_contact"]
    assert illegal["sensor_name"] == "contact_forces"
    assert illegal["sensor_body_names"][:2] == ["TORSO", "FL_HIP"]
    assert illegal["sensor_body_ids"] == list(range(13))
    assert illegal["threshold_n"] == 5.0
    assert illegal["consecutive_steps"] == 3


def test_single_episode_runner_only_accepts_registered_paired_seed_indices():
    _manifest, cell = runtime_smoke.select_cell("multi_step_stairs_L2_ascending")

    assert runtime_smoke.resolve_registered_seed_pair(cell, None, None) == (10200, 20200)
    assert runtime_smoke.resolve_registered_seed_pair(cell, 10203, 20203) == (10203, 20203)
    with pytest.raises(ValueError, match="pair index"):
        runtime_smoke.resolve_registered_seed_pair(cell, 10203, 20202)
    with pytest.raises(ValueError, match="registered range"):
        runtime_smoke.resolve_registered_seed_pair(cell, 10205, 20205)


def test_multiseed_plan_pairs_only_the_five_registered_l2_seeds():
    _manifest, cell = runtime_smoke.select_cell("multi_step_stairs_L2_ascending")

    assert multiseed.registered_seed_pairs(cell) == [
        {"geometry_seed": 10200, "initial_state_seed": 20200},
        {"geometry_seed": 10201, "initial_state_seed": 20201},
        {"geometry_seed": 10202, "initial_state_seed": 20202},
        {"geometry_seed": 10203, "initial_state_seed": 20203},
        {"geometry_seed": 10204, "initial_state_seed": 20204},
    ]

    commands = [
        multiseed.build_episode_command(
            pair,
            output=ROOT / f"episode_{index}.json",
            device="cuda:0",
            heartbeat_steps=50,
        )
        for index, pair in enumerate(multiseed.registered_seed_pairs(cell), start=1)
    ]
    assert all(command[command.index("--cell-id") + 1] == "multi_step_stairs_L2_ascending" for command in commands)
    assert all("--headless" in command for command in commands)
    assert all("--output" in command for command in commands)


def test_multiseed_aggregate_is_descriptive_and_applies_no_threshold():
    aggregate = multiseed.aggregate_records([
        {"termination_reason": "fall", "completed_step_index": 0},
        {"termination_reason": "fall", "completed_step_index": 1},
        {"termination_reason": "fall", "completed_step_index": 0},
        {"termination_reason": "completion", "completed_step_index": 5},
        {"termination_reason": "body_contact_termination", "completed_step_index": 0},
    ])

    assert aggregate["episode_count"] == 5
    assert aggregate["termination_reason_counts"] == {
        "fall": 3,
        "completion": 1,
        "body_contact_termination": 1,
    }
    assert aggregate["completed_step_index_counts"] == {"0": 3, "1": 1, "5": 1}
    assert aggregate["qualification_claim"] is False
    assert "no promotion" in aggregate["interpretation"]

