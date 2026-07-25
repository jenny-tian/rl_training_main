import sys
import torch
from dataclasses import fields
from pathlib import Path

import tomllib


def test_extension_registers_root_and_ui_modules():
    extension_toml = Path("source/rl_training/config/extension.toml")
    configured_modules = tomllib.loads(extension_toml.read_text())["python"]["module"]

    assert [module["name"] for module in configured_modules] == [
        "rl_training",
        "rl_training.ui_extension_example",
    ]


def test_online_adaptation_import_does_not_load_pxr():
    import rl_training.online_adaptation.types  # noqa: F401

    assert "pxr" not in sys.modules


def test_transition_batch_contract():
    from rl_training.online_adaptation.types import RewardSignals, SafetyState, TransitionBatch

    batch = TransitionBatch(
        history=torch.zeros(4, 25, 49),
        current_frame=torch.zeros(4, 49),
        base_action=torch.zeros(4, 12),
        residual_action=torch.zeros(4, 12),
        reward=torch.zeros(4, 1),
        next_history=torch.zeros(4, 25, 49),
        next_frame=torch.zeros(4, 49),
        next_base_action=torch.zeros(4, 12),
        done=torch.zeros(4, 1),
    )

    assert batch.history.shape == (4, 25, 49)
    assert [field.name for field in fields(SafetyState)] == [
        "projected_gravity_xy",
        "base_height",
        "joint_pos",
        "joint_vel",
        "joint_lower",
        "joint_upper",
        "default_joint_pos",
        "action_scale",
        "nonfoot_contact",
        "control_timed_out",
    ]
    assert len(fields(RewardSignals)) == 10
