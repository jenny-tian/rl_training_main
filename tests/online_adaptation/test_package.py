import ast
import sys
import torch
from dataclasses import fields
from inspect import signature
from pathlib import Path
from typing import get_type_hints

import tomllib


ROOT = Path(__file__).resolve().parents[2]
ENDING_KEYWORDS = {
    "terminated",
    "truncated",
    "bootstrap_mask",
    "discount",
    "safety_mask",
    "curriculum_stage_id",
    "episode_id",
}


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
        terminated=torch.zeros(4, 1),
        truncated=torch.zeros(4, 1),
        bootstrap_mask=torch.ones(4, 1),
        discount=torch.full((4, 1), 0.99),
        safety_mask=torch.ones(4, 1),
        curriculum_stage_id=torch.zeros(4, 1, dtype=torch.int64),
        episode_id=torch.zeros(4, 1, dtype=torch.int64),
    )

    assert batch.history.shape == (4, 25, 49)
    assert [field.name for field in fields(TransitionBatch)] == [
        "history",
        "current_frame",
        "base_action",
        "residual_action",
        "reward",
        "next_history",
        "next_frame",
        "next_base_action",
        "terminated",
        "truncated",
        "bootstrap_mask",
        "discount",
        "safety_mask",
        "curriculum_stage_id",
        "episode_id",
        "requested_residual_action",
    ]
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
    assert [field.name for field in fields(RewardSignals)] == [
        "base_lin_vel_xy",
        "command_lin_vel_xy",
        "base_ang_vel_z",
        "command_ang_vel_z",
        "projected_gravity_xy",
        "base_lin_vel_z",
        "foot_slip",
        "foot_impact",
        "joint_power",
        "nonfoot_contact",
        "foot_height",
        "foot_contact",
        "foot_edge_offset",
    ]


def test_finish_step_signature_uses_explicit_tensor_annotations():
    from rl_training.online_adaptation.agent import OnlineAdaptationAgent

    method_signature = signature(OnlineAdaptationAgent.finish_step)
    type_hints = get_type_hints(OnlineAdaptationAgent.finish_step)
    tensor_parameters = [
        "next_frame",
        "next_base_action",
        "reward",
        "terminated",
        "truncated",
        "bootstrap_mask",
        "discount",
        "safety_mask",
        "curriculum_stage_id",
        "episode_id",
    ]

    assert list(method_signature.parameters) == ["self", *tensor_parameters, "adaptation_enabled"]
    assert {name: type_hints.get(name) for name in tensor_parameters} == {
        name: torch.Tensor for name in tensor_parameters
    }
    assert type_hints.get("adaptation_enabled") is bool
    assert method_signature.parameters["adaptation_enabled"].default is False
    assert type_hints["return"] is type(None)


def test_all_finish_step_callers_supply_explicit_ending_keywords():
    missing_by_caller = []
    for source_root in (ROOT / "source", ROOT / "scripts", ROOT / "tests"):
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "finish_step"
                ):
                    continue
                supplied = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
                missing = ENDING_KEYWORDS - supplied
                if missing:
                    missing_by_caller.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} missing {sorted(missing)}"
                    )

    assert missing_by_caller == []
