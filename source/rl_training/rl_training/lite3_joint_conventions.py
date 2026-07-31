"""Canonical Lite3 policy joint order and safe normalized-action limits."""

from __future__ import annotations

import torch


LITE3_POLICY_JOINT_NAMES = (
    "FL_HipX_joint",
    "FL_HipY_joint",
    "FL_Knee_joint",
    "FR_HipX_joint",
    "FR_HipY_joint",
    "FR_Knee_joint",
    "HL_HipX_joint",
    "HL_HipY_joint",
    "HL_Knee_joint",
    "HR_HipX_joint",
    "HR_HipY_joint",
    "HR_Knee_joint",
)

LITE3_SAFE_ACTION_LOWER = (-3.97, -7.93, -2.96) * 4
LITE3_SAFE_ACTION_UPPER = (3.97, 3.70, 5.83) * 4
LITE3_SAFE_ACTION_CLIP_BY_PATTERN = {
    ".*_HipX_joint": (-3.97, 3.97),
    ".*_HipY_joint": (-7.93, 3.70),
    ".*_Knee_joint": (-2.96, 5.83),
}

_LITE3_DEFAULT_JOINT_POS = (0.0, -0.65, 1.3) * 4
_LITE3_ACTION_SCALE = (0.125, 0.25, 0.25) * 4


def _validate_action_shape(action: torch.Tensor) -> None:
    if action.ndim == 0 or action.shape[-1] != len(LITE3_POLICY_JOINT_NAMES):
        raise ValueError(
            "expected Lite3 policy action with last dimension "
            f"{len(LITE3_POLICY_JOINT_NAMES)}, got {tuple(action.shape)}"
        )


def clip_lite3_policy_action(action: torch.Tensor) -> torch.Tensor:
    """Clip a Lite3 PPO action in canonical leg-major joint order."""
    _validate_action_shape(action)
    lower = torch.tensor(LITE3_SAFE_ACTION_LOWER, device=action.device, dtype=action.dtype)
    upper = torch.tensor(LITE3_SAFE_ACTION_UPPER, device=action.device, dtype=action.dtype)
    return torch.maximum(torch.minimum(action, upper), lower)


def physical_targets_from_action(action: torch.Tensor) -> torch.Tensor:
    """Convert a canonical normalized Lite3 action into joint targets."""
    _validate_action_shape(action)
    default_joint_pos = torch.tensor(
        _LITE3_DEFAULT_JOINT_POS, device=action.device, dtype=action.dtype
    )
    action_scale = torch.tensor(_LITE3_ACTION_SCALE, device=action.device, dtype=action.dtype)
    return default_joint_pos + action_scale * action
