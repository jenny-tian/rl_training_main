import pytest
import torch

from rl_training.lite3_joint_conventions import (
    LITE3_POLICY_JOINT_NAMES,
    LITE3_SAFE_ACTION_CLIP_BY_PATTERN,
    LITE3_SAFE_ACTION_LOWER,
    LITE3_SAFE_ACTION_UPPER,
    clip_lite3_policy_action,
    physical_targets_from_action,
)


def test_policy_joint_names_are_leg_major():
    assert LITE3_POLICY_JOINT_NAMES == (
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


def test_extreme_actions_clip_to_targets_inside_soft_joint_bounds():
    action = torch.tensor([[-100.0] * 12, [100.0] * 12], dtype=torch.float64)

    clipped = clip_lite3_policy_action(action)
    targets = physical_targets_from_action(clipped)

    lower = torch.tensor(LITE3_SAFE_ACTION_LOWER, dtype=action.dtype)
    upper = torch.tensor(LITE3_SAFE_ACTION_UPPER, dtype=action.dtype)
    assert torch.equal(clipped[0], lower)
    assert torch.equal(clipped[1], upper)
    assert clipped.dtype == action.dtype
    assert clipped.device == action.device
    assert targets.dtype == action.dtype
    assert targets.device == action.device

    # Runtime-observed Isaac soft limits. The SafetyFilter reserves 0.02 rad
    # from each limit before projecting an action into a physical target.
    soft_lower = torch.tensor(
        (-0.5177699327, -2.6550798416, 0.5353399515) * 4,
        dtype=action.dtype,
    )
    soft_upper = torch.tensor(
        (0.5177699327, 0.2990798950, 2.7806601524) * 4,
        dtype=action.dtype,
    )
    safety_margin = 0.02
    assert torch.all(targets > soft_lower + safety_margin)
    assert torch.all(targets < soft_upper - safety_margin)


def test_action_clip_pattern_map_has_one_entry_per_joint_kind():
    assert LITE3_SAFE_ACTION_CLIP_BY_PATTERN == {
        ".*_HipX_joint": (-3.97, 3.97),
        ".*_HipY_joint": (-7.93, 3.70),
        ".*_Knee_joint": (-2.96, 5.83),
    }


@pytest.mark.parametrize("helper", [clip_lite3_policy_action, physical_targets_from_action])
def test_helpers_reject_an_invalid_action_last_dimension(helper):
    with pytest.raises(ValueError, match="12"):
        helper(torch.zeros(2, 11))
