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

    # Runtime-observed Isaac soft limits with the flat task's 0.9 factor. The
    # SafetyFilter reserves 0.02 rad before projecting a physical target.
    soft_lower = torch.tensor(
        (-0.4706999362, -2.5207998753, 0.6373999119) * 4,
        dtype=action.dtype,
    )
    soft_upper = torch.tensor(
        (0.4706999362, 0.1647999287, 2.6786000729) * 4,
        dtype=action.dtype,
    )
    safety_margin = 0.02
    assert torch.all(targets > soft_lower + safety_margin)
    assert torch.all(targets < soft_upper - safety_margin)


def test_action_clip_pattern_map_has_one_entry_per_joint_kind():
    assert LITE3_SAFE_ACTION_CLIP_BY_PATTERN == {
        ".*_HipX_joint": (-3.60, 3.60),
        ".*_HipY_joint": (-7.40, 3.17),
        ".*_Knee_joint": (-2.57, 5.43),
    }


@pytest.mark.parametrize("helper", [clip_lite3_policy_action, physical_targets_from_action])
def test_helpers_reject_an_invalid_action_last_dimension(helper):
    with pytest.raises(ValueError, match="12"):
        helper(torch.zeros(2, 11))


def test_cached_bounds_created_during_inference_allow_later_policy_gradients():
    from rl_training.lite3_joint_conventions import _clip_bounds, clip_lite3_policy_action
    _clip_bounds.cache_clear()
    with torch.inference_mode():
        clip_lite3_policy_action(torch.zeros(1, 12))
    action = torch.zeros(1, 12, requires_grad=True)
    clip_lite3_policy_action(action).sum().backward()
    assert torch.equal(action.grad, torch.ones_like(action))
    assert _clip_bounds.cache_info().hits >= 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cached_clipping_exactly_matches_uncached_limits_for_noncontiguous_actions(dtype):
    from rl_training.lite3_joint_conventions import LITE3_SAFE_ACTION_LOWER, LITE3_SAFE_ACTION_UPPER, clip_lite3_policy_action
    action = torch.linspace(-10, 10, 48, dtype=dtype).reshape(12, 4).T
    expected = torch.maximum(torch.minimum(action, torch.tensor(LITE3_SAFE_ACTION_UPPER,dtype=dtype)),
                             torch.tensor(LITE3_SAFE_ACTION_LOWER,dtype=dtype))
    assert torch.equal(clip_lite3_policy_action(action), expected)
