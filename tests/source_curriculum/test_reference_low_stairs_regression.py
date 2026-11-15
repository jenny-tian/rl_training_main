"""诊断的时间、坐标和能力保持边界，不启动仿真或训练。"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from reference_low_stairs_regression_analyze import first_time, yaw, lateral_components
from reference_low_stairs_regression_probe import heading_command, ramp_fraction
from reference_low_stairs_preservation import (capability_groups, joint_target_preservation_loss,
    split_heading_turn_samples)


def test_event_time_is_post_physics_and_absent_is_not_zero():
    assert first_time([False, True, True]) == .04
    assert first_time([False, False]) is None


def test_heading_feedback_wraps_and_clips_without_resetting_position():
    np.testing.assert_allclose(heading_command([.2, -.2, 2., -2., 2*np.pi+.2]), [-.1, .1, -.8, .8, -.1])
    np.testing.assert_allclose(heading_command([np.pi, -np.pi]), [-.8, -.8])


def test_world_lateral_speed_uses_rotation_and_body_side_slip():
    q = np.array([[1., 0, 0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)]])
    v = np.array([[.4, .2, .1], [.4, .2, .1]])
    np.testing.assert_allclose(yaw(q), [0, np.pi/2])
    np.testing.assert_allclose(lateral_components(q, v), [[0, .2, 0], [.4, 0, 0]], atol=1e-12)


def test_ramp_finishes_at_half_second_and_is_bounded():
    assert ramp_fraction(0) == .04
    assert ramp_fraction(24) == 1.
    assert ramp_fraction(100) == 1.
    with pytest.raises(ValueError):
        ramp_fraction(-1)


def test_groups_separate_recovery_and_both_directions():
    cmd = torch.tensor([[0., 0, 0], [0., 0, 0], [.4, 0, 0], [0, 0, .5], [0, 0, -.5], [.3, 0, .5]])
    tilt = torch.tensor([0., .3, 0, 0, 0, 0])
    assert capability_groups(cmd, tilt).tolist() == [0, 1, 2, 3, 4, -1]


def test_preservation_is_in_joint_radians_and_teacher_has_no_gradient():
    teacher = torch.zeros(2, 12, requires_grad=True)
    prediction = torch.zeros(2, 12); prediction[0, 0] = 2.; prediction[1, 1] = 1.
    prediction.requires_grad_()
    loss = joint_target_preservation_loss(prediction, teacher, torch.tensor([0, 4]))
    assert loss.item() == pytest.approx((.25/.1)**2/12)
    loss.backward()
    assert teacher.grad is None and prediction.grad[0, 0] > 0


def test_loss_is_equal_per_group_and_ignores_unprotected_motion():
    parent = torch.zeros(5, 12)
    action = torch.tensor([[1.]*12, [1.]*12, [1.]*12, [2.]*12, [1000.]*12])
    mixed = joint_target_preservation_loss(action, parent, torch.tensor([0, 0, 0, 4, -1]))
    equal = joint_target_preservation_loss(action[[0, 3]], parent[[0, 3]], torch.tensor([0, 4]))
    assert torch.equal(mixed, equal)


def test_empty_protected_group_has_zero_finite_gradient():
    x = torch.ones(3, 12, requires_grad=True)
    loss = joint_target_preservation_loss(x, torch.zeros_like(x), torch.full((3,), -1))
    loss.backward()
    assert loss == 0 and x.grad.eq(0).all()


@pytest.mark.parametrize('failure', ['nan', 'shape', 'group'])
def test_loss_rejects_invalid_anchor_data(failure):
    x = torch.zeros(2, 12); teacher = x.clone(); groups = torch.tensor([0, 1])
    if failure == 'nan': x[0, 0] = float('nan')
    if failure == 'shape': teacher = teacher[:, :11]
    if failure == 'group': groups[0] = 5
    with pytest.raises(ValueError):
        joint_target_preservation_loss(x, teacher, groups)


def test_sampling_changes_only_requested_pure_heading_turns():
    term = NS(vel_command_b=torch.zeros(7, 3), is_heading_env=torch.ones(7, dtype=torch.bool),
              is_standing_env=torch.tensor([True, False, False, False, False, False, False]))
    term.vel_command_b[1, 0] = .4
    before = term.vel_command_b.clone()
    ids = torch.arange(6)
    selected = split_heading_turn_samples(term, ids, torch.tensor([.6, .9, .6, .9, .1, .7]), torch.zeros(6))
    assert selected.tolist() == [2, 3, 5]
    assert term.vel_command_b[2, 2] == .2 and term.vel_command_b[3, 2] == -.2
    assert torch.equal(term.vel_command_b[[0, 1, 4, 6]], before[[0, 1, 4, 6]])
    assert term.is_heading_env.tolist() == [True, True, False, False, True, False, True]
    assert term.is_standing_env.tolist() == [True, False, False, False, False, False, False]


def test_sampling_rejects_duplicate_indices():
    with pytest.raises(ValueError):
        split_heading_turn_samples(None, torch.tensor([1, 1]), torch.zeros(2), torch.zeros(2))
