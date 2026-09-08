import numpy as np

from rl_training.motion_prior.gait_quality import derive_world_contact, score_gait
from rl_training.motion_prior.retarget import (
    LITE3_JOINT_LIMITS,
    foot_positions_from_lite3_joints,
    foot_reachability_mask,
    lite3_joints_from_foot_positions,
    project_joint_limits,
)


def _diagonal_clip(frame_count=101):
    time = np.arange(frame_count, dtype=np.float32) * 0.1
    feet = np.zeros((frame_count, 4, 3), dtype=np.float32)
    feet[..., 2] = -0.22
    contact = np.zeros((frame_count, 4), dtype=bool)
    contact[np.arange(frame_count) % 2 == 0, 0] = True
    contact[np.arange(frame_count) % 2 == 0, 3] = True
    contact[np.arange(frame_count) % 2 == 1, 1] = True
    contact[np.arange(frame_count) % 2 == 1, 2] = True
    feet[::2, 0, 2] = -0.12
    feet[1::2, 1, 2] = -0.12
    return time, feet, contact


def test_score_gait_recognizes_alternating_diagonal_contacts():
    time, feet, contact = _diagonal_clip()
    quality = score_gait(time, feet, contact=contact)

    assert quality.two_contact_ratio == 1.0
    assert quality.diagonal_contact_ratio == 1.0
    assert quality.all_contact_ratio == 0.0
    assert quality.transition_rate > 0.1
    assert quality.foot_swing_amplitude_m > 0.02
    assert quality.accepted


def test_score_gait_rejects_standing_all_contact_clip():
    time = np.arange(101, dtype=np.float32) * 0.1
    feet = np.zeros((101, 4, 3), dtype=np.float32)
    feet[..., 2] = -0.22
    quality = score_gait(time, feet, contact=np.ones((101, 4), dtype=bool))

    assert quality.all_contact_ratio == 1.0
    assert quality.two_contact_ratio == 0.0
    assert not quality.accepted


def test_world_contact_uses_stationary_world_feet_not_body_relative_speed():
    time = np.arange(21, dtype=np.float32) * 0.1
    world_feet = np.zeros((21, 4, 3), dtype=np.float32)
    world_feet[:, :, :2] = np.array(((0.2, 0.1), (0.2, -0.1), (-0.2, 0.1), (-0.2, -0.1)))
    world_feet[6:15, 0, 0] += np.linspace(0.0, 0.3, 9)
    world_feet[6:15, 0, 2] += 0.08

    contact = derive_world_contact(time, world_feet)

    assert np.all(contact[:, 1:])
    assert not np.any(contact[7:14, 0])


def test_joint_limit_projection_is_explicit_and_auditable():
    joints = np.tile(np.array([0.0, -0.65, 1.3] * 4, dtype=np.float32), (3, 1))
    joints[1, 0] = 1.0

    reported, violations = project_joint_limits(joints, mode="report")
    clipped, clipped_violations = project_joint_limits(joints, mode="clip")

    assert reported[1, 0] == 1.0
    assert violations.tolist() == [False, True, False]
    assert clipped[1, 0] == LITE3_JOINT_LIMITS[0, 1]
    assert np.array_equal(clipped_violations, violations)


def test_foot_reachability_mask_and_ik_screening_mode():
    valid = foot_positions_from_lite3_joints(
        np.tile(np.array([0.0, -0.65, 1.3] * 4, dtype=np.float32), (2, 1))
    )
    invalid = valid.copy()
    invalid[1, 0, 0] = 2.0

    assert foot_reachability_mask(invalid).tolist() == [True, False]
    screened = lite3_joints_from_foot_positions(invalid, allow_unreachable=True)
    assert screened.shape == (2, 12)
