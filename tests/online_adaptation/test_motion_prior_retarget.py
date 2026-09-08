import numpy as np
import pytest

from rl_training.motion_prior.retarget import (
    DEFAULT_GO2_JOINT_NAMES,
    Lite3RetargetConfig,
    foot_positions_from_lite3_joints,
    lite3_joints_from_foot_positions,
    retarget_joint_trajectory,
    retarget_motion,
)


def source_metadata():
    return {
        "source_id": "synthetic-go2",
        "source_format": "go2-telemetry",
        "retargeter": "unit-test",
        "created_utc": "2026-08-04T00:00:00Z",
        "flat_only_split": "train",
    }


def test_source_joint_mapping_applies_sign_scale_and_offset():
    source = np.zeros((2, 12), dtype=np.float32)
    source[:, 1] = 0.5

    result = retarget_joint_trajectory(source, DEFAULT_GO2_JOINT_NAMES, Lite3RetargetConfig())

    assert result.shape == (2, 12)
    assert np.allclose(result[:, 1], -0.65 - 0.5)
    assert np.allclose(result[:, 2], 1.3)


def test_foot_space_retarget_round_trips_lite3_kinematics_and_contact():
    joints = np.tile(np.array([0.08, -0.65, 1.3] * 4, dtype=np.float32), (5, 1))
    feet = foot_positions_from_lite3_joints(joints)
    contact = np.ones((5, 4), dtype=np.float32)

    result = retarget_motion(
        time=np.arange(5, dtype=np.float32) / 50.0,
        foot_pos_b=feet,
        contact=contact,
        metadata=source_metadata(),
    )

    assert result.joint_pos.shape == (5, 12)
    assert np.allclose(result.foot_pos_b, feet, atol=2e-4)
    assert np.array_equal(result.contact, contact)


def test_joint_space_retarget_generates_foot_positions():
    joints = np.tile(np.array([0.0, -0.65, 1.3] * 4, dtype=np.float32), (4, 1))
    result = retarget_motion(
        time=np.arange(4, dtype=np.float32) / 50.0,
        joint_pos=joints,
        source_joint_names=DEFAULT_GO2_JOINT_NAMES,
        metadata=source_metadata(),
    )
    assert result.foot_pos_b.shape == (4, 4, 3)
    assert np.isfinite(result.foot_pos_b).all()
    assert result.contact.shape == (4, 4)


def test_retarget_joint_limit_projection_is_explicit():
    joints = np.tile(np.array([0.0, -0.65, 1.3] * 4, dtype=np.float32), (3, 1))
    joints[1, 0] = 1.0
    result = retarget_motion(
        time=np.arange(3, dtype=np.float32) / 50.0,
        joint_pos=joints,
        source_joint_names=DEFAULT_GO2_JOINT_NAMES,
        metadata=source_metadata(),
        joint_limit_mode="clip",
    )

    assert result.joint_pos[1, 0] == pytest.approx(0.523, abs=1e-6)
    assert result.metadata["joint_limit_mode"] == "clip"
    assert result.metadata["joint_limit_violation_frame_ratio"] == pytest.approx(1.0 / 3.0)


def test_ik_rejects_unreachable_foot_positions():
    feet = foot_positions_from_lite3_joints(np.tile(np.array([0.0, -0.65, 1.3] * 4), (2, 1)))
    feet[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="unreachable"):
        lite3_joints_from_foot_positions(feet)


def test_retarget_motion_requires_a_joint_or_foot_input():
    with pytest.raises(ValueError, match="joint_pos or foot_pos_b"):
        retarget_motion(
            time=np.array([0.0, 0.1]),
            metadata=source_metadata(),
        )
