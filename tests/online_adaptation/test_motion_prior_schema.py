import json

import numpy as np
import pytest

from rl_training.motion_prior.schema import MotionTrajectory


REQUIRED_METADATA = {
    "source_id": "synthetic-go1",
    "source_format": "unit-test",
    "retargeter": "test",
    "created_utc": "2026-08-04T00:00:00Z",
    "flat_only_split": "train",
}


def make_trajectory(frames=3):
    return MotionTrajectory(
        time=np.arange(frames, dtype=np.float64) * 0.1,
        joint_pos=np.arange(frames * 12, dtype=np.float32).reshape(frames, 12),
        foot_pos_b=np.zeros((frames, 4, 3), dtype=np.float32),
        contact=np.ones((frames, 4), dtype=np.float32),
        command=np.zeros((frames, 3), dtype=np.float32),
        metadata=REQUIRED_METADATA,
    )


def test_round_trip_preserves_reference_arrays_and_metadata(tmp_path):
    trajectory = make_trajectory()
    trajectory = MotionTrajectory(
        **{
            **trajectory.__dict__,
            "joint_vel": np.ones_like(trajectory.joint_pos),
            "base_lin_vel_b": np.ones((3, 3), dtype=np.float32),
            "base_ang_vel_b": np.zeros((3, 3), dtype=np.float32),
            "projected_gravity_b": np.tile([0.0, 0.0, -1.0], (3, 1)),
        }
    )
    path = tmp_path / "walk.npz"

    trajectory.save(path)
    loaded = MotionTrajectory.load(path)

    assert np.array_equal(loaded.joint_pos, trajectory.joint_pos)
    assert np.array_equal(loaded.foot_pos_b, trajectory.foot_pos_b)
    assert np.array_equal(loaded.contact, trajectory.contact)
    assert np.array_equal(loaded.joint_vel, trajectory.joint_vel)
    assert np.array_equal(loaded.base_lin_vel_b, trajectory.base_lin_vel_b)
    assert np.array_equal(loaded.projected_gravity_b, trajectory.projected_gravity_b)
    assert loaded.metadata["source_id"] == "synthetic-go1"


def test_schema_rejects_bad_shape_or_missing_provenance():
    bad = make_trajectory()
    bad = MotionTrajectory(
        time=bad.time,
        joint_pos=np.zeros((3, 11)),
        foot_pos_b=bad.foot_pos_b,
        contact=bad.contact,
        command=bad.command,
        metadata={},
    )

    with pytest.raises(ValueError, match="joint_pos"):
        bad.validate()

    missing_metadata = make_trajectory()
    missing_metadata = MotionTrajectory(
        time=missing_metadata.time,
        joint_pos=missing_metadata.joint_pos,
        foot_pos_b=missing_metadata.foot_pos_b,
        contact=missing_metadata.contact,
        command=missing_metadata.command,
        metadata={"source_id": "only-id"},
    )
    with pytest.raises(ValueError, match="metadata"):
        missing_metadata.validate()


def test_schema_rejects_non_monotonic_time_and_object_metadata(tmp_path):
    trajectory = make_trajectory()
    invalid_time = MotionTrajectory(
        time=np.array([0.0, 0.2, 0.1]),
        joint_pos=trajectory.joint_pos,
        foot_pos_b=trajectory.foot_pos_b,
        contact=trajectory.contact,
        command=trajectory.command,
        metadata=trajectory.metadata,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        invalid_time.validate()

    trajectory.save(tmp_path / "valid.npz")
    with np.load(tmp_path / "valid.npz", allow_pickle=False) as data:
        assert json.loads(str(data["metadata"]))["source_format"] == "unit-test"
