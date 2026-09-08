import numpy as np
import pytest
import torch

from rl_training.motion_prior.manager import MotionPriorManager, prior_weight
from rl_training.motion_prior.schema import MotionTrajectory


METADATA = {
    "source_id": "synthetic",
    "source_format": "unit-test",
    "retargeter": "test",
    "created_utc": "2026-08-04T00:00:00Z",
    "flat_only_split": "train",
}


def make_trajectory(frames=3, dt=0.1):
    return MotionTrajectory(
        time=np.arange(frames, dtype=np.float32) * dt,
        joint_pos=np.arange(frames * 12, dtype=np.float32).reshape(frames, 12),
        foot_pos_b=np.zeros((frames, 4, 3), dtype=np.float32),
        contact=np.ones((frames, 4), dtype=np.float32),
        command=np.zeros((frames, 3), dtype=np.float32),
        metadata=METADATA,
    )


def test_manager_interpolates_and_loops_reference(tmp_path):
    trajectory = make_trajectory()
    trajectory.save(tmp_path / "ref.npz")
    manager = MotionPriorManager(tmp_path / "ref.npz", device="cpu")

    sample = manager.sample(torch.tensor([0.05, 0.35]))

    assert sample.joint_pos.shape == (2, 12)
    assert torch.allclose(sample.joint_pos[0], torch.arange(12, dtype=torch.float32) + 6.0)
    assert torch.allclose(sample.joint_pos[1], torch.arange(12, dtype=torch.float32) + 18.0)
    assert sample.metadata["source_id"] == "synthetic"


@pytest.mark.parametrize("mode,expected", [("on", 2.0), ("off", 0.0)])
def test_prior_schedule_modes(mode, expected):
    assert prior_weight(mode, 10, 100, 2.0) == pytest.approx(expected)


def test_decay_schedule_reaches_zero_and_rejects_unknown_mode():
    assert prior_weight("decay", 100, 100, 2.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="mode"):
        prior_weight("bad", 0, 100, 1.0)
