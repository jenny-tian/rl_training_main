from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_training.motion_prior.mdp import motion_imitation_time_out
from rl_training.motion_prior.periodic import build_periodic_reference
from rl_training.motion_prior.schema import MotionTrajectory


METADATA = {
    "source_id": "synthetic",
    "source_format": "unit-test",
    "retargeter": "test",
    "created_utc": "2026-08-04T00:00:00Z",
    "flat_only_split": "train",
}


def source_trajectory(frames=10):
    phase = np.arange(frames, dtype=np.float32)[:, None]
    return MotionTrajectory(
        time=np.arange(frames, dtype=np.float32) * 0.02,
        joint_pos=np.sin(phase * 0.5) * np.ones((frames, 12), dtype=np.float32),
        foot_pos_b=np.zeros((frames, 4, 3), dtype=np.float32),
        contact=np.ones((frames, 4), dtype=np.float32),
        metadata=METADATA,
    )


def test_periodic_reference_is_closed_and_state_complete():
    result = build_periodic_reference(
        source_trajectory(), frame_start=2, unique_frame_count=5, forward_speed=0.6
    )

    assert len(result.time) == 6
    assert result.time[-1] == pytest.approx(0.1)
    assert np.array_equal(result.joint_pos[0], result.joint_pos[-1])
    assert np.array_equal(result.joint_vel[0], result.joint_vel[-1])
    assert np.array_equal(result.contact[0], result.contact[-1])
    assert np.allclose(result.base_lin_vel_b, [0.6, 0.0, 0.0])
    assert np.allclose(result.projected_gravity_b, [0.0, 0.0, -1.0])


def test_motion_imitation_timeout_uses_cubic_curriculum():
    env = SimpleNamespace(common_step_counter=0, episode_length_buf=torch.tensor([19, 20, 21]))
    assert torch.equal(
        motion_imitation_time_out(env, start_steps=20, end_steps=600, curriculum_steps=100),
        torch.tensor([False, True, True]),
    )
    env.common_step_counter = 100
    env.episode_length_buf = torch.tensor([599, 600])
    assert torch.equal(
        motion_imitation_time_out(env, start_steps=20, end_steps=600, curriculum_steps=100),
        torch.tensor([False, True]),
    )
