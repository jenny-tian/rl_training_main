from types import SimpleNamespace

import numpy as np
import torch

from rl_training.motion_prior.manager import MotionPriorManager
from rl_training.motion_prior.retarget import foot_positions_from_lite3_joints, retarget_motion
from rl_training.motion_prior.rewards import motion_prior_joint_pos_exp


class Cfg:
    name = "robot"
    joint_ids = list(range(12))


def test_retarget_npz_manager_and_reward_pipeline(tmp_path):
    joints = np.tile(np.array([0.0, -0.65, 1.3] * 4, dtype=np.float32), (4, 1))
    feet = foot_positions_from_lite3_joints(joints)
    metadata = {
        "source_id": "synthetic-go2-walk",
        "source_format": "go2-telemetry",
        "retargeter": "lite3-analytic-ik-v1",
        "created_utc": "2026-08-04T00:00:00Z",
        "flat_only_split": "train",
    }
    path = tmp_path / "reference.npz"
    retarget_motion(
        time=np.arange(4, dtype=np.float32) / 50.0,
        foot_pos_b=feet,
        contact=np.ones((4, 4), dtype=np.float32),
        metadata=metadata,
    ).save(path)

    manager = MotionPriorManager(path)
    assert manager.metadata["source_id"] == "synthetic-go2-walk"

    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        episode_length_buf=torch.tensor([0]),
        common_step_counter=0,
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=torch.from_numpy(joints[:1])))},
    )
    reward = motion_prior_joint_pos_exp(
        env,
        Cfg(),
        reference_path=str(path),
        mode="on",
        prior_weight_value=1.0,
    )
    assert torch.allclose(reward, torch.ones(1))

    off_reward = motion_prior_joint_pos_exp(env, Cfg(), reference_path=None, mode="off")
    assert torch.equal(off_reward, torch.zeros(1))
