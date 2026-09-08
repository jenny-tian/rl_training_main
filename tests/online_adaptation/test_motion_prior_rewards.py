from types import SimpleNamespace

import numpy as np
import torch

from rl_training.motion_prior.schema import MotionTrajectory
from rl_training.motion_prior.rewards import (
    _root_quat_per_body,
    motion_prior_contact_phase,
    motion_prior_foot_pos_exp,
    motion_prior_joint_pos_exp,
    motion_prior_joint_smoothness,
    motion_prior_swing_clearance,
    motion_prior_symmetry_exp,
    motion_reference_base_ang_vel_exp,
    motion_reference_base_lin_vel_exp,
    motion_reference_joint_pos_exp,
    motion_reference_projected_gravity_exp,
)


class Cfg:
    def __init__(self, name, ids):
        self.name = name
        self.joint_ids = list(ids) if name == "robot" else []
        self.body_ids = list(ids) if name != "robot" else []


def make_env(path, *, swing=False):
    joints = torch.tensor([[0.0, -0.65, 1.3] * 4])
    feet = torch.tensor([[[0.20, 0.105, -0.325], [0.20, -0.105, -0.325], [-0.20, 0.105, -0.325], [-0.20, -0.105, -0.325]]])
    if swing:
        feet[:, 0, 2] = -0.20
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=joints,
            joint_vel=torch.zeros_like(joints),
            body_pos_b=feet,
            root_lin_vel_b=torch.tensor([[0.6, 0.0, 0.0]]),
            root_ang_vel_b=torch.zeros((1, 3)),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        )
    )
    sensor = SimpleNamespace(data=SimpleNamespace(contact=torch.tensor([[0.0, 1.0, 1.0, 0.0]] if swing else [[1.0] * 4])))
    return SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        episode_length_buf=torch.tensor([0]),
        common_step_counter=0,
        scene={"robot": asset, "contact_forces": sensor},
        _motion_prior_managers={},
    )


def write_reference(path, *, swing=False, state_complete=False):
    feet = np.array([[[0.20, 0.105, -0.325], [0.20, -0.105, -0.325], [-0.20, 0.105, -0.325], [-0.20, -0.105, -0.325]]], dtype=np.float32)
    if swing:
        feet[:, 0, 2] = -0.20
    optional = {}
    if state_complete:
        optional = {
            "joint_vel": np.zeros((2, 12), dtype=np.float32),
            "base_lin_vel_b": np.tile([0.6, 0.0, 0.0], (2, 1)).astype(np.float32),
            "base_ang_vel_b": np.zeros((2, 3), dtype=np.float32),
            "projected_gravity_b": np.tile([0.0, 0.0, -1.0], (2, 1)).astype(np.float32),
        }
    MotionTrajectory(
        time=np.array([0.0, 0.1], dtype=np.float32),
        joint_pos=np.tile(np.array([[0.0, -0.65, 1.3] * 4], dtype=np.float32), (2, 1)),
        foot_pos_b=np.tile(feet, (2, 1, 1)),
        contact=np.tile(np.array([[0.0, 1.0, 1.0, 0.0]] if swing else [[1.0] * 4], dtype=np.float32), (2, 1)),
        **optional,
        metadata={
            "source_id": "test",
            "source_format": "synthetic",
            "retargeter": "test",
            "created_utc": "2026-08-04T00:00:00Z",
            "flat_only_split": "train",
        },
    ).save(path)


def params(path):
    return {"reference_path": str(path), "mode": "on", "prior_weight_value": 1.0, "decay_steps": 100}


def test_reference_rewards_are_maximal_at_reference(tmp_path):
    path = tmp_path / "ref.npz"
    write_reference(path)
    env = make_env(path)
    assert torch.allclose(motion_prior_joint_pos_exp(env, Cfg("robot", range(12)), **params(path)), torch.ones(1))
    assert torch.allclose(motion_prior_foot_pos_exp(env, Cfg("robot", range(4)), **params(path)), torch.ones(1))
    assert torch.allclose(motion_prior_symmetry_exp(env, Cfg("robot", range(4)), **params(path)), torch.ones(1))
    assert torch.allclose(motion_prior_joint_smoothness(env, Cfg("robot", range(12)), **params(path)), torch.ones(1))


def test_contact_clearance_and_prior_off_are_supported(tmp_path):
    path = tmp_path / "swing.npz"
    write_reference(path, swing=True)
    env = make_env(path, swing=True)
    assert motion_prior_contact_phase(env, Cfg("contact_forces", range(4)), **params(path))[0] > 0.9
    assert motion_prior_swing_clearance(env, Cfg("robot", range(4)), **params(path))[0] > 0.9
    assert motion_prior_joint_pos_exp(env, Cfg("robot", range(12)), reference_path=None, mode="off")[0] == 0.0


def test_relative_swing_clearance_ignores_morphology_height_offset(tmp_path):
    path = tmp_path / "relative_swing.npz"
    write_reference(path, swing=True)
    env = make_env(path, swing=True)
    env.scene["robot"].data.body_pos_b[..., 2] -= 0.10

    absolute = motion_prior_swing_clearance(
        env, Cfg("robot", range(4)), std=0.04, **params(path)
    )
    relative = motion_prior_swing_clearance(
        env,
        Cfg("robot", range(4)),
        std=0.04,
        relative_height=True,
        **params(path),
    )

    assert relative.item() > 0.9
    assert relative.item() > absolute.item()


def test_hind_leg_weights_preserve_scale_and_penalize_hind_error_more(tmp_path):
    path = tmp_path / "weighted.npz"
    write_reference(path)
    env = make_env(path)
    env.scene["robot"].data.body_pos_b[:, 2:, 2] -= 0.08
    base = motion_prior_foot_pos_exp(
        env, Cfg("robot", range(4)), std=0.08, **params(path)
    )
    weighted = motion_prior_foot_pos_exp(
        env,
        Cfg("robot", range(4)),
        std=0.08,
        foot_weights=(1.0, 1.0, 2.0, 2.0),
        **params(path),
    )
    assert weighted.item() < base.item()
    env.scene["robot"].data.body_pos_b[:, 2:, 2] += 0.08
    assert torch.allclose(
        motion_prior_foot_pos_exp(
            env,
            Cfg("robot", range(4)),
            std=0.08,
            foot_weights=(1.0, 1.0, 2.0, 2.0),
            **params(path),
        ),
        torch.ones(1),
    )


def test_root_quaternion_expands_over_feet_without_changing_quaternion_width():
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    relative_feet = torch.zeros((2, 4, 3))

    expanded = _root_quat_per_body(root_quat, relative_feet)

    assert expanded.shape == (2, 4, 4)
    assert torch.equal(expanded[:, 0], root_quat)


def test_motion_reimagined_tracking_terms_are_maximal_at_reference(tmp_path):
    path = tmp_path / "state-complete.npz"
    write_reference(path, state_complete=True)
    env = make_env(path)
    cfg = Cfg("robot", range(12))

    assert torch.allclose(motion_reference_base_lin_vel_exp(env, cfg, **params(path)), torch.ones(1))
    assert torch.allclose(motion_reference_base_ang_vel_exp(env, cfg, **params(path)), torch.ones(1))
    assert torch.allclose(motion_reference_joint_pos_exp(env, cfg, **params(path)), torch.ones(1))
    assert torch.allclose(motion_reference_projected_gravity_exp(env, cfg, **params(path)), torch.ones(1))

    env.scene["robot"].data.joint_pos[:, -1] += 0.5
    expected = torch.exp(torch.tensor(-2.0 * 0.25))
    assert torch.allclose(motion_reference_joint_pos_exp(env, cfg, **params(path)), expected[None])
