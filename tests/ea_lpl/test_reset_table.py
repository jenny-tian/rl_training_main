import math
from types import SimpleNamespace

import numpy as np
import torch

from rl_training.ea_lpl.reset_table import DEFAULT_RESET_RANGES, initial_state_offsets, reset_root_state_from_table


class FakeAsset:
    def __init__(self, n):
        state = torch.zeros(n, 13)
        state[:, 2] = 0.375
        state[:, 3] = 1.0
        self.data = SimpleNamespace(default_root_state=state)
        self.pose = self.velocity = self.ids = None

    def write_root_pose_to_sim(self, pose, env_ids):
        self.pose, self.ids = pose.clone(), env_ids.clone()

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.velocity = velocity.clone()


class FakeScene:
    def __init__(self, asset, origins):
        self.asset, self.env_origins = asset, origins

    def __getitem__(self, name):
        assert name == "robot"
        return self.asset


def fake_env(n):
    origins = torch.stack((torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.zeros(n)), dim=1)
    return SimpleNamespace(num_envs=n, scene=FakeScene(FakeAsset(n), origins))


def test_offsets_are_deterministic_bounded_and_keyed():
    a = initial_state_offsets(20261106, 1, list(range(100)), [0] * 100)
    b = initial_state_offsets(20261106, 1, list(range(100)), [0] * 100)
    c = initial_state_offsets(20261106, 2, list(range(100)), [0] * 100)
    assert a.shape == (100, 3) and np.array_equal(a, b) and not np.array_equal(a, c)
    for column, name in enumerate(("x", "y", "yaw")):
        low, high = DEFAULT_RESET_RANGES[name]
        assert low <= a[:, column].min() and a[:, column].max() <= high
    assert len({tuple(row) for row in a}) == 100


def test_zero_ranges_reproduce_nominal_reset():
    env = fake_env(4)
    zero = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset_root_state_from_table(env, torch.arange(4), 1, ranges=zero)
    asset = env.scene.asset
    expected = asset.data.default_root_state[:, :3] + env.scene.env_origins
    assert torch.equal(asset.pose[:, :3], expected)
    assert torch.equal(asset.pose[:, 3:7], asset.data.default_root_state[:, 3:7])
    assert torch.equal(asset.velocity, torch.zeros(4, 6))


def test_counters_slots_and_yaw_quaternion():
    env = fake_env(4)
    ranges = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.1, 0.1)}
    reset_root_state_from_table(env, torch.arange(4), 7, namespace=3, slots=(10, 11, 12, 13), ranges=ranges)
    reset_root_state_from_table(env, torch.tensor([1, 3]), 7, namespace=3, slots=(10, 11, 12, 13), ranges=ranges)
    assert env._ea_last_reset_keys == {1: (3, 11, 1), 3: (3, 13, 1)}
    assert env._ea_reset_episode_counter.tolist() == [1, 2, 1, 2]
    quat = env.scene.asset.pose[0, 3:7]
    assert torch.allclose(quat, torch.tensor([math.cos(0.05), 0.0, 0.0, math.sin(0.05)]), atol=1e-7)
