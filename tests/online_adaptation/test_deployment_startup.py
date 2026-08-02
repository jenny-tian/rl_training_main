from dataclasses import dataclass

import pytest
import torch

from rl_training.online_adaptation.deployment_startup import (
    Lite3DeploymentStartupWrapper,
    Lite3StartupError,
)
from rl_training.standup_trajectory import FOLDED_JOINT_POS, STANDING_JOINT_POS


@dataclass
class State:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_height: torch.Tensor
    projected_gravity_xy: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    foot_forces: torch.Tensor


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.startup_active = False
        self.step_dt = 0.02
        self.steps = 0
        self.commands = torch.zeros(1, 3)
        self.state = State(
            FOLDED_JOINT_POS.clone(), torch.zeros(1, 12), torch.tensor([[0.25]]),
            torch.zeros(1, 2), torch.full((1, 12), -10.0), torch.full((1, 12), 10.0),
            torch.zeros(1, 4),
        )

    def reset(self, **kwargs):
        self.steps = 0
        return torch.zeros(1, 45), {"reset": True}

    def step(self, action):
        self.steps += 1
        if self.steps >= 150:
            self.state.root_height[:] = 0.30
            self.state.foot_forces[:] = 10.0
        return torch.zeros(1, 45), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}


def reader(env):
    return env.state


def test_wrapper_hides_startup_steps_and_reports_generation():
    env = FakeEnv()
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)
    observation, info = wrapped.reset()
    assert env.steps >= 150
    assert info["startup_generation"] == 1
    assert wrapped.startup_active is False
    assert env.startup_active is False
    assert observation.shape == (1, 45)
    assert wrapped.consume_startup_generation() == 1
    assert wrapped.consume_startup_generation() is None


def test_wrapper_repeats_startup_after_outer_reset():
    env = FakeEnv()
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)
    wrapped.reset()
    first_steps = env.steps
    wrapped.reset()
    assert env.steps == first_steps
    assert wrapped.startup_generation == 2


def test_wrapper_rejects_unsafe_joint_speed_with_phase_context():
    env = FakeEnv()
    env.state.joint_vel[:, 3] = 31.0
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)
    with pytest.raises(Lite3StartupError, match="phase 1.*joint 3.*30"):
        wrapped.reset()
