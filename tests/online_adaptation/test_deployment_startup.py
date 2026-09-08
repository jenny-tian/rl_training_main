from dataclasses import dataclass

import pytest
import torch

from rl_training.online_adaptation.deployment_startup import (
    Lite3DeploymentStartupWrapper,
    Lite3StartupError,
)
from rl_training.lite3_joint_conventions import physical_targets_from_action


@dataclass
class State:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_height: torch.Tensor
    projected_gravity_xy: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    foot_forces: torch.Tensor


class FakeVelocityCommand:
    def __init__(self):
        self.vel_command_b = torch.zeros(1, 3)
        self.target_vel_command_b = torch.tensor([[0.3, 0.0, 0.0]])


class FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "base_velocity"
        return self.command


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.startup_active = False
        self.step_dt = 0.02
        self.steps = 0
        self.commands = torch.zeros(1, 3)
        self.done_on_next_step = False
        self.actions = []
        self.provide_foot_contact = True
        self.reset_joint_vel = torch.zeros(1, 12)
        self.velocity_command = FakeVelocityCommand()
        self.command_manager = FakeCommandManager(self.velocity_command)
        standing = physical_targets_from_action(torch.zeros(1, 12))
        self.state = State(
            standing, torch.zeros(1, 12), torch.tensor([[0.375]]),
            torch.zeros(1, 2), torch.full((1, 12), -10.0), torch.full((1, 12), 10.0),
            torch.zeros(1, 4),
        )

    def reset(self, **kwargs):
        self.steps = 0
        self.actions.clear()
        self.velocity_command.vel_command_b.zero_()
        self.velocity_command.target_vel_command_b[:] = torch.tensor([[0.3, 0.0, 0.0]])
        self.state.joint_pos[:] = physical_targets_from_action(torch.zeros(1, 12))
        self.state.joint_vel[:] = self.reset_joint_vel
        self.state.root_height[:] = 0.375
        self.state.projected_gravity_xy.zero_()
        self.state.foot_forces.zero_()
        return torch.zeros(1, 45), {"reset": True}

    def step(self, action):
        self.actions.append(action.clone())
        self.steps += 1
        done = self.done_on_next_step
        self.done_on_next_step = False
        if self.steps >= 10 and self.provide_foot_contact:
            self.state.root_height[:] = 0.324
            self.state.foot_forces[:] = 10.0
        return torch.zeros(1, 45), torch.zeros(1), torch.tensor([done]), {}


def reader(env):
    return env.state


@pytest.mark.parametrize(
    "task",
    (
        "FlatOnline-Deeprobotics-Lite3-v0",
        "HeldoutBoxes-Deeprobotics-Lite3-v0",
        "ContinuousRough-Deeprobotics-Lite3-v0",
        "Grass-Deeprobotics-Lite3-v0",
    ),
)
def test_deployment_wrapper_accepts_only_declared_evaluation_tasks(task):
    Lite3DeploymentStartupWrapper.validate_task(task)


def test_deployment_wrapper_rejects_training_task():
    with pytest.raises(ValueError, match="not supported"):
        Lite3DeploymentStartupWrapper.validate_task("Rough-Deeprobotics-Lite3-v0")


def test_wrapper_hides_startup_steps_and_reports_generation():
    env = FakeEnv()
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)
    observation, info = wrapped.reset()
    assert env.steps == 0
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


def test_wrapper_repeats_startup_after_episode_done():
    env = FakeEnv()
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)
    wrapped.reset()
    wrapped.consume_startup_generation()
    env.done_on_next_step = True
    _, _, done, extras = wrapped.step(torch.zeros(1, 12))
    assert done.item()
    assert extras["startup_generation"] == 2
    assert wrapped.consume_startup_generation() == 2


def test_wrapper_rejects_unsafe_joint_speed_with_phase_context():
    env = FakeEnv()
    env.reset_joint_vel[:, 3] = 31.0
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)
    with pytest.raises(Lite3StartupError, match="phase 0.*joint 3.*30"):
        wrapped.reset()


def test_wrapper_starts_from_standing_pose_without_folded_trajectory():
    env = FakeEnv()
    velocity_targets = []
    wrapped = Lite3DeploymentStartupWrapper(
        env,
        state_reader=reader,
        velocity_target_writer=lambda _env, target: velocity_targets.append(target.clone()),
    )
    wrapped.reset()
    assert env.actions == []
    assert velocity_targets == []


def test_wrapper_preserves_acceleration_limited_target_command():
    env = FakeEnv()
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)

    wrapped.reset()

    assert torch.equal(env.velocity_command.vel_command_b, torch.zeros(1, 3))
    assert torch.equal(
        env.velocity_command.target_vel_command_b,
        torch.tensor([[0.3, 0.0, 0.0]]),
    )


def test_wrapper_handoff_does_not_require_contact_forces_for_standing_spawn():
    env = FakeEnv()
    env.provide_foot_contact = False
    wrapped = Lite3DeploymentStartupWrapper(env, state_reader=reader)

    wrapped.reset()

    assert wrapped.startup_active is False
    assert wrapped.startup_generation == 1
