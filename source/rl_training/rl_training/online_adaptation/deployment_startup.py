"""Shared Lite3 deployment startup lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import torch

from rl_training.lite3_joint_conventions import (
    clip_lite3_policy_action,
)
from rl_training.standup_trajectory import physical_targets_to_policy_action


class Lite3StartupError(RuntimeError):
    """Raised when deterministic standing initialization is unsafe."""


class Lite3DeploymentStartupWrapper:
    """Preserve deployment reset semantics around a standing initialization."""

    SUPPORTED_TASKS = frozenset(
        {
            "FlatOnline-Deeprobotics-Lite3-v0",
            "HeldoutBoxes-Deeprobotics-Lite3-v0",
            "ContinuousRough-Deeprobotics-Lite3-v0",
            "Grass-Deeprobotics-Lite3-v0",
            "RockyRoad-Deeprobotics-Lite3-v0",
        }
    )

    DEFAULT_STANDING_BASE_HEIGHT = 0.375

    @classmethod
    def validate_task(cls, task: str) -> None:
        if task not in cls.SUPPORTED_TASKS:
            raise ValueError(f"Lite3 deployment startup is not supported for task {task!r}")

    def __init__(
        self,
        env,
        *,
        state_reader: Callable[[object], object],
        velocity_target_writer: Callable[[object, torch.Tensor], None] | None = None,
        phase_duration: float = 1.5,
        handoff_contact_steps: int = 10,
        max_joint_velocity: float = 30.0,
    ):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.state_reader = state_reader
        self.velocity_target_writer = velocity_target_writer
        self.phase_duration = float(phase_duration)
        self.handoff_contact_steps = int(handoff_contact_steps)
        self.max_joint_velocity = float(max_joint_velocity)
        if self.phase_duration <= 0.0 or self.handoff_contact_steps <= 0:
            raise ValueError("phase_duration and handoff_contact_steps must be positive")
        self.startup_active = False
        self.startup_generation = 0
        self._pending_generation: int | None = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _set_startup_active(self, active: bool) -> None:
        self.startup_active = active
        setattr(self.unwrapped, "startup_active", active)

    def _write_velocity_target(self, target: torch.Tensor) -> None:
        if self.velocity_target_writer is not None:
            self.velocity_target_writer(self.unwrapped, target)

    def _validate_state(self, state, phase: int, elapsed: float) -> None:
        tensors = (
            state.joint_pos,
            state.joint_vel,
            state.root_height,
            state.projected_gravity_xy,
            state.foot_forces,
        )
        if not all(torch.isfinite(value).all().item() for value in tensors):
            raise Lite3StartupError(f"phase {phase} at {elapsed:.3f}s has non-finite state")
        max_speed = float(state.joint_vel.abs().amax().item())
        if max_speed > self.max_joint_velocity:
            index = int(state.joint_vel.abs().argmax().item() % state.joint_vel.shape[-1])
            raise Lite3StartupError(
                f"phase {phase} at {elapsed:.3f}s joint {index} speed {max_speed:g} exceeds {self.max_joint_velocity:g}"
            )
        if torch.any(state.joint_pos < state.joint_lower) or torch.any(state.joint_pos > state.joint_upper):
            raise Lite3StartupError(f"phase {phase} at {elapsed:.3f}s joint position limit exceeded")

    def _startup_action(self, target: torch.Tensor) -> torch.Tensor:
        return clip_lite3_policy_action(physical_targets_to_policy_action(target))

    def _validate_handoff(self) -> None:
        state = self.state_reader(self.unwrapped)
        if float(state.root_height.min().item()) < 0.20:
            raise Lite3StartupError("standing handoff base height is below 0.20 m")
        if float(state.projected_gravity_xy.norm(dim=-1).max().item()) > 0.65:
            raise Lite3StartupError("standing handoff projected gravity exceeds 0.65")

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._set_startup_active(True)
        state = self.state_reader(self.unwrapped)
        self._validate_state(state, 0, 0.0)
        self._validate_handoff()
        self._set_startup_active(False)
        self.startup_generation += 1
        self._pending_generation = self.startup_generation
        info = dict(info)
        info["startup_generation"] = self.startup_generation
        return observation, info

    def consume_startup_generation(self) -> int | None:
        generation = self._pending_generation
        self._pending_generation = None
        return generation

    def step(self, action):
        observation, reward, done, extras = self.env.step(action)
        if bool(torch.as_tensor(done).any().item()):
            observation, reset_info = self.reset()
            extras = dict(extras)
            extras["startup_generation"] = reset_info["startup_generation"]
        return observation, reward, done, extras


def isaaclab_startup_state(env) -> SimpleNamespace:
    """Read the canonical Lite3 state required by the startup controller."""
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    from rl_training.online_adaptation.isaaclab_adapter import select_joint_order
    from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES

    joint_ids, _ = robot.find_joints(list(LITE3_POLICY_JOINT_NAMES), preserve_order=True)
    foot_ids = sensor.find_bodies(".*_FOOT")[0]
    return SimpleNamespace(
        joint_pos=select_joint_order(robot.data.joint_pos, joint_ids),
        joint_vel=select_joint_order(robot.data.joint_vel, joint_ids),
        root_height=robot.data.root_pos_w[:, 2:3],
        projected_gravity_xy=robot.data.projected_gravity_b[:, :2],
        joint_lower=select_joint_order(robot.data.soft_joint_pos_limits, joint_ids)[..., 0],
        joint_upper=select_joint_order(robot.data.soft_joint_pos_limits, joint_ids)[..., 1],
        foot_forces=torch.linalg.norm(sensor.data.net_forces_w[:, foot_ids], dim=-1),
    )


def isaaclab_startup_velocity_target(env, target: torch.Tensor) -> None:
    """Write a canonical Lite3 velocity target to the articulation actuator buffers."""
    from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES

    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(list(LITE3_POLICY_JOINT_NAMES), preserve_order=True)
    if tuple(joint_names) != LITE3_POLICY_JOINT_NAMES:
        raise ValueError("Lite3 startup velocity target could not resolve canonical joint order")
    robot.set_joint_velocity_target(target, joint_ids=joint_ids)


def make_lite3_deployment_env(
    task: str,
    env_cfg,
    agent_cfg,
    *,
    gym_module=None,
    wrapper_cls=Lite3DeploymentStartupWrapper,
    env_kwargs: dict | None = None,
    env_transform: Callable[[object], object] | None = None,
):
    """Construct the RSL environment and install the shared startup wrapper."""
    wrapper_cls.validate_task(task)
    if gym_module is None:
        import gymnasium as gym_module
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = gym_module.make(task, cfg=env_cfg, **(env_kwargs or {}))
    if env_transform is not None:
        env = env_transform(env)
    rsl_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return wrapper_cls(
        rsl_env,
        state_reader=isaaclab_startup_state,
        velocity_target_writer=isaaclab_startup_velocity_target,
    )
