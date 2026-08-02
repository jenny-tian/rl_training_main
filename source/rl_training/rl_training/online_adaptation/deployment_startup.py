"""Shared Lite3 deployment startup lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.standup_trajectory import (
    FOLDED_JOINT_POS,
    STANDING_JOINT_POS,
    cubic_hermite_position,
    physical_targets_to_policy_action,
    standup_targets,
)


class Lite3StartupError(RuntimeError):
    """Raised when deterministic stand-up cannot safely complete."""


class Lite3DeploymentStartupWrapper:
    """Hide the belly-down stand-up procedure from policy-facing callers."""

    SUPPORTED_TASKS = frozenset(
        {"FlatOnline-Deeprobotics-Lite3-v0", "HeldoutBoxes-Deeprobotics-Lite3-v0"}
    )

    DEFAULT_BELLY_DOWN_BASE_HEIGHT = 0.08

    @classmethod
    def validate_task(cls, task: str) -> None:
        if task not in cls.SUPPORTED_TASKS:
            raise ValueError(f"Lite3 deployment startup is not supported for task {task!r}")

    def __init__(
        self,
        env,
        *,
        state_reader: Callable[[object], object],
        phase_duration: float = 1.5,
        handoff_contact_steps: int = 10,
        max_joint_velocity: float = 30.0,
    ):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.state_reader = state_reader
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

    def _zero_commands(self) -> None:
        manager = getattr(self.unwrapped, "command_manager", None)
        term = getattr(manager, "get_term", lambda _name: None)("base_velocity") if manager else None
        for name in ("vel_command_b", "target_vel_command_b"):
            value = getattr(term, name, None)
            if value is not None:
                value.zero_()

    def _step_count(self) -> int:
        dt = float(getattr(self.unwrapped, "step_dt", 0.02))
        return max(1, round(self.phase_duration / dt))

    def _validate_state(self, state, phase: int, elapsed: float) -> None:
        tensors = (state.joint_pos, state.joint_vel, state.root_height, state.projected_gravity_xy)
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

    def _advance_phase(self, start_pos: torch.Tensor, start_vel: torch.Tensor, phase: int) -> None:
        count = self._step_count()
        for index in range(count):
            elapsed = min((index + 1) * self.phase_duration / count, self.phase_duration)
            target, _ = standup_targets(
                start_pos,
                start_vel,
                elapsed=elapsed,
                phase_duration=self.phase_duration,
                phase=phase,
            )
            self._validate_state(self.state_reader(self.unwrapped), phase, elapsed)
            self._zero_commands()
            _, _, done, _ = self.env.step(self._startup_action(target))
            if bool(torch.as_tensor(done).any().item()):
                raise Lite3StartupError(f"phase {phase} at {elapsed:.3f}s terminated unexpectedly")

    def _validate_handoff(self) -> None:
        state = self.state_reader(self.unwrapped)
        if float(state.root_height.min().item()) < 0.20:
            raise Lite3StartupError("stand-up handoff base height is below 0.20 m")
        if float(state.projected_gravity_xy.norm(dim=-1).max().item()) > 0.65:
            raise Lite3StartupError("stand-up handoff projected gravity exceeds 0.65")
        if float(state.foot_forces.min().item()) <= 1.0:
            raise Lite3StartupError("stand-up handoff requires all four feet above 1 N")

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._set_startup_active(True)
        self._zero_commands()
        state = self.state_reader(self.unwrapped)
        start_pos = state.joint_pos.detach().clone()
        start_vel = state.joint_vel.detach().clone()
        self._advance_phase(start_pos, start_vel, 1)
        folded = FOLDED_JOINT_POS.to(device=start_pos.device, dtype=start_pos.dtype).expand_as(start_pos)
        self._advance_phase(folded, torch.zeros_like(start_vel), 2)

        standing = STANDING_JOINT_POS.to(device=start_pos.device, dtype=start_pos.dtype).expand_as(start_pos)
        handoff_start = self.state_reader(self.unwrapped)
        handoff_start_pos = handoff_start.joint_pos.detach().clone()
        handoff_start_vel = handoff_start.joint_vel.detach().clone()
        handoff_observation = observation
        handoff_step_info = {}
        contact_streak = 0
        max_handoff_steps = self.handoff_contact_steps * 5
        hold_duration = self.handoff_contact_steps * float(getattr(self.unwrapped, "step_dt", 0.02))
        for index in range(max_handoff_steps):
            state = self.state_reader(self.unwrapped)
            elapsed = self.phase_duration + index * float(getattr(self.unwrapped, "step_dt", 0.02))
            self._validate_state(state, 2, elapsed)
            all_feet_in_contact = bool(torch.all(state.foot_forces > 1.0).item())
            contact_streak = contact_streak + 1 if all_feet_in_contact else 0
            if contact_streak >= self.handoff_contact_steps:
                self._validate_handoff()
                break
            target, _ = cubic_hermite_position(
                handoff_start_pos,
                handoff_start_vel,
                standing,
                torch.zeros_like(standing),
                elapsed=min((index + 1) * float(getattr(self.unwrapped, "step_dt", 0.02)), hold_duration),
                duration=hold_duration,
            )
            self._zero_commands()
            handoff_observation, _, done, handoff_step_info = self.env.step(self._startup_action(target))
            if bool(torch.as_tensor(done).any().item()):
                raise Lite3StartupError("stand-up handoff hold terminated unexpectedly")
        else:
            raise Lite3StartupError(
                f"stand-up handoff did not maintain all four feet above 1 N for "
                f"{self.handoff_contact_steps} consecutive steps"
            )
        self._set_startup_active(False)
        self.startup_generation += 1
        self._pending_generation = self.startup_generation
        info = dict(info)
        info["startup_generation"] = self.startup_generation
        info.update(dict(handoff_step_info))
        info["startup_generation"] = self.startup_generation
        return handoff_observation, info

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
    return wrapper_cls(rsl_env, state_reader=isaaclab_startup_state)
