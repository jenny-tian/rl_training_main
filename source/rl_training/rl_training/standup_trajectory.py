"""Pure Lite3 stand-up trajectory and policy-action conversion helpers."""

from __future__ import annotations

import torch

from .lite3_joint_conventions import (
    LITE3_POLICY_JOINT_NAMES,
    physical_targets_from_action,
)


FOLDED_LEG_JOINT_POS = (0.0, -1.3545311211, 2.5494772717)
STANDING_LEG_JOINT_POS = (0.0, -0.6535351663, 1.2710916139)
FOLDED_JOINT_POS = torch.tensor([FOLDED_LEG_JOINT_POS] * 4, dtype=torch.float32).reshape(1, -1)
STANDING_JOINT_POS = torch.tensor([STANDING_LEG_JOINT_POS] * 4, dtype=torch.float32).reshape(1, -1)


def _validate_joint_tensor(value: torch.Tensor, name: str) -> None:
    if value.ndim == 0 or value.shape[-1] != len(LITE3_POLICY_JOINT_NAMES):
        raise ValueError(f"{name} must have a final dimension of 12")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_trajectory_tensor(value: torch.Tensor, name: str) -> None:
    if value.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def cubic_hermite_position(
    start: torch.Tensor,
    start_velocity: torch.Tensor,
    end: torch.Tensor,
    end_velocity: torch.Tensor,
    elapsed: float,
    duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    for value, name in ((start, "start"), (start_velocity, "start_velocity"), (end, "end"), (end_velocity, "end_velocity")):
        _validate_trajectory_tensor(value, name)
    if not (start.shape == start_velocity.shape == end.shape == end_velocity.shape):
        raise ValueError("trajectory tensors must have the same shape")
    if not torch.isfinite(torch.tensor([elapsed, duration])).all() or duration <= 0.0:
        raise ValueError("elapsed and duration must be finite and duration must be positive")
    if elapsed < 0.0 or elapsed > duration:
        raise ValueError("elapsed must be within the trajectory duration")
    tau = torch.as_tensor(elapsed / duration, device=start.device, dtype=start.dtype)
    tau2 = tau * tau
    tau3 = tau2 * tau
    h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0
    h10 = tau3 - 2.0 * tau2 + tau
    h01 = -2.0 * tau3 + 3.0 * tau2
    h11 = tau3 - tau2
    position = h00 * start + h10 * duration * start_velocity + h01 * end + h11 * duration * end_velocity
    dh00 = 6.0 * tau2 - 6.0 * tau
    dh10 = 3.0 * tau2 - 4.0 * tau + 1.0
    dh01 = -6.0 * tau2 + 6.0 * tau
    dh11 = 3.0 * tau2 - 2.0 * tau
    velocity = (dh00 * start + dh10 * duration * start_velocity + dh01 * end + dh11 * duration * end_velocity) / duration
    return position, velocity


def standup_targets(
    start: torch.Tensor,
    start_velocity: torch.Tensor,
    *,
    elapsed: float,
    phase_duration: float,
    phase: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_joint_tensor(start, "start")
    _validate_joint_tensor(start_velocity, "start_velocity")
    if start.shape != start_velocity.shape:
        raise ValueError("start and start_velocity must have the same shape")
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    target = (FOLDED_JOINT_POS if phase == 1 else STANDING_JOINT_POS).to(device=start.device, dtype=start.dtype)
    phase_start = start if phase == 1 else FOLDED_JOINT_POS.to(device=start.device, dtype=start.dtype).expand_as(start)
    phase_velocity = start_velocity if phase == 1 else torch.zeros_like(start)
    return cubic_hermite_position(phase_start, phase_velocity, target.expand_as(start), torch.zeros_like(start), elapsed, phase_duration)


def physical_targets_to_policy_action(target: torch.Tensor) -> torch.Tensor:
    _validate_joint_tensor(target, "target")
    default_action = torch.zeros_like(target)
    default_action_target = physical_targets_from_action(default_action)
    scale = physical_targets_from_action(torch.ones_like(target)) - default_action_target
    return (target - default_action_target) / scale
