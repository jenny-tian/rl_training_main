"""Pure Lite3 stand-up trajectory and policy-action conversion helpers."""

from __future__ import annotations

import torch

from .lite3_joint_conventions import (
    LITE3_POLICY_JOINT_NAMES,
    physical_targets_from_action,
)


FOLDED_LEG_JOINT_POS = (0.0, -1.3545311211, 2.5494772717)
STANDING_LEG_JOINT_POS = (0.0, -0.6535351663, 1.2710916139)
FOLDED_BASE_HEIGHT = 0.12
STANDING_BASE_HEIGHT = 0.33
THIGH_LENGTH = 0.20
SHANK_LENGTH = 0.21
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


def _leg_joints_from_height(height: torch.Tensor) -> torch.Tensor:
    """Map the reference Lite3 leg height to HipX/HipY/Knee joint positions."""
    a = (THIGH_LENGTH * THIGH_LENGTH + height * height - SHANK_LENGTH * SHANK_LENGTH) / (
        2.0 * height * THIGH_LENGTH
    )
    b = (THIGH_LENGTH * THIGH_LENGTH + SHANK_LENGTH * SHANK_LENGTH - height * height) / (
        2.0 * THIGH_LENGTH * SHANK_LENGTH
    )
    hip_y = -torch.acos(a.clamp(-1.0, 1.0))
    knee = torch.pi - torch.acos(b.clamp(-1.0, 1.0))
    return torch.cat((torch.zeros_like(height), hip_y, knee), dim=-1)


def _leg_joint_velocity_from_height(
    height: torch.Tensor, height_velocity: torch.Tensor
) -> torch.Tensor:
    """Map a height velocity to the corresponding inverse-kinematics joint velocities."""
    delta = 1.0e-6
    a = (THIGH_LENGTH * THIGH_LENGTH + height * height - SHANK_LENGTH * SHANK_LENGTH) / (
        2.0 * height * THIGH_LENGTH
    )
    b = (THIGH_LENGTH * THIGH_LENGTH + SHANK_LENGTH * SHANK_LENGTH - height * height) / (
        2.0 * THIGH_LENGTH * SHANK_LENGTH
    )
    da_dh = (1.0 - (THIGH_LENGTH * THIGH_LENGTH - SHANK_LENGTH * SHANK_LENGTH) / height.square()) / (
        2.0 * THIGH_LENGTH
    )
    db_dh = -height / (THIGH_LENGTH * SHANK_LENGTH)
    hip_y_velocity = da_dh / torch.sqrt((1.0 - a.square()).clamp_min(delta)) * height_velocity
    knee_velocity = db_dh / torch.sqrt((1.0 - b.square()).clamp_min(delta)) * height_velocity
    return torch.cat((torch.zeros_like(height_velocity), hip_y_velocity, knee_velocity), dim=-1)


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
    if phase == 1:
        target = FOLDED_JOINT_POS.to(device=start.device, dtype=start.dtype)
        return cubic_hermite_position(
            start,
            start_velocity,
            target.expand_as(start),
            torch.zeros_like(start),
            elapsed,
            phase_duration,
        )

    height_start = torch.full(
        (*start.shape[:-1], 1), FOLDED_BASE_HEIGHT, device=start.device, dtype=start.dtype
    )
    height_end = torch.full_like(height_start, STANDING_BASE_HEIGHT)
    height, height_velocity = cubic_hermite_position(
        height_start,
        torch.zeros_like(height_start),
        height_end,
        torch.zeros_like(height_end),
        elapsed,
        phase_duration,
    )
    leg_pos = _leg_joints_from_height(height)
    leg_vel = _leg_joint_velocity_from_height(height, height_velocity)
    return leg_pos.repeat(1, 4), leg_vel.repeat(1, 4)


def physical_targets_to_policy_action(target: torch.Tensor) -> torch.Tensor:
    _validate_joint_tensor(target, "target")
    default_action = torch.zeros_like(target)
    default_action_target = physical_targets_from_action(default_action)
    scale = physical_targets_from_action(torch.ones_like(target)) - default_action_target
    return (target - default_action_target) / scale
