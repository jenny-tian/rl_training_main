from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionBatch:
    history: torch.Tensor
    current_frame: torch.Tensor
    base_action: torch.Tensor
    residual_action: torch.Tensor
    reward: torch.Tensor
    next_history: torch.Tensor
    next_frame: torch.Tensor
    next_base_action: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_mask: torch.Tensor
    discount: torch.Tensor
    safety_mask: torch.Tensor
    curriculum_stage_id: torch.Tensor
    episode_id: torch.Tensor


@dataclass(frozen=True)
class SafetyState:
    projected_gravity_xy: torch.Tensor
    base_height: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    default_joint_pos: torch.Tensor
    action_scale: torch.Tensor
    nonfoot_contact: torch.Tensor
    control_timed_out: torch.Tensor


@dataclass(frozen=True)
class RewardSignals:
    base_lin_vel_xy: torch.Tensor
    command_lin_vel_xy: torch.Tensor
    base_ang_vel_z: torch.Tensor
    command_ang_vel_z: torch.Tensor
    projected_gravity_xy: torch.Tensor
    base_lin_vel_z: torch.Tensor
    foot_slip: torch.Tensor
    foot_impact: torch.Tensor
    joint_power: torch.Tensor
    nonfoot_contact: torch.Tensor


@dataclass(frozen=True)
class SafetyDecision:
    action: torch.Tensor
    residual: torch.Tensor
    paused: torch.Tensor
    projected: torch.Tensor
    reasons: tuple[str, ...]
    actor_version: int | None = None


@dataclass(frozen=True)
class UpdateStats:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    mean_q: float
    grad_norm: float
    accepted: bool
    version: int
    critic_updated: bool = False
    actor_updated: bool = False
    rejection_reason: str | None = None
