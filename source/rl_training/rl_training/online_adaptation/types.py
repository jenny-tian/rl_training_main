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
    requested_residual_action: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # Direct callers without filtering may omit the identical request.
        if self.requested_residual_action is None:
            object.__setattr__(self, "requested_residual_action", self.residual_action)


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
    foot_height: torch.Tensor
    foot_contact: torch.Tensor
    foot_edge_offset: torch.Tensor | None = None


@dataclass(frozen=True)
class SafetyDecision:
    action: torch.Tensor
    residual: torch.Tensor
    paused: torch.Tensor
    projected: torch.Tensor
    reasons: tuple[str, ...]
    actor_version: int | None = None
    risk_score: torch.Tensor | None = None
    risk_scale: torch.Tensor | None = None


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
    mean_target_q: float | None = None
    mean_td_error: float | None = None
    mean_flat_q: float | None = None
    mean_online_q: float | None = None
    flat_anchor_loss: float | None = None
    flat_batch_fraction: float = 0.0
    n_step: int = 1
    mean_unclipped_target_q: float | None = None
    target_q_clip_fraction: float | None = None
    mean_entropy_bonus: float | None = None
