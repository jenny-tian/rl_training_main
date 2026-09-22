from __future__ import annotations

import torch
from dataclasses import dataclass

from .types import RewardSignals


@dataclass(frozen=True)
class RewardWeights:
    velocity_error_scale: float = 4.0
    velocity_weight: float = 2.0
    forward_progress_weight: float = 0.5
    post_step_progress_weight: float = 2.0
    stall_penalty_weight: float = 2.0
    completion_bonus: float = 5.0
    yaw_error_scale: float = 2.0
    yaw_weight: float = 0.5
    tilt_weight: float = 2.0
    vertical_velocity_weight: float = 0.5
    foot_slip_weight: float = 0.05
    foot_impact_weight: float = 0.02
    joint_power_weight: float = 0.001
    residual_weight: float = 0.10
    residual_rate_weight: float = 0.05
    nonfoot_contact_weight: float = 2.0
    post_step_stability_weight: float = 3.0
    pre_step_clearance_weight: float = 1.0
    pre_step_clearance_target: float = 0.06
    pre_step_foot_forward_weight: float = 2.0
    pre_step_foot_forward_target: float = 0.04
    lateral_deviation_weight: float = 1.0
    lateral_deviation_limit: float = 1.65
    minimum_reward: float = -10.0
    maximum_reward: float = 5.0
    rough_swing_lift_weight: float = 0.0
    rough_swing_lift_target: float = 0.045


def compute_online_reward(
    signals: RewardSignals,
    residual: torch.Tensor,
    previous_residual: torch.Tensor,
    weights: RewardWeights,
    forward_progress: torch.Tensor | None = None,
    post_step_mask: torch.Tensor | None = None,
    pre_step_clearance_mask: torch.Tensor | None = None,
    progress_stalled_mask: torch.Tensor | None = None,
    completion_mask: torch.Tensor | None = None,
    lateral_deviation: torch.Tensor | None = None,
    rough_swing_mask: torch.Tensor | None = None,
    swing_lift: torch.Tensor | None = None,
    action_projection_mismatch: torch.Tensor | None = None,
    action_projection_penalty: float = 0.0,
) -> torch.Tensor:
    velocity_error = torch.sum(
        (signals.base_lin_vel_xy - signals.command_lin_vel_xy).square(), dim=-1, keepdim=True
    )
    velocity_term = torch.exp(-weights.velocity_error_scale * velocity_error)
    yaw_term = torch.exp(
        -weights.yaw_error_scale * (signals.base_ang_vel_z - signals.command_ang_vel_z).square()
    )
    tilt = torch.sum(signals.projected_gravity_xy.square(), dim=-1, keepdim=True)
    residual_cost = torch.mean(residual.square(), dim=-1, keepdim=True)
    residual_rate_cost = torch.mean((residual - previous_residual).square(), dim=-1, keepdim=True)
    pre_step_foot_forward_term = torch.zeros_like(velocity_term)
    if pre_step_clearance_mask is None:
        pre_step_clearance_term = torch.zeros_like(velocity_term)
    else:
        if pre_step_clearance_mask.shape != velocity_term.shape:
            raise ValueError(
                "pre_step_clearance_mask must have shape (batch, 1), "
                f"got {tuple(pre_step_clearance_mask.shape)}"
            )
        if signals.foot_height.shape != signals.foot_contact.shape:
            raise ValueError("foot_height and foot_contact must have matching shapes")
        if signals.foot_height.ndim != 2 or signals.foot_height.shape[0] != velocity_term.shape[0]:
            raise ValueError("foot_height and foot_contact must have shape (batch, feet)")
        if weights.pre_step_clearance_target <= 0.0:
            raise ValueError("pre_step_clearance_target must be positive")
        clearance_mask = torch.nan_to_num(
            pre_step_clearance_mask, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        swing_mask = 1.0 - torch.nan_to_num(
            signals.foot_contact, nan=1.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        clearance_score = torch.nan_to_num(
            signals.foot_height / weights.pre_step_clearance_target,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        best_swing_clearance = (clearance_score * swing_mask).amax(dim=-1, keepdim=True)
        pre_step_clearance_term = (
            weights.pre_step_clearance_weight * clearance_mask * best_swing_clearance
        )
        if signals.foot_edge_offset is not None:
            if signals.foot_edge_offset.shape != signals.foot_contact.shape:
                raise ValueError("foot_edge_offset and foot_contact must have matching shapes")
            if weights.pre_step_foot_forward_target <= 0.0:
                raise ValueError("pre_step_foot_forward_target must be positive")
            # Score the approach continuously: zero one target-length before
            # the edge, 0.5 at the edge, and one after crossing by the target.
            # Clamping all negative offsets to zero made a foot 1 mm behind the
            # edge indistinguishable from one several centimeters behind it.
            forward_score = torch.nan_to_num(
                (
                    signals.foot_edge_offset
                    + weights.pre_step_foot_forward_target
                )
                / (2.0 * weights.pre_step_foot_forward_target),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            # Height and forward placement must come from the same swing foot.
            # Separate maxima let the policy collect both rewards with two
            # unrelated legs while the body remains stuck at the edge.
            best_swing_placement = (
                clearance_score * forward_score * swing_mask
            ).amax(dim=-1, keepdim=True)
            pre_step_clearance_term = (
                weights.pre_step_clearance_weight
                * clearance_mask
                * best_swing_placement
            )
            pre_step_foot_forward_term = (
                weights.pre_step_foot_forward_weight
                * clearance_mask
                * best_swing_placement
            )
    if post_step_mask is None:
        post_step_term = torch.zeros_like(velocity_term)
    else:
        if post_step_mask.shape != velocity_term.shape:
            raise ValueError(
                "post_step_mask must have shape (batch, 1), "
                f"got {tuple(post_step_mask.shape)}"
            )
        mask = torch.nan_to_num(post_step_mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        post_step_instability = (
            tilt
            + signals.base_lin_vel_z.square()
            + signals.nonfoot_contact
        )
        post_step_term = weights.post_step_stability_weight * mask * post_step_instability
    post_step_progress_term = torch.zeros_like(velocity_term)
    if forward_progress is None:
        progress_term = torch.zeros_like(velocity_term)
    else:
        if forward_progress.shape != velocity_term.shape:
            raise ValueError(
                "forward_progress must have shape (batch, 1), "
                f"got {tuple(forward_progress.shape)}"
            )
        normalized_progress = torch.nan_to_num(
            forward_progress,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)
        progress_term = weights.forward_progress_weight * normalized_progress
        if post_step_mask is not None:
            post_step_progress_term = (
                weights.post_step_progress_weight
                * torch.nan_to_num(post_step_mask, nan=0.0, posinf=1.0, neginf=0.0)
                .clamp(0.0, 1.0)
                * normalized_progress.clamp_min(0.0)
            )
    if progress_stalled_mask is None:
        stall_penalty = torch.zeros_like(velocity_term)
    else:
        if progress_stalled_mask.shape != velocity_term.shape:
            raise ValueError(
                "progress_stalled_mask must have shape (batch, 1), "
                f"got {tuple(progress_stalled_mask.shape)}"
            )
        stall_penalty = weights.stall_penalty_weight * torch.nan_to_num(
            progress_stalled_mask, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
    if completion_mask is None:
        completion_term = torch.zeros_like(velocity_term)
    else:
        if completion_mask.shape != velocity_term.shape:
            raise ValueError(
                "completion_mask must have shape (batch, 1), "
                f"got {tuple(completion_mask.shape)}"
            )
        completion_term = weights.completion_bonus * torch.nan_to_num(
            completion_mask, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
    if lateral_deviation is None:
        lateral_deviation_term = torch.zeros_like(velocity_term)
    else:
        if lateral_deviation.shape != velocity_term.shape:
            raise ValueError("lateral_deviation must have shape (batch, 1)")
        if weights.lateral_deviation_limit <= 0.0:
            raise ValueError("lateral_deviation_limit must be positive")
        normalized_lateral = torch.nan_to_num(
            lateral_deviation / weights.lateral_deviation_limit,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        lateral_deviation_term = weights.lateral_deviation_weight * normalized_lateral.square()
    rough_swing_term = torch.zeros_like(velocity_term)
    if rough_swing_mask is not None and weights.rough_swing_lift_weight > 0.0:
        if rough_swing_mask.shape != velocity_term.shape:
            raise ValueError("rough_swing_mask must have shape (batch, 1)")
        if swing_lift is None or swing_lift.shape != signals.foot_contact.shape:
            raise ValueError("swing_lift must match foot_contact")
        if not weights.rough_swing_lift_target > 0.0:
            raise ValueError("rough_swing_lift_target must be positive")
        contact = torch.nan_to_num(signals.foot_contact, nan=1.0).clamp(0.0, 1.0)
        lift = torch.nan_to_num(swing_lift, nan=0.0, posinf=0.0, neginf=0.0)
        # Peak at the target: lifting ever higher is not a reward shortcut.
        score = (1.0 - (lift / weights.rough_swing_lift_target - 1.0).abs()).clamp(0.0, 1.0)
        swing_score = (score * (1.0 - contact)).mean(dim=-1, keepdim=True)
        support = (contact > 0.5).any(dim=-1, keepdim=True)
        # No bonus for hopping in place or holding a leg up while stuck.
        progress = torch.zeros_like(velocity_term) if forward_progress is None else torch.nan_to_num(
            forward_progress, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        rough_swing_term = (
            weights.rough_swing_lift_weight * swing_score * support * progress
            * torch.nan_to_num(rough_swing_mask, nan=0.0).clamp(0.0, 1.0)
        )
    if not isinstance(action_projection_penalty, (int, float)) or not torch.isfinite(
        torch.as_tensor(action_projection_penalty)
    ) or action_projection_penalty < 0.0:
        raise ValueError("action_projection_penalty must be finite and non-negative")
    if action_projection_mismatch is None:
        projection_penalty = torch.zeros_like(velocity_term)
    else:
        if action_projection_mismatch.shape != velocity_term.shape:
            raise ValueError("action_projection_mismatch must have shape (batch, 1)")
        projection_penalty = float(action_projection_penalty) * torch.nan_to_num(
            action_projection_mismatch, nan=0.0, posinf=1.0e6, neginf=0.0
        ).clamp_min(0.0)
    reward = (
        weights.velocity_weight * velocity_term
        + progress_term
        + post_step_progress_term
        + completion_term
        - lateral_deviation_term
        + pre_step_clearance_term
        + pre_step_foot_forward_term
        + rough_swing_term
        + weights.yaw_weight * yaw_term
        - weights.tilt_weight * tilt
        - weights.vertical_velocity_weight * signals.base_lin_vel_z.square()
        - weights.foot_slip_weight * signals.foot_slip
        - weights.foot_impact_weight * signals.foot_impact
        - weights.joint_power_weight * signals.joint_power
        - weights.residual_weight * residual_cost
        - weights.residual_rate_weight * residual_rate_cost
        - weights.nonfoot_contact_weight * signals.nonfoot_contact
        - post_step_term
        - stall_penalty
        - projection_penalty
    )
    return torch.clamp(reward, weights.minimum_reward, weights.maximum_reward)
