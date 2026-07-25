from __future__ import annotations

import torch
from dataclasses import dataclass

from .types import RewardSignals


@dataclass(frozen=True)
class RewardWeights:
    velocity_error_scale: float = 4.0
    velocity_weight: float = 2.0
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
    minimum_reward: float = -10.0
    maximum_reward: float = 5.0


def compute_online_reward(
    signals: RewardSignals,
    residual: torch.Tensor,
    previous_residual: torch.Tensor,
    weights: RewardWeights,
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
    reward = (
        weights.velocity_weight * velocity_term
        + weights.yaw_weight * yaw_term
        - weights.tilt_weight * tilt
        - weights.vertical_velocity_weight * signals.base_lin_vel_z.square()
        - weights.foot_slip_weight * signals.foot_slip
        - weights.foot_impact_weight * signals.foot_impact
        - weights.joint_power_weight * signals.joint_power
        - weights.residual_weight * residual_cost
        - weights.residual_rate_weight * residual_rate_cost
        - weights.nonfoot_contact_weight * signals.nonfoot_contact
    )
    return torch.clamp(reward, weights.minimum_reward, weights.maximum_reward)
