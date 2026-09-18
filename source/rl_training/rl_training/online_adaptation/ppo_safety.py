"""Collect PPO requests through a fixed safety transformation.

The Gaussian request is the policy's random variable.  The safety filter is
part of the environment transition, including its residual history.  Its
executed action must not replace the request in the likelihood, and whether
it projects a request must not select the training distribution.
"""

from __future__ import annotations

import torch

from .ppo_rollout import PPORollout
from .types import SafetyDecision


SAMPLING_CONTRACT = "requested_gaussian_through_fixed_safety_v1"


def requested_action_rollout(
    *,
    observations: torch.Tensor,
    requested_actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    next_values: torch.Tensor,
    policy_version: int,
    safety_decision: SafetyDecision,
) -> PPORollout:
    """Retain projected requests with the probability used to sample them.

    A pause is selected from the pre-action safety state and ignores the
    request entirely, so it is excluded to avoid zero-expectation actor noise.
    The current updater shares this mask with the critic; paused transitions
    remain in GAE but do not provide direct value supervision.
    """
    return PPORollout(
        observations=observations.detach(),
        actions=requested_actions.detach(),
        old_log_probs=old_log_probs.detach(),
        old_values=old_values.detach(),
        rewards=rewards.detach(),
        terminated=terminated.detach(),
        truncated=truncated.detach(),
        next_values=next_values.detach(),
        policy_version=policy_version,
        valid_mask=(~safety_decision.paused).to(dtype=torch.float32).detach(),
    )
