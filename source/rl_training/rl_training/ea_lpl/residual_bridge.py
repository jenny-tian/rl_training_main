"""Frozen base action plus a requested residual through the deployment SafetyFilter."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.types import SafetyDecision, SafetyState
from rl_training.source_scan.bridge import FrozenPolicySafetyBridge


@dataclass(frozen=True)
class ResidualSafetyTrace:
    raw_policy_action: torch.Tensor
    base_action: torch.Tensor
    residual_request: torch.Tensor
    requested_action: torch.Tensor
    executed_action: torch.Tensor
    decision: SafetyDecision
    filter_state_before: dict
    safety_state: SafetyState


class ResidualSafetyBridge:
    def __init__(self, adapter, safety_filter, *, action_clip=clip_lite3_policy_action):
        self.adapter = adapter
        self.safety_filter = safety_filter
        self.action_clip = action_clip

    def step(self, raw_base_action, residual_request, control_timed_out):
        FrozenPolicySafetyBridge._require_finite("raw policy action", raw_base_action)
        FrozenPolicySafetyBridge._require_finite("residual request", residual_request)
        if raw_base_action.shape != residual_request.shape or raw_base_action.shape[1] != 12:
            raise ValueError("base actions and residual requests must both be [num_envs, 12]")
        base = self.action_clip(raw_base_action)
        state = self.adapter.safety_state(control_timed_out)
        FrozenPolicySafetyBridge.validate_state(state, base)
        before = self.safety_filter.temporal_snapshot(base)
        decision = self.safety_filter.project(base, residual_request, state)
        FrozenPolicySafetyBridge._require_finite("executed safety action", decision.action)
        return ResidualSafetyTrace(
            raw_policy_action=raw_base_action.detach().clone(), base_action=base.detach().clone(),
            residual_request=residual_request.detach().clone(),
            requested_action=(base + residual_request).detach().clone(),
            executed_action=decision.action.detach().clone(), decision=decision,
            filter_state_before={k: v.detach().clone() for k, v in before.items()}, safety_state=state)
