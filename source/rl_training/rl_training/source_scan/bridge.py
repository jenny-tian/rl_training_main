"""Frozen-policy to SafetyFilter bridge used by the Source scan runner.

The bridge is deliberately small and tensor-facing.  It owns no simulator
state and performs no learning; the caller supplies the Isaac Lab adapter and
the already-loaded frozen policy.  Keeping this boundary separate makes it
possible to unit-test action ordering, finite-value checks, and filter traces
without starting Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.types import SafetyDecision, SafetyState


@dataclass(frozen=True)
class PolicySafetyTrace:
    """One frozen-policy control decision and its safety outcome."""

    raw_policy_action: torch.Tensor
    base_action: torch.Tensor
    requested_action: torch.Tensor
    executed_action: torch.Tensor
    decision: SafetyDecision
    filter_state_before: dict[str, torch.Tensor]
    action_latency_ms: float

    def __post_init__(self) -> None:
        shapes = {
            tuple(value.shape)
            for value in (
                self.raw_policy_action,
                self.base_action,
                self.requested_action,
                self.executed_action,
            )
        }
        if len(shapes) != 1:
            raise ValueError(f"policy/safety action shapes differ: {sorted(shapes)}")
        if self.action_latency_ms < 0.0 or not torch.isfinite(torch.tensor(self.action_latency_ms)):
            raise ValueError("action_latency_ms must be a finite nonnegative number")


class FrozenPolicySafetyBridge:
    """Run a frozen policy through the deployment SafetyFilter.

    The first-pass Source scan has no online residual.  Therefore the policy
    output is both the base action and requested action, while the executed
    action is the filter result with ``force_zero_residual=True``.  The trace
    still records all three values so later residual experiments cannot infer
    an executed action by subtraction or silently omit a projection.
    """

    def __init__(
        self,
        policy: Callable[[Any], torch.Tensor],
        adapter: Any,
        safety_filter: Any,
        *,
        action_clip: Callable[[torch.Tensor], torch.Tensor] = clip_lite3_policy_action,
    ) -> None:
        if not callable(policy):
            raise TypeError("policy must be callable")
        if not callable(getattr(adapter, "safety_state", None)):
            raise TypeError("adapter must expose safety_state(control_timed_out)")
        if not callable(getattr(safety_filter, "project", None)):
            raise TypeError("safety_filter must expose project(...)")
        if not callable(getattr(safety_filter, "temporal_snapshot", None)):
            raise TypeError("safety_filter must expose temporal_snapshot(...)")
        if not callable(action_clip):
            raise TypeError("action_clip must be callable")
        self.policy = policy
        self.adapter = adapter
        self.safety_filter = safety_filter
        self.action_clip = action_clip

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 torch tensor")
        if value.is_complex() or not value.is_floating_point():
            raise TypeError(f"{name} must use a real floating-point dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains a non-finite value")

    @classmethod
    def validate_state(cls, state: SafetyState, action: torch.Tensor) -> None:
        """Reject invalid rows before comparisons or filter memory updates.

        NaN comparisons are false, so relying on the filter thresholds alone
        can let an invalid height, tilt or joint velocity through. Only the
        fixed action scale may broadcast across environments.
        """
        batch = action.shape[0]
        for name, width in (
            ("projected_gravity_xy", 2), ("base_height", 1),
            ("joint_pos", 12), ("joint_vel", 12), ("joint_lower", 12),
            ("joint_upper", 12), ("default_joint_pos", 12), ("action_scale", 12),
        ):
            value = getattr(state, name)
            cls._require_finite(name, value)
            allowed = {(batch, width)}
            if name == "action_scale":
                allowed.add((1, width))
            if tuple(value.shape) not in allowed or value.device != action.device:
                raise ValueError(f"{name} shape/device does not match the policy action")
        for name in ("nonfoot_contact", "control_timed_out"):
            value = getattr(state, name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool")
            if value.shape != (batch, 1) or value.device != action.device:
                raise ValueError(f"{name} shape/device does not match the policy action")
        if bool((state.action_scale.abs() <= 1e-8).any().item()):
            raise ValueError("action_scale must be nonzero for every joint")
        if bool((state.joint_lower + 0.02 >= state.joint_upper - 0.02).any().item()):
            raise ValueError("joint_lower/joint_upper leave no feasible safety margin")

    def step(self, observation: Any, control_timed_out: torch.Tensor) -> PolicySafetyTrace:
        started = perf_counter()
        with torch.inference_mode():
            raw_action = self.policy(observation)
        self._require_finite("raw policy action", raw_action)
        if raw_action.shape[1] != 12:
            raise ValueError(f"Lite3 policy action must have 12 columns, got {raw_action.shape[1]}")
        base_action = self.action_clip(raw_action)
        self._require_finite("clipped policy action", base_action)
        if base_action.shape != raw_action.shape or base_action.device != raw_action.device:
            raise ValueError("action clipping changed the policy shape/device")
        if not isinstance(control_timed_out, torch.Tensor):
            raise TypeError("control_timed_out must be a tensor")
        if control_timed_out.shape != (base_action.shape[0], 1):
            raise ValueError(
                "control_timed_out must have shape "
                f"({base_action.shape[0]}, 1), got {tuple(control_timed_out.shape)}"
            )
        if control_timed_out.dtype != torch.bool:
            raise TypeError("control_timed_out must use torch.bool")
        if control_timed_out.device != base_action.device:
            raise ValueError("control_timed_out device must match the policy action")
        state = self.adapter.safety_state(control_timed_out)
        self.validate_state(state, base_action)
        requested_action = base_action.detach().clone()
        filter_state_before = self.safety_filter.temporal_snapshot(base_action)
        decision = self.safety_filter.project(
            base_action,
            torch.zeros_like(base_action),
            state,
            force_zero_residual=True,
        )
        self._require_finite("executed safety action", decision.action)
        if decision.action.shape != requested_action.shape:
            raise ValueError("SafetyFilter changed the action shape")
        if decision.paused.shape != (base_action.shape[0], 1):
            raise ValueError("SafetyFilter pause mask must have shape [batch, 1]")
        if decision.projected.shape != (base_action.shape[0], 1):
            raise ValueError("SafetyFilter projection mask must have shape [batch, 1]")
        if not isinstance(filter_state_before, dict) or not filter_state_before:
            raise TypeError("SafetyFilter temporal_snapshot must return a non-empty tensor mapping")
        for name, value in filter_state_before.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"filter state {name} must be a rank-2 torch tensor")
            if value.is_complex() or not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"filter state {name} contains an invalid value")
        latency_ms = (perf_counter() - started) * 1000.0
        return PolicySafetyTrace(
            raw_policy_action=raw_action.detach().clone(),
            base_action=base_action.detach().clone(),
            requested_action=requested_action,
            executed_action=decision.action.detach().clone(),
            decision=decision,
            filter_state_before={
                name: value.detach().clone() for name, value in filter_state_before.items()
            },
            action_latency_ms=latency_ms,
        )


__all__ = ["FrozenPolicySafetyBridge", "PolicySafetyTrace"]
