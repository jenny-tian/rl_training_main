from __future__ import annotations

import math
import torch
from dataclasses import dataclass, fields

from rl_training.lite3_stability_contract import (
    LITE3_MAX_PROJECTED_GRAVITY_XY,
    LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY,
)
from .types import SafetyDecision, SafetyState


@dataclass(frozen=True)
class SafetyLimits:
    residual_limits: torch.Tensor
    residual_rate_limit: float = 0.02
    max_projected_gravity_xy: float = LITE3_MAX_PROJECTED_GRAVITY_XY
    residual_guard_projected_gravity_xy: float = LITE3_TILT_GUARD_PROJECTED_GRAVITY_XY
    min_base_height: float = 0.20
    max_joint_velocity: float = 30.0
    nonfoot_contact_grace_steps: int = 10
    stable_steps_to_resume: int = 25
    risk_gate_enabled: bool = False
    risk_gate_min_scale: float = 0.25
    risk_gate_tilt_start: float = 0.35
    risk_gate_tilt_end: float = 0.65
    risk_gate_height_start: float = 0.28
    risk_gate_height_end: float = 0.20
    risk_gate_joint_velocity_start: float = 15.0
    risk_gate_joint_velocity_end: float = 30.0
    risk_gate_contact_scale: float = 0.5

    def __post_init__(self) -> None:
        bounded = ("risk_gate_min_scale", "risk_gate_contact_scale")
        for name in bounded:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for start, end, name in (
            (self.risk_gate_tilt_start, self.risk_gate_tilt_end, "tilt"),
            (self.risk_gate_joint_velocity_start, self.risk_gate_joint_velocity_end, "joint velocity"),
        ):
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError(f"risk gate {name} thresholds must be finite")
            if start < 0 or end < 0:
                raise ValueError(f"risk gate {name} thresholds must be nonnegative")
            if float(end) <= float(start):
                raise ValueError(f"risk gate {name} end must exceed start")
        for value, name in (
            (self.risk_gate_height_start, "height start"),
            (self.risk_gate_height_end, "height end"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"risk gate {name} must be finite and nonnegative")
        if self.risk_gate_height_end >= self.risk_gate_height_start:
            raise ValueError("risk gate height end must be below start")

    def risk_gate_contract(self) -> dict | None:
        if not self.risk_gate_enabled:
            return None
        return {"semantics": "max_state_risk_residual_envelope_v1", **risk_gate_kwargs(self)}


def risk_gate_kwargs(args) -> dict:
    """Resolve the same defaults for direct callers, workers, and supervisors."""
    return {
        item.name: getattr(args, item.name, item.default)
        for item in fields(SafetyLimits)
        if item.name.startswith("risk_gate_")
    }


class SafetyFilter:
    def __init__(self, limits: SafetyLimits):
        if limits.stable_steps_to_resume <= 0:
            raise ValueError("stable_steps_to_resume must be positive")
        if limits.nonfoot_contact_grace_steps < 0:
            raise ValueError("nonfoot_contact_grace_steps must be non-negative")
        if limits.residual_guard_projected_gravity_xy <= 0:
            raise ValueError("residual_guard_projected_gravity_xy must be positive")
        self.limits = limits
        self._previous_residual: torch.Tensor | None = None
        self._paused: torch.Tensor | None = None
        self._stable_steps: torch.Tensor | None = None
        self._contact_steps: torch.Tensor | None = None

    def _risk_gate(self, state: SafetyState) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a conservative risk score and residual scale per environment.

        The gate gives the learned residual a continuously shrinking trust
        region before the hard safety pause is reached.  The frozen base
        action remains available throughout.
        """
        gravity = state.projected_gravity_xy.norm(dim=-1, keepdim=True)
        tilt = (
            (gravity - self.limits.risk_gate_tilt_start)
            / (self.limits.risk_gate_tilt_end - self.limits.risk_gate_tilt_start)
        ).clamp(0.0, 1.0)
        height = (
            (self.limits.risk_gate_height_start - state.base_height)
            / (self.limits.risk_gate_height_start - self.limits.risk_gate_height_end)
        ).clamp(0.0, 1.0)
        joint_velocity = state.joint_vel.abs().amax(dim=-1, keepdim=True)
        velocity = (
            (joint_velocity - self.limits.risk_gate_joint_velocity_start)
            / (
                self.limits.risk_gate_joint_velocity_end
                - self.limits.risk_gate_joint_velocity_start
            )
        ).clamp(0.0, 1.0)
        contact = state.nonfoot_contact.to(dtype=gravity.dtype) * self.limits.risk_gate_contact_scale
        risk = torch.maximum(torch.maximum(tilt, height), torch.maximum(velocity, contact))
        scale = 1.0 - (1.0 - self.limits.risk_gate_min_scale) * risk
        if not self.limits.risk_gate_enabled:
            scale = torch.ones_like(scale)
        return risk, scale

    def reset_temporal_state(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear all history, or only the rows ending an asynchronous episode."""
        if env_ids is not None:
            if env_ids.ndim != 1 or env_ids.dtype != torch.long:
                raise ValueError("env_ids must be a rank-1 int64 tensor")
            if self._previous_residual is None:
                return
            if env_ids.device != self._previous_residual.device:
                raise ValueError("env_ids must be on the filter device")
            if bool(((env_ids < 0) | (env_ids >= self._previous_residual.shape[0])).any()):
                raise ValueError("env_ids are outside the filter batch")
            for memory in (self._previous_residual, self._paused, self._stable_steps, self._contact_steps):
                memory[env_ids] = 0
            return
        self._previous_residual = None
        self._paused = None
        self._stable_steps = None
        self._contact_steps = None

    def temporal_snapshot(self, base_action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Copy pre-decision filter memory for trace audits."""
        self._ensure_state(base_action)
        return {
            "filter_previous_residual_action": self._previous_residual.detach().clone(),
            "filter_paused": self._paused.detach().clone(),
            "filter_stable_steps": self._stable_steps.detach().clone(),
            "filter_contact_steps": self._contact_steps.detach().clone(),
        }

    def _ensure_state(self, base_action: torch.Tensor) -> None:
        num_envs = base_action.shape[0]
        if self._previous_residual is None or self._previous_residual.shape != base_action.shape:
            self._previous_residual = torch.zeros_like(base_action)
            self._paused = torch.zeros(num_envs, 1, dtype=torch.bool, device=base_action.device)
            self._stable_steps = torch.zeros(num_envs, 1, dtype=torch.long, device=base_action.device)
            self._contact_steps = torch.zeros(num_envs, 1, dtype=torch.long, device=base_action.device)

    def project(
        self,
        base_action: torch.Tensor,
        raw_residual: torch.Tensor,
        state: SafetyState,
        *,
        force_zero_residual: bool = False,
    ) -> SafetyDecision:
        if base_action.shape != raw_residual.shape:
            raise ValueError(f"base and residual shapes differ: {base_action.shape} vs {raw_residual.shape}")
        self._ensure_state(base_action)
        assert self._previous_residual is not None
        assert self._paused is not None
        assert self._stable_steps is not None
        assert self._contact_steps is not None

        risk_score, risk_scale = self._risk_gate(state)

        gravity_unsafe = state.projected_gravity_xy.norm(dim=-1, keepdim=True) > self.limits.max_projected_gravity_xy
        height_unsafe = state.base_height < self.limits.min_base_height
        velocity_unsafe = state.joint_vel.abs().amax(dim=-1, keepdim=True) > self.limits.max_joint_velocity
        contact_active = state.nonfoot_contact.to(dtype=torch.bool)
        tilt_guard = (
            state.projected_gravity_xy.norm(dim=-1, keepdim=True)
            > self.limits.residual_guard_projected_gravity_xy
        )
        self._contact_steps = torch.where(
            contact_active,
            self._contact_steps + 1,
            torch.zeros_like(self._contact_steps),
        )
        contact_unsafe = self._contact_steps > self.limits.nonfoot_contact_grace_steps
        timeout_unsafe = state.control_timed_out.to(dtype=torch.bool)
        persistent_unsafe = gravity_unsafe | height_unsafe | velocity_unsafe | contact_unsafe

        self._paused |= persistent_unsafe
        self._stable_steps = torch.where(
            persistent_unsafe,
            torch.zeros_like(self._stable_steps),
            torch.where(
                timeout_unsafe,
                self._stable_steps,
                torch.where(self._paused, self._stable_steps + 1, torch.zeros_like(self._stable_steps)),
            ),
        )
        resumed = self._paused & (self._stable_steps >= self.limits.stable_steps_to_resume)
        self._paused &= ~resumed
        self._stable_steps = torch.where(resumed, torch.zeros_like(self._stable_steps), self._stable_steps)
        paused = self._paused | timeout_unsafe

        limits = self.limits.residual_limits.to(device=raw_residual.device, dtype=raw_residual.dtype)
        residual = torch.maximum(torch.minimum(raw_residual, limits), -limits)
        residual = residual * risk_scale
        residual = torch.maximum(
            torch.minimum(residual, self._previous_residual + self.limits.residual_rate_limit),
            self._previous_residual - self.limits.residual_rate_limit,
        )
        if self.limits.risk_gate_enabled:
            # A newly reduced safety envelope takes priority over slew history.
            # Recovery still passes through the normal rate limit.
            envelope = limits * risk_scale
            residual = torch.maximum(torch.minimum(residual, envelope), -envelope)
        if force_zero_residual:
            residual = torch.zeros_like(residual)
        residual_guard = paused | contact_active | tilt_guard
        residual = torch.where(residual_guard, torch.zeros_like(residual), residual)

        normalized_action = base_action + residual
        physical_target = state.default_joint_pos + state.action_scale * normalized_action
        lower = state.joint_lower + 0.02
        upper = state.joint_upper - 0.02
        projected_target = torch.maximum(torch.minimum(physical_target, upper), lower)
        safe_scale = torch.where(state.action_scale.abs() > 1e-8, state.action_scale, torch.ones_like(state.action_scale))
        projected_action = (projected_target - state.default_joint_pos) / safe_scale
        projected_action = torch.where(state.action_scale.abs() > 1e-8, projected_action, normalized_action)
        # Preserve the requested normalized action exactly when the physical
        # target was already feasible. An unnecessary inverse transform adds
        # ~1e-7 rounding noise that can change contact dynamics, and used to
        # differ between zero-gain SAC and zero-increment LoRA.
        joint_projected = projected_target != physical_target
        projected_action = torch.where(joint_projected, projected_action, normalized_action)
        paused_projected_action = projected_action
        accepted_residual = torch.where(joint_projected, projected_action - base_action, residual)
        accepted_residual = torch.where(paused, torch.zeros_like(accepted_residual), accepted_residual)
        projected_action = base_action + accepted_residual
        projected_action = torch.where(paused, paused_projected_action, projected_action)

        # The normalized-action -> physical-target -> normalized-action round
        # trip introduces float32 error around 1e-7.  Treating that numerical
        # noise as a safety intervention invalidates otherwise on-policy rows.
        projection_tolerance = 1.0e-6
        changed = (
            (accepted_residual - raw_residual).abs().amax(dim=-1, keepdim=True)
            > projection_tolerance
        ) | (
            (projected_action - base_action - accepted_residual).abs().amax(dim=-1, keepdim=True)
            > projection_tolerance
        )
        self._previous_residual = accepted_residual.detach().clone()

        reason_masks = (
            ("tilt", gravity_unsafe),
            ("height", height_unsafe),
            ("joint_velocity", velocity_unsafe),
            ("nonfoot_contact", contact_unsafe),
            ("control_timeout", timeout_unsafe),
        )
        # One device-to-host transfer replaces five synchronizing scalar reads.
        flags = torch.stack([mask.any() for _, mask in reason_masks]).cpu().tolist()
        reasons = tuple(name for (name, _), active in zip(reason_masks, flags) if active)

        return SafetyDecision(
            action=projected_action,
            residual=accepted_residual,
            paused=paused.clone(),
            projected=changed,
            reasons=reasons,
            risk_score=risk_score.detach().clone(),
            risk_scale=risk_scale.detach().clone(),
        )
