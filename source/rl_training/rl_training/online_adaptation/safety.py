from __future__ import annotations

import torch
from dataclasses import dataclass

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
        unsafe = gravity_unsafe | height_unsafe | velocity_unsafe | contact_unsafe | timeout_unsafe

        self._paused |= unsafe
        self._stable_steps = torch.where(
            unsafe,
            torch.zeros_like(self._stable_steps),
            torch.where(self._paused, self._stable_steps + 1, torch.zeros_like(self._stable_steps)),
        )
        resumed = self._paused & (self._stable_steps >= self.limits.stable_steps_to_resume)
        self._paused &= ~resumed
        self._stable_steps = torch.where(resumed, torch.zeros_like(self._stable_steps), self._stable_steps)

        limits = self.limits.residual_limits.to(device=raw_residual.device, dtype=raw_residual.dtype)
        residual = torch.maximum(torch.minimum(raw_residual, limits), -limits)
        residual = torch.maximum(
            torch.minimum(residual, self._previous_residual + self.limits.residual_rate_limit),
            self._previous_residual - self.limits.residual_rate_limit,
        )
        if force_zero_residual:
            residual = torch.zeros_like(residual)
        residual_guard = self._paused | contact_active | tilt_guard
        residual = torch.where(residual_guard, torch.zeros_like(residual), residual)

        normalized_action = base_action + residual
        physical_target = state.default_joint_pos + state.action_scale * normalized_action
        lower = state.joint_lower + 0.02
        upper = state.joint_upper - 0.02
        projected_target = torch.maximum(torch.minimum(physical_target, upper), lower)
        safe_scale = torch.where(state.action_scale.abs() > 1e-8, state.action_scale, torch.ones_like(state.action_scale))
        projected_action = (projected_target - state.default_joint_pos) / safe_scale
        projected_action = torch.where(state.action_scale.abs() > 1e-8, projected_action, normalized_action)
        paused_projected_action = projected_action
        accepted_residual = projected_action - base_action
        accepted_residual = torch.where(self._paused, torch.zeros_like(accepted_residual), accepted_residual)
        if force_zero_residual:
            accepted_residual = torch.where(
                accepted_residual.abs() <= 1.0e-6,
                torch.zeros_like(accepted_residual),
                accepted_residual,
            )
        projected_action = base_action + accepted_residual
        projected_action = torch.where(self._paused, paused_projected_action, projected_action)

        changed = (
            (accepted_residual - raw_residual).abs().amax(dim=-1, keepdim=True) > 1e-7
        ) | ((projected_action - base_action - accepted_residual).abs().amax(dim=-1, keepdim=True) > 1e-7)
        self._previous_residual = accepted_residual.detach().clone()

        reasons: list[str] = []
        for name, mask in (
            ("tilt", gravity_unsafe),
            ("height", height_unsafe),
            ("joint_velocity", velocity_unsafe),
            ("nonfoot_contact", contact_unsafe),
            ("control_timeout", timeout_unsafe),
        ):
            if mask.any().item():
                reasons.append(name)

        return SafetyDecision(
            action=projected_action,
            residual=accepted_residual,
            paused=self._paused.clone(),
            projected=changed,
            reasons=tuple(reasons),
        )
