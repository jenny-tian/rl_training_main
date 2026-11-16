"""Current-model safety memory excludes corrections to the frozen base action.

The legacy filter and its historical experiments remain unchanged. Thresholds,
physical joint projection, pause/resume, contact guards and residual amplitude
limits are inherited. Only the residual slew-memory semantics are corrected.
"""
from __future__ import annotations

import torch

from rl_training.online_adaptation.safety import SafetyFilter


def projected_base_action(base_action, state, *, paused=False):
    """Same physical projection as SafetyFilter; exact identity when feasible."""
    target = state.default_joint_pos + state.action_scale * base_action
    projected = torch.maximum(torch.minimum(target, state.joint_upper - .02), state.joint_lower + .02)
    if (state.action_scale.abs() <= 1e-8).any():
        raise ValueError('current model requires nonzero action scales')
    physical = torch.where(projected != target,
                           (projected - state.default_joint_pos) / state.action_scale, base_action)
    # Match the inherited action's floating-point arithmetic too. The native
    # filter recomposes base + correction except when paused. A different
    # rounding path would itself look like a tiny learned residual.
    recomposed = base_action + (physical - base_action)
    return torch.where(torch.as_tensor(paused, device=base_action.device), physical, recomposed)


class CurrentModelSafetyFilter(SafetyFilter):
    """Remember bounded learned-residual effects, not base-limit corrections.

    Each call still executes P(base + gated/slew-limited request). The memory
    for the next call becomes P(base + residual) - P(base), rather than
    P(base + residual) - base. Thus a zero request never acquires a spurious
    residual solely because a previous base target required projection.

    The returned decision.residual retains legacy total-delta semantics, so
    audit code can still see the full correction to the raw base. Consumers
    must use execution_components() when analysing the learned effect.
    """
    SEMANTICS = 'projected_action_minus_projected_base_v1'

    def __init__(self, limits):
        super().__init__(limits)
        self._base_correction = None
        self._learned_effect = None

    def project(self, base_action, raw_residual, state, *, force_zero_residual=False):
        if base_action.shape != raw_residual.shape:
            raise ValueError('base and residual shapes differ')
        decision = super().project(base_action, raw_residual, state,
                                   force_zero_residual=force_zero_residual)
        safe_base = projected_base_action(base_action, state, paused=decision.paused)
        effect = decision.action - safe_base
        limits = self.limits.residual_limits.to(device=effect.device, dtype=effect.dtype)
        # Projection cannot enlarge the normalized residual relative to the
        # projected base. Reject substantive violations; only trim roundoff.
        if (effect.abs() > limits + 2.e-6).any():
            raise ValueError('learned effect exceeds the registered residual envelope')
        memory = torch.maximum(torch.minimum(effect, limits), -limits)
        self._previous_residual = memory.detach().clone()
        self._base_correction = (safe_base - base_action).detach().clone()
        self._learned_effect = effect.detach().clone()
        return decision

    def execution_components(self):
        if self._learned_effect is None:
            raise RuntimeError('no safety decision recorded')
        return dict(base_correction=self._base_correction.clone(), learned_effect=self._learned_effect.clone())

    def reset_temporal_state(self, env_ids=None):
        super().reset_temporal_state(env_ids)
        if env_ids is None:
            self._base_correction = self._learned_effect = None
        elif self._learned_effect is not None:
            self._base_correction[env_ids] = 0
            self._learned_effect[env_ids] = 0

    def temporal_snapshot(self, base_action):
        value = super().temporal_snapshot(base_action)
        # Tensor marker lets current-model consumers reject legacy snapshots.
        # Historical shadow.py ignores extra fields, so it must NOT be used
        # for this filter; current-model EA must use from_snapshot below.
        value['current_model_memory_v1'] = torch.ones_like(value['filter_paused'], dtype=torch.long)
        return value

    @classmethod
    def from_snapshot(cls, limits, snapshot):
        if 'current_model_memory_v1' not in snapshot or not (snapshot['current_model_memory_v1'] == 1).all():
            raise ValueError('not a current-model safety snapshot')
        instance = cls(limits)
        for name, key in (('_previous_residual', 'filter_previous_residual_action'),
                          ('_paused', 'filter_paused'), ('_stable_steps', 'filter_stable_steps'),
                          ('_contact_steps', 'filter_contact_steps')):
            setattr(instance, name, snapshot[key].clone())
        return instance
