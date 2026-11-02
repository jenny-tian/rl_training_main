"""Tensor-only Source episode endings and reset/transition accounting.

Task failures end individual episodes. Invalid tensors raise before changing
memory. Completion is reset-relative progress >=9m and 50 consecutive stable
control steps; the 2000-step limit remains a truncation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_training.online_adaptation.safety import SafetyLimits
from rl_training.online_adaptation.types import SafetyState
from .bridge import FrozenPolicySafetyBridge


@dataclass(frozen=True)
class SourceRunSeeds:
    run: int

    def __post_init__(self):
        if type(self.run) is not int or self.run not in (1, 2, 3):
            raise ValueError("Source run must be 1, 2 or 3")

    @property
    def ppo(self):
        return 130000 + self.run

    @property
    def geometry(self):
        return (30000, 33000, 36000)[self.run - 1]

    @property
    def sampler(self):
        return 131000 + self.run

    @property
    def reset_state_id(self):
        # Deterministic resets: this is an identifier, not an RNG seed consumed
        # by the environment. Never reseed global Torch after actor creation.
        return (40000, 43000, 46000)[self.run - 1]


class SourceEpisodeLifecycle:
    def __init__(self, num_envs, device, limits: SafetyLimits):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.limits = limits
        self.hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.contact_steps = torch.zeros_like(self.hold)
        self.completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.unsafe = torch.zeros_like(self.completed)
        self.timeout = torch.zeros_like(self.completed)

    def update(self, state: SafetyState, progress, episode_steps):
        action = torch.empty(self.num_envs, 12, device=self.device)
        # The shared validator performs no filter or lifecycle memory mutation.
        FrozenPolicySafetyBridge.validate_state(state, action)
        if progress.shape != (self.num_envs,) or progress.device != self.device:
            raise ValueError("progress must have one row per environment on the lifecycle device")
        if not bool(torch.isfinite(progress).all()):
            raise FloatingPointError("progress contains a non-finite value")
        if (episode_steps.shape != progress.shape or episode_steps.device != self.device
                or episode_steps.dtype != torch.long or bool((episode_steps < 1).any())):
            raise ValueError("episode_steps must contain positive int64 counters")
        contact = state.nonfoot_contact[:, 0]
        self.contact_steps = torch.where(contact, self.contact_steps + 1, 0)
        self.unsafe = (
            (state.projected_gravity_xy.norm(dim=-1) > self.limits.max_projected_gravity_xy)
            | (state.base_height[:, 0] < self.limits.min_base_height)
            | (state.joint_vel.abs().amax(dim=-1) > self.limits.max_joint_velocity)
            | (self.contact_steps > self.limits.nonfoot_contact_grace_steps)
            | state.control_timed_out[:, 0]
        )
        stable = ~self.unsafe & ~contact
        self.hold = torch.where((progress >= 9.0) & stable, self.hold + 1, 0)
        self.completed = self.hold >= 50
        self.timeout = episode_steps >= 2000

    def reset(self, env_ids):
        for value in (self.hold, self.contact_steps, self.completed, self.unsafe, self.timeout):
            value[env_ids] = 0


class SourceExposureLedger:
    """Count old-cell transitions separately from new-episode assignments."""
    def __init__(self, labels, device):
        self.labels = tuple(labels)
        self.resets = torch.zeros(len(labels), dtype=torch.long, device=device)
        self.transitions = torch.zeros_like(self.resets)

    def record(self, columns, *, reset=False):
        if (columns.ndim != 1 or columns.dtype != torch.long
                or columns.device != self.resets.device
                or bool(((columns < 0) | (columns >= len(self.labels))).any())):
            raise ValueError("invalid Source exposure column indices")
        counts = torch.bincount(columns, minlength=len(self.labels))
        if reset:
            self.resets += counts
        else:
            self.transitions += counts

    def report(self):
        return {name: {"episode_assignments": int(self.resets[i]),
                       "transitions": int(self.transitions[i])}
                for i, name in enumerate(self.labels)}
