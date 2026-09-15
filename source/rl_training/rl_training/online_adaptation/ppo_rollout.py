"""On-policy rollout records and advantage computation for online PPO.

The dataclass intentionally stores both termination types.  A true terminal
state has no value bootstrap, while a time-limit truncation bootstraps from
the observation immediately before reset and still cuts the GAE recursion at
the episode boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _column(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.unsqueeze(-1)
    elif value.ndim != 2 or value.shape[-1] != 1:
        raise ValueError(f"{name} must have shape [T] or [T, 1], got {tuple(value.shape)}")
    return value


def _binary(value: Tensor, name: str, length: int) -> Tensor:
    value = _column(value, name)
    if value.shape[0] != length:
        raise ValueError(f"{name} length must match rollout length {length}")
    if value.is_complex() or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain finite binary values")
    if not bool(((value == 0) | (value == 1)).all().item()):
        raise ValueError(f"{name} must contain binary values")
    return value.to(dtype=torch.float32)


@dataclass(frozen=True)
class PPORollout:
    """A contiguous on-policy block.

    ``next_values[t]`` is evaluated on the terminal observation for a
    truncation and on the ordinary next observation otherwise.  Samples from
    different policy versions must be split into separate blocks before an
    update.
    """

    observations: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    next_values: Tensor
    policy_version: int | Tensor = 0
    valid_mask: Tensor | None = None
    action_semantics: str = "requested"

    def __post_init__(self) -> None:
        if not isinstance(self.observations, Tensor) or self.observations.ndim < 1:
            raise ValueError("observations must be a tensor with a time dimension")
        length = self.observations.shape[0]
        for name in ("actions", "old_log_probs", "old_values", "rewards", "next_values"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.shape[0] != length:
                raise ValueError(f"{name} must have the same leading length as observations")
        if self.actions.ndim < 2:
            raise ValueError("actions must have shape [T, action_dim]")
        for name in ("observations", "actions"):
            value = getattr(self, name)
            if value.is_complex() or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain finite real values")
        for name in ("old_log_probs", "old_values", "rewards", "next_values"):
            value = _column(getattr(self, name), name)
            object.__setattr__(self, name, value)
        term = _binary(self.terminated, "terminated", length)
        trunc = _binary(self.truncated, "truncated", length)
        if bool((term.bool() & trunc.bool()).any().item()):
            raise ValueError("a rollout row cannot be both terminated and truncated")
        object.__setattr__(self, "terminated", term)
        object.__setattr__(self, "truncated", trunc)
        if self.valid_mask is not None:
            mask = _binary(self.valid_mask, "valid_mask", length)
            object.__setattr__(self, "valid_mask", mask)
        if self.action_semantics != "requested":
            raise ValueError("PPO old log-probabilities must describe requested actions")
        if isinstance(self.policy_version, Tensor):
            version = self.policy_version.reshape(-1)
            if version.numel() not in (1, length):
                raise ValueError("policy_version tensor must be scalar or have one value per row")
            if not bool(torch.isfinite(version.to(dtype=torch.float32)).all().item()):
                raise ValueError("policy_version must be finite")
            object.__setattr__(self, "policy_version", version.to(dtype=torch.int64))
        elif isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int):
            raise TypeError("policy_version must be an integer or tensor")

    @property
    def length(self) -> int:
        return int(self.observations.shape[0])

    def version_tensor(self) -> Tensor:
        if isinstance(self.policy_version, Tensor):
            value = self.policy_version
            if value.numel() == 1:
                return value.expand(self.length)
            return value
        return torch.full((self.length,), self.policy_version, dtype=torch.int64, device=self.observations.device)

    def compute_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[Tensor, Tensor]:
        return compute_gae(
            self.rewards,
            self.old_values,
            self.next_values,
            self.terminated,
            self.truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute GAE and value targets with correct timeout handling."""

    if not (0.0 < gamma <= 1.0) or not (0.0 <= gae_lambda <= 1.0):
        raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
    rewards, values, next_values = (_column(v, n) for v, n in ((rewards, "rewards"), (values, "values"), (next_values, "next_values")))
    length = rewards.shape[0]
    if length == 0:
        raise ValueError("cannot compute GAE for an empty rollout")
    if any(v.shape[0] != length for v in (values, next_values)):
        raise ValueError("rewards, values, and next_values must have equal length")
    terminated = _binary(terminated, "terminated", length)
    truncated = _binary(truncated, "truncated", length)
    if bool((terminated.bool() & truncated.bool()).any().item()):
        raise ValueError("a row cannot be both terminated and truncated")
    # Bootstrap at time limits, but never after a true terminal failure.
    bootstrap = 1.0 - terminated
    # GAE recursion must stop at either kind of episode boundary.
    continuation = 1.0 - torch.maximum(terminated, truncated)
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for index in range(length - 1, -1, -1):
        delta = rewards[index] + gamma * bootstrap[index] * next_values[index] - values[index]
        running = delta + gamma * gae_lambda * continuation[index] * running
        advantages[index] = running
    return advantages, advantages + values


__all__ = ["PPORollout", "compute_gae"]
