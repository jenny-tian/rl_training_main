"""A compact, version-safe online PPO updater.

This implementation is intentionally independent of RSL-RL's runner.  It can
be used with a small actor containing LoRA layers and a newly initialized
deployable critic, while preserving the control process's immutable policy
version during an update.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from .ppo_rollout import PPORollout


@dataclass(frozen=True)
class PPOConfig:
    clip_eps: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Online LoRA updates see much less data than the original flat PPO run.
    # A conservative default limits destructive policy jumps after a terrain
    # transition; callers can still opt into a larger rate explicitly.
    actor_lr: float = 5e-5
    critic_lr: float = 5e-5
    epochs: int = 2
    minibatch_size: int = 64
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    flat_anchor_coef: float = 0.0
    max_anchor_mse: float | None = None
    max_grad_norm: float = 1.0
    target_kl: float | None = 0.01
    normalize_advantages: bool = True
    max_parameter_abs: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.clip_eps < 1.0):
            raise ValueError("clip_eps must be in (0, 1)")
        if not (0.0 < self.gamma <= 1.0) or not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        for name in ("actor_lr", "critic_lr"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if isinstance(self.minibatch_size, bool) or self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.target_kl is not None and (not math.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be positive or None")
        if not math.isfinite(self.flat_anchor_coef) or self.flat_anchor_coef < 0:
            raise ValueError("flat_anchor_coef must be finite and non-negative")
        if self.max_anchor_mse is not None and (
            not math.isfinite(self.max_anchor_mse) or self.max_anchor_mse <= 0
        ):
            raise ValueError("max_anchor_mse must be positive or None")
        if self.max_parameter_abs is not None and (
            not math.isfinite(self.max_parameter_abs) or self.max_parameter_abs <= 0
        ):
            raise ValueError("max_parameter_abs must be positive or None")


@dataclass(frozen=True)
class PPOUpdateStats:
    accepted: bool
    rejection_reason: str | None
    approx_kl: float
    actor_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    num_samples: int
    policy_version_before: int
    policy_version_after: int
    # Diagnostics use each valid rollout row once, before any optimization or
    # advantage normalization.  num_samples retains its epoch-weighted count.
    value_mean: float | None = None
    return_mean: float | None = None
    advantage_mean: float | None = None
    advantage_std: float | None = None
    explained_variance: float | None = None
    num_valid_samples: int = 0
    total_rollout_samples: int = 0


def _as_column(value: Tensor, name: str) -> Tensor:
    if value.ndim == 1:
        return value.unsqueeze(-1)
    if value.ndim == 2 and value.shape[-1] == 1:
        return value
    raise ValueError(f"{name} must have shape [N] or [N, 1]")


def _distribution_from_policy(policy: nn.Module, observations: Any) -> Distribution:
    """Obtain a torch distribution from common actor interfaces."""

    getter = getattr(policy, "get_distribution", None)
    if callable(getter):
        distribution = getter(observations)
        if isinstance(distribution, Distribution):
            return distribution
    # RSL-RL MLPModel keeps a stateful distribution and updates it during
    # forward.  Calling ``forward(..., stochastic_output=False)`` gives its
    # mean parameters without drawing a second action.
    distribution_module = getattr(policy, "distribution", None)
    if distribution_module is not None and hasattr(policy, "forward"):
        try:
            output = policy(observations, stochastic_output=False)
        except TypeError:
            output = policy(observations)
        if hasattr(distribution_module, "update"):
            distribution_module.update(output)
            current = getattr(distribution_module, "_distribution", None)
            if isinstance(current, Distribution):
                return current
    output = policy(observations)
    if isinstance(output, Distribution):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, Distribution):
                return item
    raise TypeError("policy must expose get_distribution(), a distribution module, or return a torch Distribution")


def _evaluate_actions(policy: nn.Module, observations: Any, actions: Tensor) -> tuple[Tensor, Tensor]:
    evaluator = getattr(policy, "evaluate_actions", None)
    if callable(evaluator):
        result = evaluator(observations, actions)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise TypeError("evaluate_actions must return (log_prob, entropy)")
        log_prob, entropy = result[:2]
        return _as_column(log_prob, "log_prob"), _as_column(entropy, "entropy")
    distribution = _distribution_from_policy(policy, observations)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    if log_prob.ndim > 1:
        log_prob = log_prob.sum(dim=-1)
    if entropy.ndim > 1:
        entropy = entropy.sum(dim=-1)
    return _as_column(log_prob, "log_prob"), _as_column(entropy, "entropy")


def _policy_mean(policy: nn.Module, observations: Any) -> Tensor:
    getter = getattr(policy, "mean_action", None)
    if callable(getter):
        return getter(observations)
    getter = getattr(policy, "get_mean", None)
    if callable(getter):
        return getter(observations)
    distribution = _distribution_from_policy(policy, observations)
    return distribution.mean


def _critic_value(critic: nn.Module, observations: Any) -> Tensor:
    result = critic(observations)
    if isinstance(result, (tuple, list)):
        result = result[0]
    if not isinstance(result, Tensor):
        raise TypeError("critic must return a tensor")
    return _as_column(result, "critic value")


class OnlinePPOUpdater:
    """Update only trainable policy adapters and a separate critic.

    The caller must start a fresh rollout after a successful update.  A
    rollout carrying a different policy version is rejected before any
    optimizer step, which prevents stale/transition samples from becoming
    on-policy training data.
    """

    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        config: PPOConfig | None = None,
        *,
        policy_version: int = 0,
    ) -> None:
        if not isinstance(policy, nn.Module) or not isinstance(critic, nn.Module):
            raise TypeError("policy and critic must be torch modules")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        self.policy = policy
        self.critic = critic
        self.config = config or PPOConfig()
        self.policy_version = policy_version
        actor_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        if not actor_parameters:
            raise ValueError("policy has no trainable parameters; inject LoRA or unfreeze an adapter first")
        critic_parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
        if not critic_parameters:
            raise ValueError("critic has no trainable parameters")
        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.config.critic_lr)

    def _check_rollout(self, rollout: PPORollout) -> None:
        if not isinstance(rollout, PPORollout):
            raise TypeError("update expects a PPORollout")
        versions = rollout.version_tensor()
        if versions.numel() == 0 or not bool((versions == self.policy_version).all().item()):
            raise ValueError(
                f"rollout policy version {versions.unique().tolist()} does not match current version {self.policy_version}"
            )
        for name in ("old_log_probs", "old_values", "rewards", "next_values"):
            value = getattr(rollout, name)
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"rollout {name} contains non-finite values")

    def _snapshot(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            copy.deepcopy(self.policy.state_dict()),
            copy.deepcopy(self.critic.state_dict()),
            copy.deepcopy(self.actor_optimizer.state_dict()),
            copy.deepcopy(self.critic_optimizer.state_dict()),
        )

    def _flat_anchor_loss(self, observations: Any, means: Tensor) -> Tensor:
        current = _policy_mean(self.policy, observations)
        if current.shape != means.shape:
            raise ValueError(
                f"flat anchor means shape {tuple(means.shape)} does not match policy mean {tuple(current.shape)}"
            )
        return (current - means.detach()).square().mean()

    def _restore(self, snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
        policy_state, critic_state, actor_opt, critic_opt = snapshot
        self.policy.load_state_dict(policy_state)
        self.critic.load_state_dict(critic_state)
        self.actor_optimizer.load_state_dict(actor_opt)
        self.critic_optimizer.load_state_dict(critic_opt)

    def state_dict(self) -> dict[str, Any]:
        """Return an independent checkpoint, including policy version."""

        return {
            "schema_version": 1,
            "policy_version": self.policy_version,
            "policy": copy.deepcopy(self.policy.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
            "critic_optimizer": copy.deepcopy(self.critic_optimizer.state_dict()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint atomically and reject incompatible versions."""

        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("unsupported online PPO checkpoint schema")
        version = state.get("policy_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("online PPO policy_version must be a non-negative integer")
        required = {"policy", "critic", "actor_optimizer", "critic_optimizer"}
        if set(state) != {"schema_version", "policy_version", *required}:
            raise ValueError("online PPO checkpoint keys are incompatible")
        snapshot = self._snapshot()
        previous_version = self.policy_version
        try:
            self.policy.load_state_dict(state["policy"], strict=True)
            self.critic.load_state_dict(state["critic"], strict=True)
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.policy_version = version
            if any(not bool(torch.isfinite(p).all().item()) for p in self.policy.parameters()) or any(
                not bool(torch.isfinite(p).all().item()) for p in self.critic.parameters()
            ):
                raise ValueError("online PPO checkpoint contains non-finite parameters")
        except BaseException:
            self._restore(snapshot)
            self.policy_version = previous_version
            raise

    def update(
        self,
        rollout: PPORollout,
        *,
        flat_anchor_observations: Any | None = None,
        flat_anchor_means: Tensor | None = None,
    ) -> PPOUpdateStats:
        self._check_rollout(rollout)
        if rollout.length == 0:
            raise ValueError("cannot update from an empty rollout")
        if (flat_anchor_observations is None) != (flat_anchor_means is None):
            raise ValueError("flat_anchor_observations and flat_anchor_means must be supplied together")
        if (self.config.flat_anchor_coef or self.config.max_anchor_mse is not None) and flat_anchor_means is None:
            raise ValueError("anchor constraints require flat anchor observations and means")

        advantages, returns = rollout.compute_returns(self.config.gamma, self.config.gae_lambda)
        if rollout.valid_mask is not None:
            valid = rollout.valid_mask.to(device=advantages.device, dtype=advantages.dtype)
        else:
            valid = torch.ones_like(advantages)
        if not bool((valid.sum() > 0).item()):
            raise ValueError("rollout contains no valid rows")
        # Saved values are the critic predictions made while collecting this
        # block.  Compute diagnostics before normalizing advantages or changing
        # the critic, so rejected candidates report the same input statistics.
        valid_rows = valid.bool()
        diagnostic_values = rollout.old_values.detach()[valid_rows].to(dtype=torch.float64)
        diagnostic_returns = returns.detach()[valid_rows].to(dtype=torch.float64)
        diagnostic_advantages = advantages.detach()[valid_rows].to(dtype=torch.float64)
        return_variance = float(diagnostic_returns.var(unbiased=False))
        explained_variance = None
        if math.isfinite(return_variance) and return_variance > 0.0:
            residual_variance = float((diagnostic_returns - diagnostic_values).var(unbiased=False))
            candidate_explained_variance = 1.0 - residual_variance / return_variance
            if math.isfinite(candidate_explained_variance):
                explained_variance = candidate_explained_variance
        diagnostics = {
            "value_mean": float(diagnostic_values.mean()),
            "return_mean": float(diagnostic_returns.mean()),
            "advantage_mean": float(diagnostic_advantages.mean()),
            "advantage_std": float(diagnostic_advantages.std(unbiased=False)),
            "explained_variance": explained_variance,
            "num_valid_samples": diagnostic_advantages.numel(),
            "total_rollout_samples": rollout.length,
        }
        if self.config.normalize_advantages:
            selected = advantages[valid.bool()]
            advantages = (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)

        snapshot = self._snapshot()
        old_log_probs = rollout.old_log_probs.detach()
        total_actor = total_value = total_entropy = total_clip = 0.0
        total_count = 0
        # A deterministic permutation per epoch is sufficient; random state is
        # captured by the session checkpoint in the caller.
        for _ in range(self.config.epochs):
            indices = torch.randperm(rollout.length, device=rollout.observations.device)
            for start in range(0, rollout.length, self.config.minibatch_size):
                batch_indices = indices[start : start + self.config.minibatch_size]
                batch_valid = valid[batch_indices, 0] > 0.5
                if not bool(batch_valid.any().item()):
                    continue
                idx = batch_indices[batch_valid]
                obs = rollout.observations[idx]
                actions = rollout.actions[idx]
                new_log_prob, entropy = _evaluate_actions(self.policy, obs, actions)
                ratio = torch.exp(new_log_prob - old_log_probs[idx])
                batch_adv = advantages[idx]
                unclipped = ratio * batch_adv
                clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * batch_adv
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                anchor_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
                if flat_anchor_observations is not None and self.config.flat_anchor_coef:
                    anchor_loss = self._flat_anchor_loss(flat_anchor_observations, flat_anchor_means)
                actor_loss = policy_loss - self.config.entropy_coef * entropy.mean() + self.config.flat_anchor_coef * anchor_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.config.max_grad_norm:
                    nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], self.config.max_grad_norm)
                self.actor_optimizer.step()

                value = _critic_value(self.critic, obs)
                value_loss = 0.5 * (value - returns[idx].detach()).square().mean() * self.config.value_coef
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                if self.config.max_grad_norm:
                    nn.utils.clip_grad_norm_([p for p in self.critic.parameters() if p.requires_grad], self.config.max_grad_norm)
                self.critic_optimizer.step()

                count = int(idx.numel())
                total_actor += float(actor_loss.detach()) * count
                total_value += float(value_loss.detach()) * count
                total_entropy += float(entropy.mean().detach()) * count
                total_clip += float((ratio - 1.0).abs().gt(self.config.clip_eps).float().mean().detach()) * count
                total_count += count

        if total_count == 0:
            self._restore(snapshot)
            raise ValueError("rollout contains no valid rows")

        # Re-evaluate all rows under the candidate policy.  The squared
        # log-prob change is stable for both Gaussian and categorical actors.
        with torch.no_grad():
            new_log_prob, _ = _evaluate_actions(self.policy, rollout.observations, rollout.actions)
            valid_rows = valid.bool()
            approx_kl = float(0.5 * (new_log_prob - old_log_probs).square()[valid_rows].mean())
            anchor_mse = (
                float(self._flat_anchor_loss(flat_anchor_observations, flat_anchor_means))
                if flat_anchor_observations is not None
                else None
            )
            finite = all(bool(torch.isfinite(parameter).all().item()) for parameter in self.policy.parameters()) and all(
                bool(torch.isfinite(parameter).all().item()) for parameter in self.critic.parameters()
            )
            within_range = self.config.max_parameter_abs is None or all(
                bool(parameter.detach().abs().max().item() <= self.config.max_parameter_abs)
                for parameter in list(self.policy.parameters()) + list(self.critic.parameters())
            )
        reason: str | None = None
        if not finite:
            reason = "non_finite_parameters"
        elif not within_range:
            reason = "parameter_limit"
        elif (
            self.config.max_anchor_mse is not None
            and anchor_mse is not None
            and anchor_mse > self.config.max_anchor_mse
        ):
            reason = "anchor_mse"
        elif self.config.target_kl is not None and approx_kl > self.config.target_kl:
            reason = "target_kl"
        if reason is not None:
            self._restore(snapshot)
            return PPOUpdateStats(
                False, reason, approx_kl, total_actor / total_count, total_value / total_count,
                total_entropy / total_count, total_clip / total_count, total_count,
                self.policy_version, self.policy_version,
                **diagnostics,
            )
        previous = self.policy_version
        self.policy_version += 1
        return PPOUpdateStats(
            True, None, approx_kl, total_actor / total_count, total_value / total_count,
            total_entropy / total_count, total_clip / total_count, total_count,
            previous, self.policy_version,
            **diagnostics,
        )


__all__ = ["PPOConfig", "PPOUpdateStats", "OnlinePPOUpdater"]
