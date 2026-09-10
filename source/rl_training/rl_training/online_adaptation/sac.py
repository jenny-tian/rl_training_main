from __future__ import annotations

import copy
from collections.abc import Callable
import torch
from dataclasses import dataclass, fields
from math import isfinite, log
from numbers import Real
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from typing import Any

from .metrics import AdaptationPerformanceDecision
from .networks import ResidualActor, TwinCritic
from .types import TransitionBatch, UpdateStats


CandidateActorValidator = Callable[[ResidualActor, int], str | None]


class CandidateActorRejected(RuntimeError):
    pass


class _CriticUpdateRejected(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float = -6.0
    max_grad_norm: float = 1.0
    flat_anchor_weight: float = 0.0
    max_critic_loss: float | None = None
    max_mean_td_error: float | None = None
    max_critic_grad_norm: float | None = None
    target_q_min: float | None = None
    target_q_max: float | None = None
    critic_huber_delta: float | None = None
    deterministic_actor_weight: float = 0.0
    initial_alpha: float = 1.0
    alpha_max: float | None = None

    def __post_init__(self) -> None:
        for name in ("initial_alpha", "alpha_max"):
            value = getattr(self, name)
            if value is None and name == "alpha_max":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.alpha_max is not None and self.initial_alpha > self.alpha_max:
            raise ValueError("initial_alpha must not exceed alpha_max")
        if (
            isinstance(self.flat_anchor_weight, bool)
            or not isinstance(self.flat_anchor_weight, (int, float))
            or not isfinite(self.flat_anchor_weight)
            or self.flat_anchor_weight < 0.0
        ):
            raise ValueError("flat_anchor_weight must be finite and nonnegative")
        for name in (
            "max_critic_loss",
            "max_mean_td_error",
            "max_critic_grad_norm",
            "critic_huber_delta",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive when configured")
        for name in ("target_q_min", "target_q_max"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be finite when configured")
        if (
            self.target_q_min is not None
            and self.target_q_max is not None
            and self.target_q_min >= self.target_q_max
        ):
            raise ValueError("target_q_min must be less than target_q_max")
        if (
            isinstance(self.deterministic_actor_weight, bool)
            or not isinstance(self.deterministic_actor_weight, (int, float))
            or not isfinite(self.deterministic_actor_weight)
            or self.deterministic_actor_weight < 0.0
        ):
            raise ValueError("deterministic_actor_weight must be finite and nonnegative")


def compute_sac_target(
    *,
    reward: torch.Tensor,
    discount: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    next_q: torch.Tensor,
    alpha_log_prob: torch.Tensor,
) -> torch.Tensor:
    return reward + bootstrap_mask * discount * (next_q - alpha_log_prob)


class SACUpdater:
    CHECKPOINT_SCHEMA_VERSION = 1
    CRITIC_ACTION_SEMANTICS = "requested_residual_before_safety_v1"

    def __init__(self, actor: ResidualActor, critic: TwinCritic, config: SACConfig):
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.target_critic.requires_grad_(False)
        self.config = config
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        device = next(actor.parameters()).device
        self.log_alpha = nn.Parameter(torch.tensor(log(config.initial_alpha), device=device))
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=config.alpha_lr)
        self.version = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @staticmethod
    def _batch_is_finite(batch: TransitionBatch) -> bool:
        return all(torch.isfinite(getattr(batch, field.name)).all().item() for field in fields(batch))

    @staticmethod
    def _gradients_are_finite(module: nn.Module) -> bool:
        return all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in module.parameters())

    @staticmethod
    def _parameters_are_finite(module: nn.Module) -> bool:
        return all(torch.isfinite(parameter).all().item() for parameter in module.parameters())

    def _rejected_stats(
        self,
        reason: str,
        *,
        critic_loss: float = 0.0,
        mean_q: float = 0.0,
        grad_norm: float = 0.0,
        mean_target_q: float | None = None,
        mean_td_error: float | None = None,
        mean_unclipped_target_q: float | None = None,
        target_q_clip_fraction: float | None = None,
        mean_entropy_bonus: float | None = None,
    ) -> UpdateStats:
        return UpdateStats(
            critic_loss=critic_loss,
            actor_loss=0.0,
            alpha_loss=0.0,
            alpha=float(self.alpha.detach()),
            mean_q=mean_q,
            grad_norm=grad_norm,
            accepted=False,
            version=self.version,
            critic_updated=False,
            actor_updated=False,
            rejection_reason=reason,
            mean_target_q=mean_target_q,
            mean_td_error=mean_td_error,
            mean_unclipped_target_q=mean_unclipped_target_q,
            target_q_clip_fraction=target_q_clip_fraction,
            mean_entropy_bonus=mean_entropy_bonus,
            mean_flat_q=None,
            mean_online_q=None,
            flat_anchor_loss=None,
            flat_batch_fraction=0.0,
            n_step=1,
        )

    def update(
        self,
        critic_batch: TransitionBatch,
        actor_batch: TransitionBatch | None = None,
        flat_anchor_batch: TransitionBatch | None = None,
        *,
        update_actor: bool = True,
        candidate_validator: CandidateActorValidator | None = None,
        residual_gain: float = 1.0,
    ) -> UpdateStats:
        if (
            isinstance(residual_gain, bool)
            or not isinstance(residual_gain, Real)
            or not isfinite(float(residual_gain))
            or not 0.0 <= float(residual_gain) <= 1.0
        ):
            raise ValueError("residual_gain must be a finite scalar in [0, 1]")
        residual_gain = float(residual_gain)
        if not self._batch_is_finite(critic_batch):
            return self._rejected_stats("nonfinite_batch")
        before = copy.deepcopy(self.state_dict())
        critic_grad_norm = torch.zeros((), device=next(self.critic.parameters()).device)
        if update_actor and actor_batch is None:
            actor_batch = critic_batch

        try:
            with torch.no_grad():
                next_residual, next_log_prob, _ = self.actor.sample(
                    critic_batch.next_history,
                    critic_batch.next_frame,
                    critic_batch.next_base_action,
                )
                next_residual = next_residual * residual_gain
                target_q1, target_q2 = self.target_critic(
                    critic_batch.next_history,
                    critic_batch.next_frame,
                    critic_batch.next_base_action,
                    next_residual,
                )
                target = compute_sac_target(
                    reward=critic_batch.reward,
                    discount=critic_batch.discount,
                    bootstrap_mask=critic_batch.bootstrap_mask,
                    next_q=torch.minimum(target_q1, target_q2),
                    alpha_log_prob=self.alpha.detach() * next_log_prob,
                )
                raw_target = target
                if self.config.target_q_min is not None or self.config.target_q_max is not None:
                    target = target.clamp(
                        min=self.config.target_q_min,
                        max=self.config.target_q_max,
                    )
                target_diagnostics = dict(
                    mean_unclipped_target_q=float(raw_target.mean()),
                    target_q_clip_fraction=float((target != raw_target).float().mean()),
                    mean_entropy_bonus=float((-self.alpha.detach() * next_log_prob).mean()),
                )

            q1, q2 = self.critic(
                critic_batch.history,
                critic_batch.current_frame,
                critic_batch.base_action,
                # Q models a requested residual with the safety filter included
                # in the environment dynamics. Never relabel a request with the
                # different executed action after a rate/joint/tilt intervention.
                critic_batch.requested_residual_action,
            )
            if self.config.critic_huber_delta is None:
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            else:
                critic_loss = F.huber_loss(
                    q1, target, delta=self.config.critic_huber_delta
                ) + F.huber_loss(
                    q2, target, delta=self.config.critic_huber_delta
                )
            if not torch.isfinite(critic_loss).item():
                raise FloatingPointError
            mean_td_error = (target - torch.minimum(q1, q2)).abs().mean()
            if not torch.isfinite(mean_td_error).item():
                raise FloatingPointError
            if (
                self.config.max_critic_loss is not None
                and float(critic_loss.detach()) > self.config.max_critic_loss
            ):
                raise _CriticUpdateRejected("critic_loss_limit")
            if (
                self.config.max_mean_td_error is not None
                and float(mean_td_error.detach()) > self.config.max_mean_td_error
            ):
                raise _CriticUpdateRejected("mean_td_error_limit")
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            if not torch.isfinite(critic_grad_norm).item() or not self._gradients_are_finite(self.critic):
                raise FloatingPointError
            if (
                self.config.max_critic_grad_norm is not None
                and float(critic_grad_norm.detach()) > self.config.max_critic_grad_norm
            ):
                raise _CriticUpdateRejected("critic_grad_norm_limit")
            self.critic_optimizer.step()

            actor_loss = torch.full((), float("nan"), device=q1.device)
            alpha_loss = torch.full((), float("nan"), device=q1.device)
            actor_grad_norm = torch.zeros((), device=q1.device)
            flat_anchor_loss_value: torch.Tensor | None = None
            if update_actor:
                if actor_batch is None or not self._batch_is_finite(actor_batch):
                    raise FloatingPointError
                if (
                    flat_anchor_batch is not None
                    and not self._batch_is_finite(flat_anchor_batch)
                ):
                    raise FloatingPointError
                self.critic.requires_grad_(False)
                residual, log_prob, _ = self.actor.sample(
                    actor_batch.history,
                    actor_batch.current_frame,
                    actor_batch.base_action,
                )
                residual = residual * residual_gain
                actor_q1, actor_q2 = self.critic(
                    actor_batch.history,
                    actor_batch.current_frame,
                    actor_batch.base_action,
                    residual,
                )
                actor_loss = (self.alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()
                if self.config.deterministic_actor_weight > 0.0:
                    deterministic_residual, _, _ = self.actor.sample(
                        actor_batch.history,
                        actor_batch.current_frame,
                        actor_batch.base_action,
                        deterministic=True,
                    )
                    deterministic_residual = deterministic_residual * residual_gain
                    deterministic_q1, deterministic_q2 = self.critic(
                        actor_batch.history,
                        actor_batch.current_frame,
                        actor_batch.base_action,
                        deterministic_residual,
                    )
                    actor_loss = actor_loss - self.config.deterministic_actor_weight * torch.minimum(
                        deterministic_q1, deterministic_q2
                    ).mean()
                if flat_anchor_batch is not None:
                    _, _, flat_mean = self.actor.sample(
                        flat_anchor_batch.history,
                        flat_anchor_batch.current_frame,
                        flat_anchor_batch.base_action,
                        deterministic=True,
                    )
                    flat_mean = flat_mean * residual_gain
                    flat_anchor_loss_value = (
                        self.config.flat_anchor_weight * flat_mean.square().mean()
                    )
                    actor_loss = actor_loss + flat_anchor_loss_value
                if not torch.isfinite(actor_loss).item():
                    raise FloatingPointError
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                if not torch.isfinite(actor_grad_norm).item() or not self._gradients_are_finite(self.actor):
                    raise FloatingPointError
                self.actor_optimizer.step()
                self.critic.requires_grad_(True)

                alpha_loss = -(self.log_alpha * (log_prob + self.config.target_entropy).detach()).mean()
                if not torch.isfinite(alpha_loss).item():
                    raise FloatingPointError
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                if self.log_alpha.grad is None or not torch.isfinite(self.log_alpha.grad).all().item():
                    raise FloatingPointError
                torch.nn.utils.clip_grad_norm_((self.log_alpha,), self.config.max_grad_norm)
                self.alpha_optimizer.step()
                if self.config.alpha_max is not None:
                    with torch.no_grad():
                        self.log_alpha.clamp_(max=log(self.config.alpha_max))

            if not (
                self._parameters_are_finite(self.actor)
                and self._parameters_are_finite(self.critic)
                and torch.isfinite(self.log_alpha).item()
            ):
                raise FloatingPointError

            with torch.no_grad():
                for target_parameter, parameter in zip(
                    self.target_critic.parameters(), self.critic.parameters(), strict=True
                ):
                    target_parameter.lerp_(parameter, self.config.tau)
            if not self._parameters_are_finite(self.target_critic):
                raise FloatingPointError
        except _CriticUpdateRejected as error:
            self.critic.requires_grad_(True)
            self.load_state_dict(before)
            return self._rejected_stats(
                error.reason,
                critic_loss=float(critic_loss.detach()),
                mean_q=float(torch.minimum(q1, q2).mean().detach()),
                grad_norm=float(critic_grad_norm.detach()),
                mean_target_q=float(target.mean().detach()),
                mean_td_error=float(mean_td_error.detach()),
                **target_diagnostics,
            )
        except FloatingPointError:
            self.critic.requires_grad_(True)
            self.load_state_dict(before)
            return self._rejected_stats("nonfinite_update")
        except BaseException:
            self.critic.requires_grad_(True)
            self.load_state_dict(before)
            raise

        if update_actor and candidate_validator is not None:
            try:
                rejection_reason = candidate_validator(self.actor, self.version + 1)
            except BaseException:
                self.critic.requires_grad_(True)
                self.load_state_dict(before)
                raise
            if rejection_reason is not None:
                self.critic.requires_grad_(True)
                self.load_state_dict(before)
                return self._rejected_stats(rejection_reason)

        if update_actor:
            self.version += 1
        grad_norm = max(float(critic_grad_norm.detach()), float(actor_grad_norm.detach()))
        return UpdateStats(
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            alpha_loss=float(alpha_loss.detach()),
            alpha=float(self.alpha.detach()),
            mean_q=float(torch.minimum(q1, q2).mean().detach()),
            grad_norm=grad_norm,
            accepted=True,
            version=self.version,
            critic_updated=True,
            actor_updated=update_actor,
            rejection_reason=None,
            mean_target_q=float(target.mean().detach()),
            mean_td_error=float(mean_td_error.detach()),
            **target_diagnostics,
            mean_flat_q=None,
            mean_online_q=None,
            flat_anchor_loss=(
                None
                if flat_anchor_loss_value is None
                else float(flat_anchor_loss_value.detach())
            ),
            flat_batch_fraction=0.0,
            n_step=1,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "critic_action_semantics": self.CRITIC_ACTION_SEMANTICS,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "version": self.version,
        }

    @classmethod
    def validate_checkpoint_contract(cls, state: dict[str, Any]) -> None:
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] != cls.CHECKPOINT_SCHEMA_VERSION
            or state.get("critic_action_semantics") != cls.CRITIC_ACTION_SEMANTICS
        ):
            raise ValueError(
                "SAC checkpoint has missing or incompatible action semantics/schema; "
                "use a freshly generated initializer for training, or load only "
                "historical actor weights for evaluation"
            )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.validate_checkpoint_contract(state)
        if self.config.alpha_max is not None:
            saved_alpha = state["log_alpha"].exp().item()
            if not isfinite(saved_alpha) or saved_alpha > self.config.alpha_max * (1.0 + 1e-6):
                raise ValueError("checkpoint temperature exceeds alpha_max; create a compatible initializer")
        before = copy.deepcopy(self.state_dict())
        try:
            self._restore_state(state)
        except BaseException:
            self._restore_state(before)
            raise

    def _restore_state(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.log_alpha.device))
        self.version = int(state["version"])
        self.target_critic.requires_grad_(False)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=False))


class StableStateManager:
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        self.patience = patience
        self.min_delta = min_delta
        self._best_score: float | None = None
        self._stable_state: dict[str, Any] | None = None
        self._degradation_count = 0

    def checkpoint(self, updater: SACUpdater) -> None:
        self._stable_state = copy.deepcopy(updater.state_dict())
        self._degradation_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": self._best_score,
            "stable_state": copy.deepcopy(self._stable_state),
            "degradation_count": self._degradation_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected_keys = {
            "patience", "min_delta", "best_score", "stable_state",
            "degradation_count",
        }
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("stable manager state has an unsupported schema")
        if state["patience"] != self.patience or state["min_delta"] != self.min_delta:
            raise ValueError("stable manager configuration differs from checkpoint")
        degradation_count = state["degradation_count"]
        if type(degradation_count) is not int or not 0 <= degradation_count <= self.patience:
            raise ValueError("stable manager degradation_count is invalid")
        if state["stable_state"] is not None:
            SACUpdater.validate_checkpoint_contract(state["stable_state"])
        self._best_score = state["best_score"]
        self._stable_state = copy.deepcopy(state["stable_state"])
        self._degradation_count = degradation_count

    def rollback(self, updater: SACUpdater) -> bool:
        if self._stable_state is None:
            return False
        updater.load_state_dict(copy.deepcopy(self._stable_state))
        self._degradation_count = 0
        return True

    def observe(self, score: float, updater: SACUpdater) -> bool:
        if self._best_score is None or score > self._best_score + self.min_delta:
            self._best_score = score
            self.checkpoint(updater)
            self._degradation_count = 0
            return False

        self._degradation_count += 1
        if self._degradation_count >= self.patience:
            return self.rollback(updater)
        return False

    def observe_adaptation(
        self,
        decision: AdaptationPerformanceDecision,
        updater: SACUpdater,
    ) -> bool:
        if decision.improved:
            self.checkpoint(updater)
            return False
        if decision.should_rollback:
            return self.rollback(updater)
        return False
