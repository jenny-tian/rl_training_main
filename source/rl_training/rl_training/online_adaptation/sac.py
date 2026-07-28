from __future__ import annotations

import copy
import torch
from dataclasses import dataclass, fields
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from typing import Any

from .networks import ResidualActor, TwinCritic
from .types import TransitionBatch, UpdateStats


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float = -6.0
    max_grad_norm: float = 1.0


class SACUpdater:
    def __init__(self, actor: ResidualActor, critic: TwinCritic, config: SACConfig):
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.target_critic.requires_grad_(False)
        self.config = config
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        device = next(actor.parameters()).device
        self.log_alpha = nn.Parameter(torch.zeros((), device=device))
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

    def _rejected_stats(self) -> UpdateStats:
        return UpdateStats(
            critic_loss=float("nan"),
            actor_loss=float("nan"),
            alpha_loss=float("nan"),
            alpha=float(self.alpha.detach()),
            mean_q=float("nan"),
            grad_norm=float("nan"),
            accepted=False,
            version=self.version,
            critic_updated=False,
            actor_updated=False,
        )

    def update(
        self,
        critic_batch: TransitionBatch,
        actor_batch: TransitionBatch | None = None,
        *,
        update_actor: bool = True,
    ) -> UpdateStats:
        if not self._batch_is_finite(critic_batch):
            return self._rejected_stats()
        if update_actor and actor_batch is None:
            actor_batch = critic_batch

        before = copy.deepcopy(self.state_dict())
        try:
            with torch.no_grad():
                next_residual, next_log_prob, _ = self.actor.sample(
                    critic_batch.next_history,
                    critic_batch.next_frame,
                    critic_batch.next_base_action,
                )
                target_q1, target_q2 = self.target_critic(
                    critic_batch.next_history,
                    critic_batch.next_frame,
                    critic_batch.next_base_action,
                    next_residual,
                )
                target = critic_batch.reward + (1.0 - critic_batch.done) * self.config.gamma * (
                    torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
                )

            q1, q2 = self.critic(
                critic_batch.history,
                critic_batch.current_frame,
                critic_batch.base_action,
                critic_batch.residual_action,
            )
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            if not torch.isfinite(critic_loss).item():
                raise FloatingPointError
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            if not torch.isfinite(critic_grad_norm).item() or not self._gradients_are_finite(self.critic):
                raise FloatingPointError
            self.critic_optimizer.step()

            actor_loss = torch.full((), float("nan"), device=q1.device)
            alpha_loss = torch.full((), float("nan"), device=q1.device)
            actor_grad_norm = torch.zeros((), device=q1.device)
            if update_actor:
                if actor_batch is None or not self._batch_is_finite(actor_batch):
                    raise FloatingPointError
                self.critic.requires_grad_(False)
                residual, log_prob, _ = self.actor.sample(
                    actor_batch.history,
                    actor_batch.current_frame,
                    actor_batch.base_action,
                )
                actor_q1, actor_q2 = self.critic(
                    actor_batch.history,
                    actor_batch.current_frame,
                    actor_batch.base_action,
                    residual,
                )
                actor_loss = (self.alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()
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
        except FloatingPointError:
            self.critic.requires_grad_(True)
            self.load_state_dict(before)
            return self._rejected_stats()

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
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "version": self.version,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
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
