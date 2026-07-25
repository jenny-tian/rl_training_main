from __future__ import annotations

from dataclasses import dataclass

import torch

from .history import HistoryBuffer
from .networks import ResidualActor, TwinCritic
from .publisher import DoubleBufferedActorPublisher
from .replay_buffer import ReplayBuffer
from .sac import SACConfig, SACUpdater
from .safety import SafetyFilter, SafetyLimits
from .types import SafetyDecision, SafetyState, TransitionBatch


@dataclass(frozen=True)
class AgentConfig:
    history_length: int = 25
    frame_dim: int = 49
    action_dim: int = 12
    replay_capacity: int = 10_000
    min_replay_size: int = 256
    batch_size: int = 128
    updates_per_second: float = 5.0


@dataclass(frozen=True)
class PendingStep:
    history: torch.Tensor
    frame: torch.Tensor
    base_action: torch.Tensor
    residual_action: torch.Tensor
    actor_version: int


class OnlineAdaptationAgent:
    def __init__(
        self,
        *,
        num_envs: int,
        actor: ResidualActor,
        critic: TwinCritic,
        safety_limits: SafetyLimits,
        config: AgentConfig = AgentConfig(),
        sac_config: SACConfig = SACConfig(),
        device: str | torch.device | None = None,
        replay_storage_dtype: torch.dtype = torch.float16,
    ):
        if device is None:
            device = next(actor.parameters()).device
        self.device = torch.device(device)
        self.config = config
        self.history = HistoryBuffer(
            num_envs=num_envs,
            history_length=config.history_length,
            frame_dim=config.frame_dim,
            device=self.device,
        )
        self.replay = ReplayBuffer(
            capacity=config.replay_capacity,
            history_length=config.history_length,
            frame_dim=config.frame_dim,
            action_dim=config.action_dim,
            storage_dtype=replay_storage_dtype,
        )
        self.safety = SafetyFilter(safety_limits)
        self.updater = SACUpdater(actor, critic, sac_config)
        self.publisher = DoubleBufferedActorPublisher(actor)
        self._pending: PendingStep | None = None

    def begin_step(
        self,
        frame: torch.Tensor,
        base_action: torch.Tensor,
        safety_state: SafetyState,
        deterministic: bool,
    ) -> SafetyDecision:
        if self._pending is not None:
            raise RuntimeError("finish_step must be called before the next begin_step")
        self.history.append(frame)
        history = self.history.snapshot()
        actor, version = self.publisher.current()
        with torch.no_grad():
            residual, _, _ = actor.sample(history, frame, base_action, deterministic=deterministic)
        decision = self.safety.project(base_action, residual, safety_state)
        self._pending = PendingStep(
            history=history,
            frame=frame.detach().clone(),
            base_action=base_action.detach().clone(),
            residual_action=decision.residual.detach().clone(),
            actor_version=version,
        )
        return decision

    def finish_step(
        self,
        next_frame: torch.Tensor,
        next_base_action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if self._pending is None:
            raise RuntimeError("begin_step must be called before finish_step")
        reset_mask = done.squeeze(-1).bool()
        next_history = self.history.preview(next_frame, reset_mask=reset_mask)
        self.replay.add_batch(
            history=self._pending.history,
            current_frame=self._pending.frame,
            base_action=self._pending.base_action,
            residual_action=self._pending.residual_action,
            reward=reward,
            next_history=next_history,
            next_frame=next_frame,
            next_base_action=next_base_action,
            done=done.float(),
        )
        done_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.history.reset(done_ids)
        self._pending = None

    def fallback_pending(self, safety_state: SafetyState) -> SafetyDecision:
        """Replace the pending residual with the frozen-base fallback after a control timeout."""
        if self._pending is None:
            raise RuntimeError("begin_step must be called before fallback_pending")
        fallback = self.safety.project(
            self._pending.base_action,
            torch.zeros_like(self._pending.residual_action),
            safety_state,
        )
        self._pending = PendingStep(
            history=self._pending.history,
            frame=self._pending.frame,
            base_action=self._pending.base_action,
            residual_action=fallback.residual.detach().clone(),
            actor_version=self._pending.actor_version,
        )
        return fallback

    def sample_update_batch(self, generator: torch.Generator | None = None) -> TransitionBatch:
        return self.replay.sample(self.config.batch_size, self.device, generator=generator)

    @staticmethod
    def should_update(replay_size: int, min_replay_size: int, paused: bool) -> bool:
        return replay_size >= min_replay_size and not paused
