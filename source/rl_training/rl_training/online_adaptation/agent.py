from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from numbers import Real

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
    residual_gain: float = 1.0


@dataclass(frozen=True)
class PendingStep:
    history: torch.Tensor
    frame: torch.Tensor
    base_action: torch.Tensor
    residual_action: torch.Tensor
    learning_mask: torch.Tensor
    actor_version: int | None


def _validate_residual_gain(value: object) -> float:
    error_message = "residual_gain must be a real finite scalar in [0, 1]"
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.dtype == torch.bool or value.is_complex():
            raise ValueError(error_message)
        try:
            valid = torch.isfinite(value) & (value >= 0) & (value <= 1)
            if not bool(valid.item()):
                raise ValueError(error_message)
            return float(value.item())
        except (RuntimeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(error_message) from error
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(error_message)
    try:
        if value != value or value < 0 or value > 1:
            raise ValueError(error_message)
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(error_message) from error


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
        self._residual_gain_lock = threading.Lock()
        self._residual_gain = _validate_residual_gain(config.residual_gain)
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
        self.adaptation_replay = ReplayBuffer(
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

    @property
    def residual_gain(self) -> float:
        with self._residual_gain_lock:
            return self._residual_gain

    def set_residual_gain(self, value: float) -> None:
        validated = _validate_residual_gain(value)
        with self._residual_gain_lock:
            self._residual_gain = validated

    def begin_step(
        self,
        frame: torch.Tensor,
        base_action: torch.Tensor,
        safety_state: SafetyState,
        deterministic: bool,
    ) -> SafetyDecision:
        if self._pending is not None:
            raise RuntimeError("finish_step must be called before the next begin_step")
        residual_gain = self.residual_gain
        self.history.append(frame)
        history = self.history.snapshot()
        actor, version = self.publisher.current()
        with torch.no_grad():
            residual, _, _ = actor.sample(history, frame, base_action, deterministic=deterministic)
            residual = residual * residual_gain
        decision = replace(
            self.safety.project(
                base_action,
                residual,
                safety_state,
                force_zero_residual=residual_gain == 0.0,
            ),
            actor_version=version,
        )
        self._pending = PendingStep(
            history=history,
            frame=frame.detach().clone(),
            base_action=base_action.detach().clone(),
            residual_action=decision.residual.detach().clone(),
            learning_mask=(~decision.paused).detach().clone(),
            actor_version=version,
        )
        return decision

    def finish_step(
        self,
        next_frame: torch.Tensor,
        next_base_action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        bootstrap_mask: torch.Tensor,
        discount: torch.Tensor,
        safety_mask: torch.Tensor,
        curriculum_stage_id: torch.Tensor,
        episode_id: torch.Tensor,
        adaptation_enabled: bool = False,
    ) -> None:
        if self._pending is None:
            raise RuntimeError("begin_step must be called before finish_step")
        reset_mask = (terminated.bool() | truncated.bool()).squeeze(-1)
        next_history = self.history.preview(next_frame, reset_mask=reset_mask)
        transition = dict(
            history=self._pending.history,
            current_frame=self._pending.frame,
            base_action=self._pending.base_action,
            residual_action=self._pending.residual_action,
            reward=reward,
            next_history=next_history,
            next_frame=next_frame,
            next_base_action=next_base_action,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask,
            discount=discount,
            safety_mask=safety_mask,
            curriculum_stage_id=curriculum_stage_id,
            episode_id=episode_id,
            valid_mask=self._pending.learning_mask & safety_mask.bool(),
        )
        if adaptation_enabled:
            ReplayBuffer.add_batch_atomically(
                (self.replay, self.adaptation_replay),
                **transition,
            )
        else:
            self.replay.add_batch(**transition)
        reset_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.history.reset(reset_ids)
        self._pending = None

    def fallback_pending(self, safety_state: SafetyState) -> SafetyDecision:
        """Replace the pending residual with the frozen-base fallback after a control timeout."""
        if self._pending is None:
            raise RuntimeError("begin_step must be called before fallback_pending")
        fallback = replace(
            self.safety.project(
                self._pending.base_action,
                torch.zeros_like(self._pending.residual_action),
                safety_state,
                force_zero_residual=True,
            ),
            actor_version=None,
        )
        self._pending = PendingStep(
            history=self._pending.history,
            frame=self._pending.frame,
            base_action=self._pending.base_action,
            residual_action=fallback.residual.detach().clone(),
            learning_mask=(~fallback.paused).detach().clone(),
            actor_version=None,
        )
        return fallback

    def sample_update_batch(self, generator: torch.Generator | None = None) -> TransitionBatch:
        return self.sample_critic_batch(generator=generator)

    def sample_critic_batch(self, generator: torch.Generator | None = None) -> TransitionBatch:
        return self.replay.sample(self.config.batch_size, self.device, generator=generator)

    def sample_actor_batch(self, generator: torch.Generator | None = None) -> TransitionBatch:
        return self.adaptation_replay.sample(self.config.batch_size, self.device, generator=generator)

    def clear_adaptation_replay(self) -> None:
        self.adaptation_replay.clear()

    @staticmethod
    def should_update(replay_size: int, min_replay_size: int, paused: bool) -> bool:
        return replay_size >= min_replay_size and not paused

    @staticmethod
    def should_update_actor(
        replay_size: int,
        min_replay_size: int,
        paused: bool,
        actor_enabled: bool,
    ) -> bool:
        return actor_enabled and replay_size >= min_replay_size and not paused

    @staticmethod
    def all_environments_paused(paused: torch.Tensor) -> bool:
        return bool(paused.bool().all().item())
