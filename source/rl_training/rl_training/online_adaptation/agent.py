from __future__ import annotations

import copy
import threading
from collections import deque
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, fields, replace
from numbers import Real
from typing import Any

import torch

from .history import HistoryBuffer
from .networks import ResidualActor, TwinCritic
from .nstep import NStepAccumulator, RawTransition
from .publisher import DoubleBufferedActorPublisher
from .replay_buffer import ReplayBuffer, sample_stratified_batch
from .sac import SACConfig, SACUpdater
from .safety import SafetyFilter, SafetyLimits
from .types import SafetyDecision, SafetyState, TransitionBatch, UpdateStats


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
    n_step: int = 1
    gamma: float = 0.99
    flat_fraction_initial: float = 0.75
    flat_fraction_ready: float = 0.25
    online_ready_size: int = 1024
    flat_critic_warmup_updates: int = 1000
    critic_replay_mode: str = "dual"

    def __post_init__(self) -> None:
        for name in ("flat_fraction_initial", "flat_fraction_ready"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not torch.isfinite(torch.tensor(float(value))).item()
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a finite scalar in [0, 1]")
        if type(self.online_ready_size) is not int or self.online_ready_size <= 0:
            raise ValueError("online_ready_size must be a positive int")
        if (
            type(self.flat_critic_warmup_updates) is not int
            or self.flat_critic_warmup_updates < 0
        ):
            raise ValueError("flat_critic_warmup_updates must be a nonnegative int")
        if self.critic_replay_mode not in {"online-only", "dual"}:
            raise ValueError("critic_replay_mode must be online-only or dual")


@dataclass(frozen=True)
class PendingStep:
    history: torch.Tensor
    frame: torch.Tensor
    base_action: torch.Tensor
    residual_action: torch.Tensor
    learning_mask: torch.Tensor
    terrain_valid: torch.Tensor
    actor_version: int | None
    requested_residual_action: torch.Tensor
    mean_residual_action: torch.Tensor


def _normalize_replay_state_for_load(state: Mapping[str, Any]) -> dict[str, Any]:
    """Move replay storage tensors to CPU before ReplayBuffer validation."""
    if not isinstance(state, Mapping):
        raise TypeError("replay state must be a mapping")
    normalized: dict[str, Any] = {}
    for name, value in state.items():
        try:
            detached = value.detach()
        except AttributeError:
            normalized[name] = value
            continue
        try:
            cpu_value = detached.to(device="cpu")
        except (AttributeError, RuntimeError, TypeError) as error:
            raise ValueError(f"replay state {name} cannot be moved to CPU") from error
        if not isinstance(cpu_value, torch.Tensor):
            raise TypeError(f"replay state {name} must normalize to a tensor")
        normalized[name] = cpu_value.clone()
    return normalized


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
        if config.gamma != sac_config.gamma:
            raise ValueError(
                "AgentConfig.gamma must match SACConfig.gamma so n-step replay "
                f"discounts and SAC targets stay consistent: {config.gamma!r} != {sac_config.gamma!r}"
            )
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
        self.flat_replay: ReplayBuffer | None = None
        self._last_flat_batch_fraction = 0.0
        self.flat_critic_warmup_complete = config.flat_critic_warmup_updates == 0
        self.flat_critic_warmup_update_count = 0
        self.safety = SafetyFilter(safety_limits)
        self.updater = SACUpdater(actor, critic, sac_config)
        self.publisher = DoubleBufferedActorPublisher(actor)
        self._pending: PendingStep | None = None
        self._last_committed_global = 0
        self._last_committed_adaptation = 0
        self._nstep = NStepAccumulator(num_envs=num_envs, n_step=config.n_step, gamma=config.gamma)
        self._nstep_valid_queues: list[deque[torch.Tensor]] = [deque() for _ in range(num_envs)]
        self._nstep_adaptation_queues: list[deque[torch.Tensor]] = [deque() for _ in range(num_envs)]

    @property
    def residual_gain(self) -> float:
        with self._residual_gain_lock:
            return self._residual_gain

    def set_residual_gain(self, value: float) -> None:
        validated = _validate_residual_gain(value)
        with self._residual_gain_lock:
            self._residual_gain = validated

    @property
    def last_committed_transition_counts(self) -> tuple[int, int]:
        """Counts committed by the most recent :meth:`finish_step` call."""
        return self._last_committed_global, self._last_committed_adaptation

    def reset_for_deployment_handoff(self) -> None:
        """Clear temporal state that must never cross a physical reset."""
        self.history.reset()
        self._pending = None
        self._last_committed_global = 0
        self._last_committed_adaptation = 0
        self.safety.reset_temporal_state()
        self._nstep.clear()
        for queue in self._nstep_valid_queues:
            queue.clear()
        for queue in self._nstep_adaptation_queues:
            queue.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "residual_gain": self.residual_gain,
            "updater": self.updater.state_dict(),
            "replay": self.replay.state_dict(),
            "adaptation_replay": self.adaptation_replay.state_dict(),
            "nstep": self._nstep.state_dict(),
            "nstep_valid_queues": self._mask_queue_state(self._nstep_valid_queues),
            "nstep_adaptation_queues": self._mask_queue_state(
                self._nstep_adaptation_queues
            ),
            "flat_critic_warmup_complete": self.flat_critic_warmup_complete,
            "flat_critic_warmup_update_count": self.flat_critic_warmup_update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected_keys = set(self.state_dict())
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("agent state has an unsupported schema")
        if state["config"] != self.config:
            raise ValueError("agent configuration differs from checkpoint")
        residual_gain = _validate_residual_gain(state["residual_gain"])
        warmup_complete = state["flat_critic_warmup_complete"]
        warmup_count = state["flat_critic_warmup_update_count"]
        if type(warmup_complete) is not bool:
            raise ValueError("flat critic warmup completion state must be boolean")
        if (
            type(warmup_count) is not int
            or not 0 <= warmup_count <= self.config.flat_critic_warmup_updates
            or warmup_complete != (warmup_count == self.config.flat_critic_warmup_updates)
        ):
            raise ValueError("flat critic warmup progress is inconsistent")
        valid_queues = self._validate_mask_queue_state(state["nstep_valid_queues"])
        adaptation_queues = self._validate_mask_queue_state(
            state["nstep_adaptation_queues"]
        )
        replay_state = _normalize_replay_state_for_load(state["replay"])
        adaptation_replay_state = _normalize_replay_state_for_load(
            state["adaptation_replay"]
        )
        if replay_state.get("frozen") is not False or adaptation_replay_state.get("frozen") is not False:
            raise ValueError("agent online replay buffers must remain mutable")
        before = self.state_dict()
        # Module state_dict tensors alias live parameters; replay/n-step already
        # return independent copies, but the updater needs an explicit snapshot.
        before["updater"] = copy.deepcopy(before["updater"])
        try:
            self.updater.load_state_dict(state["updater"])
            self.replay.load_state_dict(replay_state)
            self.adaptation_replay.load_state_dict(adaptation_replay_state)
            self._nstep.load_state_dict(state["nstep"])
            self._nstep_valid_queues = valid_queues
            self._nstep_adaptation_queues = adaptation_queues
            self.set_residual_gain(residual_gain)
            self.flat_critic_warmup_complete = warmup_complete
            self.flat_critic_warmup_update_count = warmup_count
        except BaseException:
            self.updater.load_state_dict(before["updater"])
            self.replay.load_state_dict(before["replay"])
            self.adaptation_replay.load_state_dict(before["adaptation_replay"])
            self._nstep.load_state_dict(before["nstep"])
            self._nstep_valid_queues = self._validate_mask_queue_state(
                before["nstep_valid_queues"]
            )
            self._nstep_adaptation_queues = self._validate_mask_queue_state(
                before["nstep_adaptation_queues"]
            )
            self.set_residual_gain(before["residual_gain"])
            self.flat_critic_warmup_complete = before["flat_critic_warmup_complete"]
            self.flat_critic_warmup_update_count = before[
                "flat_critic_warmup_update_count"
            ]
            raise
        self.publisher.publish(self.updater.actor, self.updater.version)
        self.history.reset()
        self._pending = None
        self.safety._previous_residual = None
        self.safety._paused = None
        self.safety._stable_steps = None
        self.safety._contact_steps = None

    @staticmethod
    def _mask_queue_state(
        queues: list[deque[torch.Tensor]],
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple(
            tuple(mask.detach().to(device="cpu").clone() for mask in queue)
            for queue in queues
        )

    def _validate_mask_queue_state(
        self, state: object
    ) -> list[deque[torch.Tensor]]:
        if not isinstance(state, tuple) or len(state) != self._nstep.num_envs:
            raise ValueError("agent n-step mask state must contain one queue per environment")
        queues: list[deque[torch.Tensor]] = []
        for queue in state:
            if not isinstance(queue, tuple) or len(queue) >= self.config.n_step:
                raise ValueError("agent n-step mask queues have invalid lengths")
            loaded: deque[torch.Tensor] = deque()
            for mask in queue:
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.shape != (1, 1)
                    or mask.requires_grad
                ):
                    raise ValueError("agent n-step masks must be detached [1, 1] tensors")
                loaded.append(mask.detach().to(device=self.device).clone())
            queues.append(loaded)
        return queues

    def begin_step(
        self,
        frame: torch.Tensor,
        base_action: torch.Tensor,
        safety_state: SafetyState,
        deterministic: bool,
        terrain_valid: torch.Tensor | None = None,
        base_only: bool = False,
    ) -> SafetyDecision:
        if self._pending is not None:
            raise RuntimeError("finish_step must be called before the next begin_step")
        if not isinstance(frame, torch.Tensor) or frame.ndim != 2:
            raise ValueError("frame must be a [batch, frame_dim] tensor")
        if terrain_valid is None:
            terrain_valid = torch.ones(
                frame.shape[0], 1, dtype=torch.bool, device=frame.device
            )
        elif (
            not isinstance(terrain_valid, torch.Tensor)
            or terrain_valid.shape != (frame.shape[0], 1)
            or terrain_valid.requires_grad
        ):
            raise ValueError("terrain_valid must be a detached [batch, 1] tensor")
        terrain_valid = terrain_valid.detach().to(device=frame.device, dtype=torch.bool)
        batch_terrain_valid = terrain_valid.all(dim=0, keepdim=True)
        residual_gain = self.residual_gain
        self.history.append(frame)
        history = self.history.snapshot()
        actor, version = self.publisher.current()
        with torch.no_grad():
            if base_only:
                residual = torch.zeros_like(base_action)
                mean_residual = residual
            else:
                residual, _, mean_residual = actor.sample(history, frame, base_action, deterministic=deterministic)
            residual = residual * residual_gain
            residual = torch.where(batch_terrain_valid, residual, torch.zeros_like(residual))
            mean_residual = torch.where(batch_terrain_valid, mean_residual * residual_gain, torch.zeros_like(mean_residual))
        decision = replace(
            self.safety.project(
                base_action,
                residual,
                safety_state,
                force_zero_residual=residual_gain == 0.0 or not bool(batch_terrain_valid.item()),
            ),
            actor_version=version if bool(batch_terrain_valid.item()) else None,
        )
        self._pending = PendingStep(
            history=history,
            frame=frame.detach().clone(),
            base_action=base_action.detach().clone(),
            residual_action=decision.residual.detach().clone(),
            requested_residual_action=residual.detach().clone(),
            learning_mask=(~decision.paused).detach().clone(),
            terrain_valid=batch_terrain_valid.expand_as(decision.paused).detach().clone(),
            actor_version=version if bool(batch_terrain_valid.item()) else None,
            mean_residual_action=mean_residual.detach().clone(),
        )
        return decision

    def pending_trajectory_fields(self) -> dict[str, torch.Tensor]:
        """Detached pre-action snapshot; never resample the actor for diagnostics."""
        pending = self._pending
        if pending is None:
            raise RuntimeError("trajectory snapshot requires a pending step")
        return {name: value.detach().clone() for name, value in {
            "history": pending.history,
            "current_frame": pending.frame,
            "base_action": pending.base_action,
            "requested_residual_action": pending.requested_residual_action,
            "requested_action": pending.base_action + pending.requested_residual_action,
            "mean_action": pending.base_action + pending.mean_residual_action,
        }.items()}

    def finish_inference_step(self, *, ended: torch.Tensor) -> None:
        """Advance control history without touching replay or any optimizer."""
        if self._pending is None:
            raise RuntimeError("begin_step must precede inference completion")
        reset_ids = torch.nonzero(ended.bool().reshape(-1), as_tuple=False).squeeze(-1)
        if reset_ids.numel():
            self.history.reset(reset_ids)
        self._pending = None
        self._last_committed_global = 0
        self._last_committed_adaptation = 0

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
        target_reset_mask = reset_mask & ~bootstrap_mask.bool().squeeze(-1)
        next_history = self.history.preview(next_frame, reset_mask=target_reset_mask)
        raw_transition = RawTransition(
            history=self._pending.history,
            current_frame=self._pending.frame,
            base_action=self._pending.base_action,
            residual_action=self._pending.residual_action,
            requested_residual_action=self._pending.requested_residual_action,
            reward=reward,
            next_history=next_history,
            next_frame=next_frame,
            next_base_action=next_base_action,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask,
            safety_mask=safety_mask,
            curriculum_stage_id=curriculum_stage_id,
            episode_id=episode_id,
        )
        valid_mask = self._pending.learning_mask & safety_mask.bool()
        nstep_state = self._nstep.state_dict()
        valid_queue_state = [
            deque(mask.detach().clone() for mask in queue)
            for queue in self._nstep_valid_queues
        ]
        adaptation_queue_state = [
            deque(mask.detach().clone() for mask in queue)
            for queue in self._nstep_adaptation_queues
        ]
        committed_global = 0
        committed_adaptation = 0
        try:
            emitted = self._nstep.append(raw_transition)
            emitted_valid_masks, emitted_adaptation_masks = self._append_nstep_masks(
                valid_mask=valid_mask,
                terrain_valid=self._pending.terrain_valid,
                terminated=terminated,
                truncated=truncated,
                adaptation_enabled=adaptation_enabled,
            )
            transition = self._concatenate_batches(emitted)
            global_valid_mask = self._concatenate_masks(emitted_valid_masks)
            adaptation_valid_mask = self._concatenate_masks(emitted_adaptation_masks)
            if transition is not None and global_valid_mask is not None and adaptation_valid_mask is not None:
                # An n-step target is unusable when any transition in its
                # horizon used a safety fallback.  The accumulator already
                # reduces the per-step safety masks across that horizon; fold
                # the reduced mask into the commit masks so an invalid final
                # step cannot leak a zero-bootstrap target into an otherwise
                # valid prefix transition.
                horizon_safe = transition["safety_mask"].bool()
                committed_global = int((global_valid_mask & horizon_safe).bool().sum().item())
                committed_adaptation = int((adaptation_valid_mask & horizon_safe).bool().sum().item())
                self._add_emitted_transition(
                    transition,
                    global_valid_mask=global_valid_mask & horizon_safe,
                    adaptation_valid_mask=adaptation_valid_mask & horizon_safe,
                )
        except BaseException:
            self._nstep.load_state_dict(nstep_state)
            self._nstep_valid_queues = valid_queue_state
            self._nstep_adaptation_queues = adaptation_queue_state
            raise
        reset_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.history.reset(reset_ids)
        self._pending = None
        self._last_committed_global = committed_global
        self._last_committed_adaptation = committed_adaptation

    def _append_nstep_masks(
        self,
        *,
        valid_mask: torch.Tensor,
        terrain_valid: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        adaptation_enabled: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        valid_emissions: list[list[torch.Tensor]] = []
        adaptation_emissions: list[list[torch.Tensor]] = []
        end_mask = (terminated.bool() | truncated.bool()).reshape(-1)
        adaptation_mask = valid_mask & terrain_valid.bool() & bool(adaptation_enabled)
        for env_id in range(len(self._nstep_valid_queues)):
            valid_queue = self._nstep_valid_queues[env_id]
            adaptation_queue = self._nstep_adaptation_queues[env_id]
            valid_queue.append(valid_mask.detach().narrow(0, env_id, 1).clone())
            adaptation_queue.append(adaptation_mask.detach().narrow(0, env_id, 1).clone())
            emitted_valid: list[torch.Tensor] = []
            emitted_adaptation: list[torch.Tensor] = []
            if bool(end_mask[env_id].item()):
                while valid_queue:
                    emitted_valid.append(valid_queue[0])
                    emitted_adaptation.append(adaptation_queue[0])
                    valid_queue.popleft()
                    adaptation_queue.popleft()
            elif len(valid_queue) >= self.config.n_step:
                emitted_valid.append(valid_queue[0])
                emitted_adaptation.append(adaptation_queue[0])
                valid_queue.popleft()
                adaptation_queue.popleft()
            valid_emissions.append(emitted_valid)
            adaptation_emissions.append(emitted_adaptation)

        return self._merge_masks(valid_emissions), self._merge_masks(adaptation_emissions)

    @staticmethod
    def _merge_masks(per_env_emissions: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        merged: list[torch.Tensor] = []
        max_count = max((len(emissions) for emissions in per_env_emissions), default=0)
        for index in range(max_count):
            rows = [emissions[index] for emissions in per_env_emissions if index < len(emissions)]
            merged.append(torch.cat(rows, dim=0))
        return merged

    def _add_emitted_transition(
        self,
        transition: dict[str, torch.Tensor],
        *,
        global_valid_mask: torch.Tensor,
        adaptation_valid_mask: torch.Tensor,
    ) -> None:
        if not bool(adaptation_valid_mask.bool().any().item()):
            self.replay.add_batch(**transition, valid_mask=global_valid_mask)
            return
        if torch.equal(global_valid_mask.bool(), adaptation_valid_mask.bool()):
            ReplayBuffer.add_batch_atomically(
                (self.replay, self.adaptation_replay),
                **transition,
                valid_mask=global_valid_mask,
            )
            return
        self._add_distinct_replay_masks_atomically(
            transition,
            global_valid_mask=global_valid_mask,
            adaptation_valid_mask=adaptation_valid_mask,
        )

    def _add_distinct_replay_masks_atomically(
        self,
        transition: dict[str, torch.Tensor],
        *,
        global_valid_mask: torch.Tensor,
        adaptation_valid_mask: torch.Tensor,
    ) -> None:
        buffers = (self.replay, self.adaptation_replay)
        lock_order = sorted(buffers, key=id)
        prepared = [
            (self.replay, self.replay._prepare_batch(**transition, valid_mask=global_valid_mask)),
            (
                self.adaptation_replay,
                self.adaptation_replay._prepare_batch(**transition, valid_mask=adaptation_valid_mask),
            ),
        ]
        with ExitStack() as stack:
            for buffer in lock_order:
                stack.enter_context(buffer._lock)
            for buffer in buffers:
                buffer._require_mutable_locked()
            snapshots = [
                (buffer, buffer._snapshot_for_commit(batch)) for buffer, batch in prepared
            ]
            try:
                for buffer, batch in prepared:
                    buffer._commit_prepared(batch)
            except BaseException:
                for buffer, snapshot in reversed(snapshots):
                    buffer._restore_snapshot(snapshot)
                raise

    @staticmethod
    def _concatenate_batches(batches: list[TransitionBatch]) -> dict[str, torch.Tensor] | None:
        if not batches:
            return None
        return {
            field.name: torch.cat([getattr(batch, field.name) for batch in batches], dim=0)
            for field in fields(TransitionBatch)
        }

    @staticmethod
    def _concatenate_masks(masks: list[torch.Tensor]) -> torch.Tensor | None:
        if not masks:
            return None
        return torch.cat(masks, dim=0)

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
            requested_residual_action=torch.zeros_like(fallback.residual),
            mean_residual_action=torch.zeros_like(fallback.residual),
            learning_mask=(~fallback.paused).detach().clone(),
            terrain_valid=self._pending.terrain_valid,
            actor_version=None,
        )
        return fallback

    def sample_update_batch(self, generator: torch.Generator | None = None) -> TransitionBatch:
        return self.sample_critic_batch(generator=generator)

    def sample_critic_batch(self, generator: torch.Generator | None = None) -> TransitionBatch:
        if self.flat_replay is None or self.config.critic_replay_mode == "online-only":
            self._last_flat_batch_fraction = 0.0
            return self.replay.sample(
                self.config.batch_size, self.device, generator=generator
            )
        requested_fraction = (
            self.config.flat_fraction_ready
            if len(self.replay) >= self.config.online_ready_size
            else self.config.flat_fraction_initial
        )
        sampled = sample_stratified_batch(
            flat_replay=self.flat_replay,
            online_replay=self.replay,
            batch_size=self.config.batch_size,
            flat_fraction=requested_fraction,
            device=self.device,
            generator=generator,
        )
        self._last_flat_batch_fraction = sampled.flat_fraction
        return sampled.transitions

    @property
    def last_flat_batch_fraction(self) -> float:
        return self._last_flat_batch_fraction

    def sample_actor_batch(self, generator: torch.Generator | None = None) -> TransitionBatch:
        return self.adaptation_replay.sample(self.config.batch_size, self.device, generator=generator)

    def update(
        self,
        critic_batch: TransitionBatch | None = None,
        actor_batch: TransitionBatch | None = None,
        *,
        update_actor: bool = True,
        generator: torch.Generator | None = None,
    ):
        if critic_batch is None:
            critic_batch = self.sample_critic_batch(generator=generator)
        stats = self.updater.update(
            critic_batch=critic_batch,
            actor_batch=actor_batch,
            residual_gain=self.residual_gain,
            update_actor=update_actor,
        )
        return replace(
            stats,
            flat_batch_fraction=self.last_flat_batch_fraction,
            n_step=self.config.n_step,
        )

    def prewarm_critics_from_flat(
        self, generator: torch.Generator | None = None
    ) -> tuple[UpdateStats, ...]:
        if self.flat_critic_warmup_complete:
            return ()
        if self.flat_replay is None:
            raise RuntimeError("flat replay must be loaded before critic prewarm")
        if not self.flat_replay.frozen:
            raise RuntimeError("flat replay must be frozen before critic prewarm")
        results = []
        remaining = (
            self.config.flat_critic_warmup_updates
            - self.flat_critic_warmup_update_count
        )
        for _ in range(remaining):
            batch = self.flat_replay.sample(
                self.config.batch_size, self.device, generator=generator
            )
            stats = self.updater.update(
                batch,
                update_actor=False,
                residual_gain=self.residual_gain,
            )
            stats = replace(
                stats, flat_batch_fraction=1.0, n_step=self.config.n_step
            )
            if not stats.accepted:
                raise RuntimeError(
                    f"flat critic prewarm update rejected: {stats.rejection_reason}"
                )
            self.flat_critic_warmup_update_count += 1
            results.append(stats)
        self.flat_critic_warmup_complete = True
        return tuple(results)

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
