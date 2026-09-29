"""Run one uninterrupted Lite3 pass over the held-out flat-to-box track."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from isaaclab.app import AppLauncher
from rl_training.online_adaptation.runner_contract import (
    configure_forward_command, pin_forward_command, pin_forward_observation,
    seeded_initial_reset, actuator_delay_metadata, configure_initial_state, initial_root_state,
)


from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.actor_update import update_and_publish_actor
from rl_training.online_adaptation.metrics import AdaptationPerformance
from rl_training.online_adaptation.novelty import NoveltyGate, NoveltyState
from rl_training.online_adaptation.policy_drift import ActorDriftMonitor, PolicyDrift
from rl_training.online_adaptation.rsl_rl_compat import split_rsl_rl_endings
from rl_training.online_adaptation.terrain_protocol import (
    FlatOnlineProtocol,
    TerrainGeometry,
    apply_geometry_to_env_cfg,
    default_flat_online_protocol,
    development_geometry,
    sealed_geometry,
)
from rl_training.online_adaptation.track_completion import TrackCompletion
from rl_training.online_adaptation.types import UpdateStats
from rl_training.online_adaptation.swing_lift import SwingLiftTracker, rough_swing_reward_active
from rl_training.online_adaptation.training_contract import training_contract, validate_training_contract
from rl_training.online_adaptation.task_objective import resolve_task_objective, qualification_failure
from rl_training.online_adaptation.managed_trial import attach_session

from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser, risk_gate_kwargs


def risk_gate_record(decision) -> dict[str, float | None]:
    risk = getattr(decision, "risk_score", None)
    scale = getattr(decision, "risk_scale", None)
    return {
        "risk_score": float(risk.mean()) if risk is not None else None,
        "risk_score_max": float(risk.max()) if risk is not None else None,
        "risk_scale": float(scale.mean()) if scale is not None else None,
    }


def resolve_control_gain(*, mode, managed_frozen, residual_gain, adapting_residual_gain):
    """Deploy a managed learned candidate at its registered adaptation scale.

    The online waiting-state gain can be zero; reusing it for candidate
    evaluation would silently erase the learned policy's contribution.
    """
    if mode == "base":
        return 0.0
    if managed_frozen and mode == "residual_frozen":
        return adapting_residual_gain
    return residual_gain


def control_is_deterministic(
    *,
    stochastic_control: bool,
    adapting_stochastic_control: bool,
    learning_enabled: bool,
    novelty_state: NoveltyState,
) -> bool:
    adapting_exploration = (
        adapting_stochastic_control
        and learning_enabled
        and novelty_state is NoveltyState.ADAPTING
    )
    return not (stochastic_control or adapting_exploration)


@dataclass(frozen=True)
class AdaptationCommand:
    kind: Literal["start_episode", "checkpoint", "rollback", "fall_reset", "end_episode"]
    episode: int
    reason: str = ""
    completion: threading.Event | None = field(default=None, compare=False, repr=False)


@dataclass
class RuntimeCounters:
    critic_updates: int = 0
    actor_updates: int = 0
    actor_updates_before_novelty: int = 0
    actor_updates_after_novelty: int = 0
    first_actor_update_step: int | None = None
    novelty_entries: int = 0
    rollbacks: int = 0
    performance_rollbacks: int = 0
    safety_rollbacks: int = 0
    fall_resets: int = 0
    state_durations: dict[str, int] = field(default_factory=dict)
    actor_updates_by_episode: dict[int, int] = field(default_factory=dict)
    terrain_invalid_steps: int = 0
    terrain_steps: int = 0
    critic_transitions_committed: int = 0
    actor_transitions_committed: int = 0
    update_attempts: int = 0
    update_rejections: dict[str, int] = field(default_factory=dict)

    def record_state(self, state: str) -> None:
        self.state_durations[state] = self.state_durations.get(state, 0) + 1

    def record_critic_update(self) -> None:
        self.critic_updates += 1

    def record_actor_update(self, *, step: int, adaptation_episode: int) -> None:
        self.actor_updates += 1
        if adaptation_episode > 0:
            self.actor_updates_after_novelty += 1
            self.actor_updates_by_episode[adaptation_episode] = (
                self.actor_updates_by_episode.get(adaptation_episode, 0) + 1
            )
        else:
            self.actor_updates_before_novelty += 1
        if self.first_actor_update_step is None:
            self.first_actor_update_step = step

    def actor_updates_for_episode(self, adaptation_episode: int) -> int:
        return self.actor_updates_by_episode.get(adaptation_episode, 0)

    def record_novelty_entry(self) -> None:
        self.novelty_entries += 1

    def record_rollback(self, reason: str) -> None:
        self.rollbacks += 1
        if reason == "performance_degradation":
            self.performance_rollbacks += 1
        else:
            self.safety_rollbacks += 1

    def record_fall_reset(self) -> None:
        self.fall_resets += 1

    def record_terrain_validity(self, terrain_valid: Any) -> None:
        valid = bool(terrain_valid.bool().all().item())
        self.terrain_steps += 1
        if not valid:
            self.terrain_invalid_steps += 1

    def record_transition_commits(
        self, *, critic_transitions: int, actor_transitions: int
    ) -> None:
        if (
            type(critic_transitions) is not int
            or critic_transitions < 0
            or type(actor_transitions) is not int
            or actor_transitions < 0
        ):
            raise ValueError("committed transition counts must be nonnegative integers")
        self.critic_transitions_committed += critic_transitions
        self.actor_transitions_committed += actor_transitions

    def record_update_result(self, stats: UpdateStats) -> None:
        """Record every non-stale optimizer result for auditable accounting."""
        self.update_attempts += 1
        if not stats.accepted:
            reason = stats.rejection_reason or "unknown"
            self.update_rejections[reason] = self.update_rejections.get(reason, 0) + 1


class TransitionUpdateBudget:
    """Thread-safe update budget charged by committed replay transitions."""

    def __init__(
        self,
        *,
        critic_updates_per_transition: float,
        actor_updates_per_transition: float,
        max_pending_critic_updates: int = 256,
        max_pending_actor_updates: int = 32,
    ) -> None:
        values = (
            critic_updates_per_transition,
            actor_updates_per_transition,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("transition update rates must be finite and nonnegative")
        if type(max_pending_critic_updates) is not int or max_pending_critic_updates <= 0:
            raise ValueError("max_pending_critic_updates must be a positive integer")
        if type(max_pending_actor_updates) is not int or max_pending_actor_updates <= 0:
            raise ValueError("max_pending_actor_updates must be a positive integer")
        self.critic_updates_per_transition = float(critic_updates_per_transition)
        self.actor_updates_per_transition = float(actor_updates_per_transition)
        self.max_pending_critic_updates = max_pending_critic_updates
        self.max_pending_actor_updates = max_pending_actor_updates
        self._critic = 0.0
        self._actor = 0.0
        self._lock = threading.Lock()

    def add(self, *, critic_transitions: int, actor_transitions: int) -> None:
        if type(critic_transitions) is not int or critic_transitions < 0:
            raise ValueError("critic_transitions must be a nonnegative integer")
        if type(actor_transitions) is not int or actor_transitions < 0:
            raise ValueError("actor_transitions must be a nonnegative integer")
        with self._lock:
            self._critic = min(
                float(self.max_pending_critic_updates),
                self._critic + critic_transitions * self.critic_updates_per_transition,
            )
            self._actor = min(
                float(self.max_pending_actor_updates),
                self._actor + actor_transitions * self.actor_updates_per_transition,
            )

    def take_critic(self) -> bool:
        with self._lock:
            if self._critic < 1.0:
                return False
            self._critic -= 1.0
            return True

    def take_actor(self, *, consume_critic: bool = False) -> bool:
        with self._lock:
            if self._actor < 1.0 or (consume_critic and self._critic < 1.0):
                return False
            self._actor -= 1.0
            if consume_critic:
                self._critic -= 1.0
            return True

    def clear_actor(self) -> None:
        """Discard budget tied to the episode-local adaptation replay."""
        with self._lock:
            self._actor = 0.0

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {"critic": self._critic, "actor": self._actor}

    def restore_for_new_trial(self, snapshot) -> None:
        if not isinstance(snapshot, dict) or set(snapshot) != {"critic", "actor"}:
            raise ValueError("invalid saved update budget")
        for key, limit in (("critic", self.max_pending_critic_updates), ("actor", self.max_pending_actor_updates)):
            value = snapshot[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError("invalid saved update budget credit")
        with self._lock:
            self._critic = float(snapshot["critic"])
            # Actor credits belong to the previous physical adaptation episode.
            self._actor = 0.0


@dataclass(frozen=True)
class RuntimeStepResult:
    adaptation_enabled: bool
    clear_adaptation_replay: bool
    commands: tuple[AdaptationCommand, ...]
    record_fields: dict[str, Any]


@dataclass(frozen=True)
class PostStepSnapshot:
    mask: Any
    policy_observation: Any
    frame: Any
    signals: Any
    base_x: Any
    base_y: Any = None
    foot_y: Any = None


def observe_completion_snapshot(
    completion: Any,
    snapshot: PostStepSnapshot,
    *,
    swing_lift: Any,
    velocity_error: float,
    tilt_rms: float,
    environment_done: bool,
    safety_violation: bool,
) -> bool:
    """Qualify the current physical pose before assigning completion reward."""
    if completion is None:
        return False
    completion.observe_pose(
        float(snapshot.base_x.mean().item()),
        float(snapshot.base_y.mean().item()),
        foot_y=snapshot.foot_y.detach().reshape(-1).cpu().tolist(),
        swing_lift=swing_lift.detach().reshape(-1).cpu().tolist(),
        velocity_error=velocity_error,
        tilt_rms=tilt_rms,
        foot_slip=float(snapshot.signals.foot_slip.mean().item()),
    )
    return bool(
        completion.reached
        and completion.qualification_satisfied
        and not environment_done
        and not safety_violation
    )


def replace_base_policy_group(observation: Any, policy_observation: Any) -> Any:
    """Restore a terminal policy group in an RSL-RL observation container."""
    import torch

    if not isinstance(observation, Mapping):
        raise RuntimeError("base policy observation must be a mapping")
    reset_policy_observation = observation.get("policy")
    if not isinstance(reset_policy_observation, torch.Tensor):
        raise RuntimeError(
            "base policy observation must contain a tensor 'policy' group"
        )
    if not isinstance(policy_observation, torch.Tensor):
        raise RuntimeError("terminal policy observation must be a tensor")
    if (
        policy_observation.shape != reset_policy_observation.shape
        or policy_observation.dtype != reset_policy_observation.dtype
        or policy_observation.device != reset_policy_observation.device
    ):
        raise RuntimeError("terminal policy observation must match the reset policy group")
    clone = getattr(observation, "clone", None)
    if not callable(clone):
        raise RuntimeError("base policy observation must support clone()")
    restored = clone()
    restored["policy"] = policy_observation.detach().clone()
    return restored


def select_post_step_snapshot(
    *, done: Any, terminal: PostStepSnapshot | None, post_reset: PostStepSnapshot
) -> PostStepSnapshot:
    """Select pre-reset state for ended rows in the single-environment worker."""
    import torch

    done_column = torch.as_tensor(done).bool().reshape(-1, 1)
    if not bool(done_column.any().item()):
        return post_reset
    if done_column.shape[0] != 1:
        raise RuntimeError("online adaptation worker requires exactly one environment")
    if terminal is None or not torch.equal(
        torch.as_tensor(terminal.mask).bool().reshape(-1, 1), done_column
    ):
        raise RuntimeError("pre-reset terminal capture mask does not match done rows")
    return terminal


def clone_reward_signals(signals: Any) -> Any:
    """Detach simulator-backed reward tensors before automatic reset mutates them."""
    values: dict[str, Any] = {}
    for item in fields(signals):
        value = getattr(signals, item.name)
        values[item.name] = (
            None if value is None else value.detach().clone()
        )
    return replace(signals, **values)


@dataclass(frozen=True)
class TrackCompletionDecision:
    reached: bool
    passed: bool
    should_stop: bool


@dataclass
class PostStepRecoveryRamp:
    """Bound the residual intervention during the first steps after a step."""

    transition_steps: int = 25
    target_gain: float = 0.25
    active: bool = False
    elapsed_steps: int = 0
    start_gain: float = 0.0

    def __post_init__(self) -> None:
        if type(self.transition_steps) is not int or self.transition_steps <= 0:
            raise ValueError("transition_steps must be a positive integer")
        if not math.isfinite(float(self.target_gain)) or not 0.0 <= self.target_gain <= 1.0:
            raise ValueError("target_gain must be finite and in [0, 1]")

    def step(self, *, reached: bool, current_gain: float) -> tuple[float, float, bool]:
        """Return (effective gain, progress, recovery-mask-active)."""
        if not reached:
            return float(current_gain), 0.0, False
        if not self.active:
            self.active = True
            self.elapsed_steps = 0
            self.start_gain = float(current_gain)
        self.elapsed_steps = min(self.elapsed_steps + 1, self.transition_steps)
        progress = self.elapsed_steps / self.transition_steps
        gain = self.start_gain + (self.target_gain - self.start_gain) * progress
        return float(gain), float(progress), self.elapsed_steps < self.transition_steps


def evaluate_track_completion_step(
    completion: TrackCompletion,
    *,
    environment_done: bool,
    safety_violation: bool,
) -> TrackCompletionDecision:
    if type(environment_done) is not bool or type(safety_violation) is not bool:
        raise TypeError("environment_done and safety_violation must be bools")
    # Reaching the X boundary is only one part of a formal traversal.  In
    # strict mode the completion tracker also requires in-lane obstacle
    # interaction, swing evidence, and a stable-gait window.  Keep this
    # helper in lock-step with the worker's stop condition and the summary
    # validator; otherwise a run could be recorded as passed even though the
    # worker correctly refused to stop for a lateral bypass or slide.
    passed = (
        completion.reached
        and completion.qualification_satisfied
        and not environment_done
        and not safety_violation
    )
    return TrackCompletionDecision(
        reached=completion.reached,
        passed=passed,
        should_stop=passed,
    )


def policy_drift_record(drift: PolicyDrift) -> dict[str, float | int]:
    return {
        "policy_residual_mse": drift.residual_mse,
        "policy_joint_abs_change_mean": drift.joint_abs_change_mean,
        "policy_joint_abs_change_max": drift.joint_abs_change_max,
        "policy_parameter_l2": drift.parameter_l2,
        "policy_drift_episode": drift.adaptation_episode,
        "policy_drift_actor_version": drift.actor_version,
        "policy_drift_update_step": drift.update_step,
    }


def step_clearance_reward_active(
    *,
    learning_enabled: bool,
    progress_stalled: bool,
    base_x: float,
    step_edge_x: float | None,
    interaction_distance: float,
) -> bool:
    """Cover stalls immediately before or after the geometric step edge."""
    if not learning_enabled or not progress_stalled or step_edge_x is None:
        return False
    return (
        step_edge_x - interaction_distance
        <= base_x
        < step_edge_x + interaction_distance
    )


def actor_replay_transition_eligible(
    *,
    progress_stalled: bool,
    steps_without_progress: int,
    replay_size: int,
    minimum_replay_size: int,
    stalled_replay_stride: int,
) -> bool:
    if type(progress_stalled) is not bool:
        raise TypeError("progress_stalled must be bool")
    if type(replay_size) is not int or replay_size < 0:
        raise ValueError("replay_size must be a nonnegative integer")
    if type(minimum_replay_size) is not int or minimum_replay_size <= 0:
        raise ValueError("minimum_replay_size must be a positive integer")
    if type(steps_without_progress) is not int or steps_without_progress < 0:
        raise ValueError("steps_without_progress must be a nonnegative integer")
    if type(stalled_replay_stride) is not int or stalled_replay_stride <= 0:
        raise ValueError("stalled_replay_stride must be a positive integer")
    return (
        not progress_stalled
        or replay_size < minimum_replay_size
        or steps_without_progress % stalled_replay_stride == 0
    )


def actor_update_options(args: Any) -> dict[str, float | None]:
    """Return publication-gate options selected by the worker CLI."""
    return {"max_residual_mse": args.max_policy_residual_mse}


def write_actor_update_event(stream: Any, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()


def resolve_worker_geometry(
    terrain_stage: str, terrain_seed: int, *, terrain_split: str
) -> TerrainGeometry:
    if terrain_split == "sealed":
        return sealed_geometry(terrain_seed)
    if terrain_split == "development":
        return development_geometry(terrain_stage, terrain_seed)
    raise ValueError(f"unknown terrain split: {terrain_split}")


def split_worker_endings(step_result: tuple[Any, ...], *, target_semantics: str = "timeout-aware"):
    import torch

    if len(step_result) == 5:
        terminated = step_result[2]
        truncated = step_result[3]
        if (
            not isinstance(terminated, torch.Tensor)
            or not isinstance(truncated, torch.Tensor)
            or terminated.shape != truncated.shape
            or terminated.is_complex()
            or truncated.is_complex()
            or not bool(torch.isfinite(terminated).all().item())
            or not bool(torch.isfinite(truncated).all().item())
            or not bool(((terminated == 0) | (terminated == 1)).all().item())
            or not bool(((truncated == 0) | (truncated == 1)).all().item())
            or bool((terminated.bool() & truncated.bool()).any().item())
        ):
            raise RuntimeError("cannot distinguish termination from truncation")
        done = terminated.bool() | truncated.bool()
        try:
            resolved_terminated, resolved_truncated, bootstrap_mask = (
                split_rsl_rl_endings(done, {"time_outs": truncated})
            )
        except RuntimeError as error:
            raise RuntimeError(
                "cannot distinguish termination from truncation"
            ) from error
    elif len(step_result) == 4:
        try:
            resolved_terminated, resolved_truncated, bootstrap_mask = (
                split_rsl_rl_endings(step_result[2], step_result[3])
            )
        except RuntimeError as error:
            raise RuntimeError(
                "cannot distinguish termination from truncation"
            ) from error
    else:
        raise RuntimeError("cannot distinguish termination from truncation")
    if target_semantics == "legacy-done":
        bootstrap_mask = 1.0 - (
            resolved_terminated.bool() | resolved_truncated.bool()
        ).to(dtype=torch.float32)
    elif target_semantics != "timeout-aware":
        raise ValueError("target_semantics must be legacy-done or timeout-aware")
    return (
        resolved_terminated,
        resolved_truncated,
        bootstrap_mask.to(dtype=torch.float32),
    )


def termination_reasons_from_step_result(
    step_result: tuple[Any, ...],
) -> tuple[str, ...]:
    """Extract Isaac Lab termination terms preserved in the step extras."""
    if len(step_result) == 5:
        extras = step_result[4]
    elif len(step_result) == 4:
        extras = step_result[3]
    else:
        return ()
    if not isinstance(extras, Mapping):
        return ()
    log = extras.get("log", extras)
    if not isinstance(log, Mapping):
        return ()
    prefix = "Episode_Termination/"
    reasons: list[str] = []
    for key, value in log.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        try:
            scalar = value.item() if hasattr(value, "item") else value
            active = float(scalar) >= 0.5
        except (TypeError, ValueError):
            continue
        if active:
            reasons.append(key[len(prefix) :])
    return tuple(reasons)


CONTACT_TERMINATION_PROTOCOL = "sustained_nonfoot_contact_5n_3steps_v1"


def write_episode_summary(path: str | Path, summary: dict[str, Any]) -> None:
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def actor_version_for_record(decision: Any) -> int | None:
    return decision.actor_version


def rollback_record_fields(current_reason: str, last_reason: str) -> dict[str, str]:
    return {
        "rollback_reason": current_reason,
        "last_rollback_reason": last_reason,
    }


def prepare_rollback_fallback(
    pending_completion: threading.Event | None,
) -> tuple[threading.Event | None, bool]:
    if pending_completion is None:
        return None, False
    if pending_completion.is_set():
        return None, False
    return pending_completion, True


def update_pause_event(
    paused_event: Any, *, paused: bool, control_timed_out: bool
) -> None:
    """Latch physical safety pauses, but never latch an isolated timeout fallback."""
    if control_timed_out:
        paused_event.clear()
    elif paused:
        paused_event.set()
    else:
        paused_event.clear()


def enforce_prepared_rollback_fallback(
    *,
    online_agent: Any,
    decision: Any,
    safety_state: Any,
    fallback_required: bool,
) -> tuple[Any, bool]:
    if not fallback_required:
        return decision, False
    return online_agent.fallback_pending(safety_state), True


def enforce_control_timeout_fallback(*, online_agent: Any, timed_out_state: Any) -> Any:
    return online_agent.fallback_pending(timed_out_state)


def transition_episode_publication(
    *,
    shared_lock: Any,
    actor_gate_event: Any,
    read_active_episode: Callable[[], int],
    write_active_episode: Callable[[int], None],
    next_episode: int,
    after_transition: Callable[[int], None] | None = None,
) -> int:
    with shared_lock:
        previous_episode = read_active_episode()
        actor_gate_event.clear()
        write_active_episode(next_episode)
        if after_transition is not None:
            after_transition(previous_episode)
        return previous_episode


def close_active_episode_for_worker_boundary(
    *,
    active_episode: int,
    shared_lock: Any,
    actor_gate_event: Any,
    read_active_episode: Callable[[], int],
    write_active_episode: Callable[[int], None],
    enqueue_command: Callable[[AdaptationCommand], None],
    novelty_gate: Any,
    adaptation_performance: Any,
) -> Any | None:
    """Close worker-local adaptation state before a checkpointed boundary."""
    if active_episode > 0:
        transition_episode_publication(
            shared_lock=shared_lock,
            actor_gate_event=actor_gate_event,
            read_active_episode=read_active_episode,
            write_active_episode=write_active_episode,
            next_episode=0,
            after_transition=lambda episode: enqueue_command(
                AdaptationCommand("end_episode", episode)
            ),
        )
    else:
        with shared_lock:
            actor_gate_event.clear()
            write_active_episode(0)
    adaptation_performance.start_episode()
    return novelty_gate.force_cooldown()


def enable_actor_gate_if_active(
    *,
    shared_lock: Any,
    actor_gate_event: Any,
    read_active_episode: Callable[[], int],
    adaptation_episode: int,
) -> bool:
    with shared_lock:
        if read_active_episode() != adaptation_episode:
            return False
        actor_gate_event.set()
        return True


def cutoff_before_final_step(
    *,
    step: int,
    max_steps: int,
    active_episode: int,
    actor_gate_event: Any,
) -> tuple[int, bool]:
    if step != max_steps:
        return active_episode, False
    actor_gate_event.clear()
    return active_episode, True


def apply_adaptation_command(
    command: AdaptationCommand,
    *,
    checkpointed_episode: int,
    updater: Any,
    publisher: Any,
    stable_manager: Any,
    drift_monitor: ActorDriftMonitor,
    actor_gate_event: Any,
    clear_adaptation_replay: Callable[[], None],
    enable_actor_if_episode_active: Callable[[int], bool],
    clear_actor_update_budget: Callable[[], None] | None = None,
) -> tuple[int, str | None]:
    def clear_episode_local_state() -> None:
        clear_adaptation_replay()
        if clear_actor_update_budget is not None:
            clear_actor_update_budget()

    if command.kind == "start_episode":
        clear_episode_local_state()
        stable_manager.checkpoint(updater)
        drift_monitor.start_episode(updater.actor, adaptation_episode=command.episode)
        checkpointed_episode = command.episode
        enable_actor_if_episode_active(command.episode)
    elif command.kind == "checkpoint":
        if command.episode == checkpointed_episode:
            stable_manager.checkpoint(updater)
    elif command.kind == "rollback":
        if command.episode != checkpointed_episode:
            if command.completion is not None:
                command.completion.set()
            return checkpointed_episode, None
        if not stable_manager.rollback(updater):
            raise RuntimeError("stable rollback unavailable")
        publisher.publish(updater.actor, updater.version)
        drift_monitor.clear_episode(adaptation_episode=command.episode)
        clear_episode_local_state()
        checkpointed_episode = 0
        actor_gate_event.clear()
        if command.completion is not None:
            command.completion.set()
        return checkpointed_episode, command.reason
    elif command.kind == "fall_reset":
        # A fall ends the physical trial, but is not sufficient evidence that
        # all previously learned residual behavior is bad. Preserve the live
        # updater and only reset the episode-local adaptation state.
        if command.episode != checkpointed_episode:
            if command.completion is not None:
                command.completion.set()
            return checkpointed_episode, None
        publisher.publish(updater.actor, updater.version)
        drift_monitor.clear_episode(adaptation_episode=command.episode)
        clear_episode_local_state()
        checkpointed_episode = 0
        actor_gate_event.clear()
        if command.completion is not None:
            command.completion.set()
        return checkpointed_episode, "fall_reset"
    elif command.kind == "end_episode":
        if command.episode != checkpointed_episode:
            return checkpointed_episode, None
        drift_monitor.clear_episode(adaptation_episode=command.episode)
        clear_episode_local_state()
        checkpointed_episode = 0
        actor_gate_event.clear()
    return checkpointed_episode, None


def record_update_result(
    *,
    stats: UpdateStats,
    policy_drift: PolicyDrift | None,
    last_update: dict[str, Any],
    counters: RuntimeCounters,
    step: int,
    adaptation_episode: int,
) -> dict[str, Any] | None:
    if stats.rejection_reason == "stale_adaptation_episode":
        return None
    counters.record_update_result(stats)
    update_record = asdict(stats)
    if not stats.actor_updated:
        for field in ("actor_loss", "alpha_loss"):
            if not math.isfinite(float(update_record[field])):
                update_record[field] = None
    actor_event = None
    if stats.accepted and stats.actor_updated:
        assert policy_drift is not None, "accepted Actor update requires policy drift"
        drift_record = policy_drift_record(policy_drift)
        assert all(
            math.isfinite(float(value)) for value in drift_record.values()
        ), "accepted Actor update requires finite policy drift"
        actor_event = {**update_record, **drift_record}
    last_update.clear()
    last_update.update(update_record)
    if policy_drift is not None:
        last_update.update(policy_drift_record(policy_drift))
    if stats.accepted and stats.critic_updated:
        counters.record_critic_update()
    if stats.accepted and stats.actor_updated:
        counters.record_actor_update(step=step, adaptation_episode=adaptation_episode)
    return actor_event


def record_target_critic_update(*, stats, records, last_update, counters,
                                step, adaptation_episode, path, residual_gain=None):
    """Use the common null semantics for uncomputed actor/temperature losses."""
    record_update_result(stats=stats, policy_drift=None, last_update=last_update,
                         counters=counters, step=step, adaptation_episode=adaptation_episode)
    record = dict(last_update)
    if residual_gain is not None:
        record["residual_gain"] = float(residual_gain)
    serialized = json.dumps(record, sort_keys=True, allow_nan=False)
    with Path(path).open("a") as stream:
        stream.write(serialized + "\n")
    records.append(record)


def resolve_track_boundary_x(env_cfg: Any) -> float:
    if hasattr(env_cfg, "track_boundary_x"):
        return float(env_cfg.track_boundary_x)
    return float(getattr(env_cfg, "heldout_start_x", -7.0))


def build_track_completion(
    env_cfg: Any,
    *,
    stage_name: str,
    initial_x: float,
    initial_y: float | None = None,
    protocol: FlatOnlineProtocol,
) -> TrackCompletion:
    """Build reset-relative completion geometry from the generated track cfg."""
    try:
        generator = env_cfg.scene.terrain.terrain_generator
        track_length = generator.size[0]
        sub_terrains = generator.sub_terrains
        if "flat_online" in sub_terrains:
            flat_online = sub_terrains["flat_online"]
            step_length = flat_online.step_length
        elif any(name in sub_terrains for name in ("continuous_rough", "grass", "rocky_road")):
            flat_online = next(
                sub_terrains[name]
                for name in ("continuous_rough", "grass", "rocky_road")
                if name in sub_terrains
            )
            step_length = (
                flat_online.transition_length
                if hasattr(flat_online, "transition_length")
                else flat_online.step_length
            )
        else:
            raise KeyError("no supported online terrain subterrain")
        flat_length = flat_online.flat_length
        lane_width = getattr(flat_online, "lane_width", None)
    except (AttributeError, KeyError, IndexError, TypeError) as error:
        raise ValueError("env_cfg does not expose FlatOnline track geometry") from error
    track_width = float(generator.size[1])
    if lane_width is None:
        lane_width = track_width
    return TrackCompletion.start(
        stage_name=stage_name,
        initial_x=initial_x,
        flat_length=flat_length,
        step_length=step_length,
        track_length=track_length,
        step_clearance_margin_m=protocol.step_clearance_margin_m,
        track_end_buffer_m=protocol.track_end_buffer_m,
        completion_mode=protocol.completion_mode,
        progress_cost_version=protocol.progress_cost_version,
        initial_y=initial_y,
        lane_width_m=float(lane_width) if initial_y is not None else None,
        track_width_m=track_width if initial_y is not None else None,
        lateral_margin_m=protocol.lateral_margin_m,
        required_swing_lift_m=protocol.required_swing_lift_m,
        min_challenge_observation_steps=protocol.min_challenge_observation_steps,
        min_challenge_swing_steps=protocol.min_challenge_swing_steps,
        min_stable_completion_steps=protocol.min_stable_completion_steps,
        max_completion_velocity_error=protocol.max_completion_velocity_error,
        max_completion_tilt_rms=protocol.max_completion_tilt_rms,
        max_completion_foot_slip=protocol.max_completion_foot_slip,
        edge_slip_margin_m=protocol.edge_slip_margin_m,
    )


def start_control_timing(
    *, device: Any, clock: Callable[[], float] = time.perf_counter
) -> float:
    """Return a control timestamp after draining asynchronous device work."""
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
    return clock()


@contextmanager
def control_phase(
    *, control_update_lock: Any, device: Any, clock: Callable[[], float] = time.perf_counter
):
    """Keep learner device work out of the measured control path."""
    with control_update_lock:
        yield start_control_timing(device=device, clock=clock)


def prepare_control_inputs(
    *, adapter: Any, previous_action: Any, device: Any
) -> tuple[Any, Any, Any]:
    """Materialize simulator-backed inputs before starting the control deadline."""
    import torch

    frame = adapter.frame(previous_action)
    timeout = torch.zeros(frame.shape[0], 1, dtype=torch.bool, device=device)
    safety_state = adapter.safety_state(timeout)
    return frame, timeout, safety_state


def validate_terrain_checkpoint_metadata(
    saved_metadata: Mapping[str, object] | None,
    current_metadata: Mapping[str, object],
) -> None:
    """Reject checkpoints whose terrain feature contract differs from runtime."""
    current_source = current_metadata.get("terrain_feature_source")
    if not isinstance(current_source, str):
        raise ValueError("current terrain metadata has no valid feature source")
    if saved_metadata is None:
        if current_source != "geometry":
            raise ValueError(
                "experiment checkpoint lacks terrain metadata and cannot be resumed in height_scan mode"
            )
        return
    if not isinstance(saved_metadata, Mapping):
        raise ValueError("experiment checkpoint terrain metadata is invalid")
    if dict(saved_metadata) != dict(current_metadata):
        raise ValueError("experiment checkpoint terrain metadata differs from the current runtime")




def normalized_forward_progress(
    delta_x: Any,
    *,
    forward_speed: float,
    step_dt: float,
    valid_mask: Any | None = None,
) -> Any:
    """Convert reset-safe world-X progress to a bounded reward signal."""
    import torch

    if not isinstance(delta_x, torch.Tensor):
        raise TypeError("delta_x must be a torch.Tensor")
    nominal_step = max(abs(float(forward_speed)) * float(step_dt), 0.01)
    progress = torch.nan_to_num(delta_x / nominal_step, nan=0.0, posinf=1.0, neginf=-1.0)
    progress = progress.clamp(-1.0, 1.0)
    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != progress.shape:
            raise ValueError("valid_mask must match delta_x shape")
        progress = torch.where(valid_mask.bool(), progress, torch.zeros_like(progress))
    return progress


def warm_up_control_path(
    *,
    base_policy: Any,
    online_agent: Any,
    adapter: Any,
    observation: Any,
    previous_action: Any,
    safety_limits: Any,
    safety_filter_cls: Any,
    deterministic: bool,
    device: Any,
) -> None:
    import torch

    with torch.inference_mode():
        frame = adapter.frame(previous_action)
        base_action = clip_lite3_policy_action(base_policy(observation))
        actor, _ = online_agent.publisher.current()
        history = torch.zeros(
            (frame.shape[0], online_agent.config.history_length, frame.shape[-1]),
            dtype=frame.dtype,
            device=frame.device,
        )
        actor.sample(history, frame, base_action, deterministic=deterministic)
        timeout = torch.zeros(frame.shape[0], 1, dtype=torch.bool, device=frame.device)
        safety_state = adapter.safety_state(timeout)
        safety_filter_cls(safety_limits).project(base_action, torch.zeros_like(base_action), safety_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class AdaptationRuntimeOrchestrator:
    """Pure runtime state machine for novelty-triggered actor updates."""

    def __init__(
        self,
        *,
        mode: Literal["base", "residual_frozen", "online"],
        novelty_gate: NoveltyGate,
        adaptation_performance: AdaptationPerformance,
        counters: RuntimeCounters | None = None,
    ) -> None:
        self.mode = mode
        self.novelty_gate = novelty_gate
        self.adaptation_performance = adaptation_performance
        self.counters = counters if counters is not None else RuntimeCounters()
        self.adaptation_episode = 0
        self.active_episode = 0
        self.last_rollback_reason = ""

    def observe_step(
        self,
        *,
        step: int,
        velocity_error: float,
        tilt_rms: float,
        nonfoot_contact: int,
        reward: float,
        paused: bool = False,
        control_timed_out: bool = False,
        done: bool = False,
        safety_reasons: tuple[str, ...] = (),
    ) -> RuntimeStepResult:
        commands: list[AdaptationCommand] = []
        clear_adaptation_replay = False
        rollback_reason = ""
        fall_reset_reason = ""
        safety_reason = ""
        control_timeout_preserved_actor = self.active_episode > 0 and control_timed_out

        if done:
            fall_reset_reason = "fall_reset"
        elif self.active_episode > 0 and not control_timed_out and paused:
            safety_reason = safety_reasons[0] if safety_reasons else "safety_pause"

        if safety_reason:
            episode_to_rollback = self.active_episode
            novelty_decision = self.novelty_gate.force_cooldown()
            self.adaptation_performance.start_episode()
            self.active_episode = 0
            self.counters.record_rollback(safety_reason)
            self.last_rollback_reason = safety_reason
            rollback_reason = safety_reason
            clear_adaptation_replay = True
            commands.append(AdaptationCommand("rollback", episode_to_rollback, safety_reason))
        elif fall_reset_reason:
            episode_to_reset = self.active_episode
            novelty_decision = self.novelty_gate.force_cooldown()
            self.adaptation_performance.start_episode()
            self.active_episode = 0
            self.counters.record_fall_reset()
            self.last_rollback_reason = fall_reset_reason
            rollback_reason = fall_reset_reason
            if episode_to_reset > 0:
                clear_adaptation_replay = True
                commands.append(
                    AdaptationCommand("fall_reset", episode_to_reset, fall_reset_reason)
                )
        else:
            novelty_decision = self.novelty_gate.observe(
                velocity_error,
                tilt_rms,
                bool(nonfoot_contact),
                reward,
                paused=paused or control_timed_out,
            )

        if self.mode == "online" and novelty_decision.entered_adapting and not done:
            self.adaptation_performance.start_episode()
            self.adaptation_episode += 1
            self.active_episode = self.adaptation_episode
            self.counters.record_novelty_entry()
            clear_adaptation_replay = True
            commands.append(AdaptationCommand("start_episode", self.adaptation_episode))

        adaptation_enabled = (
            self.mode == "online"
            and novelty_decision.actor_update_enabled
            and self.active_episode > 0
            and not paused
        )

        if adaptation_enabled:
            if self.counters.actor_updates_for_episode(self.active_episode) > 0:
                self.adaptation_performance.mark_actor_updated()
            performance_decision = self.adaptation_performance.add(
                velocity_error=velocity_error,
                tilt_rms=tilt_rms,
                nonfoot_contact=nonfoot_contact,
            )
            if performance_decision is not None:
                if performance_decision.improved:
                    commands.append(AdaptationCommand("checkpoint", self.active_episode))
                elif performance_decision.should_rollback:
                    episode_to_rollback = self.active_episode
                    rollback_reason = performance_decision.rollback_reason or "performance_degradation"
                    novelty_decision = self.novelty_gate.force_cooldown()
                    self.active_episode = 0
                    self.counters.record_rollback(rollback_reason)
                    self.last_rollback_reason = rollback_reason
                    clear_adaptation_replay = True
                    commands.append(AdaptationCommand("rollback", episode_to_rollback, rollback_reason))
                    self.adaptation_performance.start_episode()
        elif novelty_decision.left_adapting and self.active_episode > 0:
            episode_to_end = self.active_episode
            self.active_episode = 0
            clear_adaptation_replay = True
            self.adaptation_performance.start_episode()
            commands.append(AdaptationCommand("end_episode", episode_to_end))

        self.counters.record_state(novelty_decision.state.value)
        return RuntimeStepResult(
            adaptation_enabled=adaptation_enabled,
            clear_adaptation_replay=clear_adaptation_replay,
            commands=tuple(commands),
            record_fields={
                "step": step,
                "adaptation_state": novelty_decision.state.value,
                "novelty_score": novelty_decision.novelty_score,
                "velocity_excess": novelty_decision.velocity_excess,
                "tilt_excess": novelty_decision.tilt_excess,
                "reward_drop": novelty_decision.reward_drop,
                "actor_update_enabled": adaptation_enabled,
                "adaptation_episode": self.active_episode,
                **rollback_record_fields(rollback_reason, self.last_rollback_reason),
                "performance_evaluation_enabled": self.adaptation_performance.evaluation_enabled,
                "control_timeout_preserved_actor": control_timeout_preserved_actor,
            },
        )

    def record_critic_update(self) -> None:
        self.counters.record_critic_update()

    def record_actor_update(self, *, step: int) -> None:
        self.counters.record_actor_update(step=step, adaptation_episode=self.active_episode)


def main() -> int:
    args = build_play_parser().parse_args()
    managed = attach_session(args, "sac")
    task_objective = resolve_task_objective(args)
    objective_metadata = None if task_objective is None else task_objective.contract()
    objective_sha256 = None if task_objective is None else task_objective.sha256()
    default_protocol = default_flat_online_protocol()
    if args.protocol_max_stage_episodes >= args.protocol_max_episodes:
        try:
            target_stage = next(
                stage for stage in default_protocol.stages if stage.name == args.terrain_stage
            )
        except StopIteration as error:
            raise ValueError(f"unknown target stage: {args.terrain_stage}") from error
        stages = (
            replace(
                target_stage,
                max_episodes=args.protocol_max_stage_episodes,
            ),
        )
    else:
        stages = tuple(
            replace(stage, max_episodes=args.protocol_max_stage_episodes)
            for stage in default_protocol.stages
        )
    protocol = replace(
        default_protocol,
        stages=stages,
        max_total_episodes=args.protocol_max_episodes,
        max_total_steps=args.protocol_max_total_steps,
    )
    if args.protocol_sha256 != protocol.sha256():
        raise ValueError("--protocol-sha256 does not match the runtime protocol")
    geometry = (
        TerrainGeometry(step_height=0.0, rough_amplitude=0.0, seed=args.terrain_seed)
        if args.flat_retention_probe
        else resolve_worker_geometry(
            args.terrain_stage, args.terrain_seed, terrain_split=args.terrain_split
        )
    )
    if args.terrain_rough_amplitude is not None:
        geometry = replace(
            geometry,
            step_height=0.0,
            rough_amplitude=args.terrain_rough_amplitude,
        )
    learning_enabled = args.mode == "online" and not args.evaluation_only
    resting_residual_gain = resolve_control_gain(
        mode=args.mode, managed_frozen=managed is not None and managed.frozen,
        residual_gain=args.residual_gain, adapting_residual_gain=args.adapting_residual_gain,
    )
    if args.target_warmup_steps:
        if args.update_schedule != "control-step" or args.n_step != 1:
            raise ValueError("target warmup requires control-step updates and n=1")
        if not 1 <= args.target_critic_warmup_updates <= 1000:
            raise ValueError("target critic warmup updates must be in 1..1000")
        if not max(128, args.min_replay_size) <= args.target_warmup_steps <= args.replay_capacity:
            raise ValueError("target warmup must fit replay and provide at least one full 128-row minibatch")
    if task_objective is not None and learning_enabled and (
        args.flat_replay is not None or args.residual_checkpoint is not None or args.novelty_baseline_in is not None
    ):
        raise ValueError("finite-horizon-v1 learning requires a clean initializer or a matching experiment checkpoint; legacy replay/critic cannot be imported")
    if (
        args.mode != "base"
        and args.residual_checkpoint is None
        and args.experiment_checkpoint_in is None
        and not (args.mode == "online" and args.flat_replay is not None)
        and not (args.mode == "residual_frozen" and args.residual_gain == 0.0)
        and not (task_objective is not None and args.mode == "online")
    ):
        raise ValueError(
            f"--residual-checkpoint or --experiment-checkpoint-in is required for mode {args.mode}"
        )
    if args.task not in {
        "FlatOnline-Deeprobotics-Lite3-v0",
        "ContinuousRough-Deeprobotics-Lite3-v0",
        "Grass-Deeprobotics-Lite3-v0",
        "RockyRoad-Deeprobotics-Lite3-v0",
    }:
        raise ValueError("the curriculum worker must use a supported Lite3 online terrain task")
    if args.updates_per_second <= 0.0 or args.actor_updates_per_second <= 0.0:
        raise ValueError("critic and actor update frequencies must be positive")
    if args.residual_rate_limit <= 0.0:
        raise ValueError("residual rate limit must be positive")
    if args.video:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    updater_thread: threading.Thread | None = None
    trajectory = None
    latency_probe = None
    stop_event = threading.Event()
    failure_writer = None
    try:
        import gymnasium as gym
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
        from rl_training.online_adaptation.checkpointing import (
            load_experiment_checkpoint,
            save_experiment_checkpoint,
        )
        from rl_training.online_adaptation.isaaclab_adapter import (
            Lite3IsaacLabAdapter,
        )
        from rl_training.online_adaptation.terrain_observation import HeightScanConfig
        from rl_training.online_adaptation.deployment_startup import make_lite3_deployment_env
        from rl_training.online_adaptation.metrics import AdaptationPerformance, RollingMetrics
        from rl_training.online_adaptation.networks import LatentResidualActor, ResidualActor, TwinCritic
        from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate, NoveltyState
        from rl_training.online_adaptation.progress_novelty import ForwardProgressMonitor
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.replay_buffer import (
            ReplayBuffer,
            adapt_replay_state_frame_dim,
        )
        from rl_training.online_adaptation.rsl_rl_compat import (
            freeze_rsl_rl_models,
            load_frozen_rsl_rl_checkpoint,
            prepare_rsl_rl_agent_cfg,
            split_rsl_rl_endings,
        )
        from rl_training.online_adaptation.runtime import (
            as_torch_device,
            safety_violation_occurred,
            write_failure_record,
        )
        from rl_training.online_adaptation.sac import SACConfig, StableStateManager
        from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
        from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
        from rl_training.online_adaptation.teacher_steering import load_teacher_decoder
        from rl_training.online_adaptation.step_trajectory import StepTrajectory

        failure_writer = write_failure_record

        torch.manual_seed(args.seed)
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir))
        metrics_stream = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")
        actor_updates_stream = (run_dir / "actor_updates.jsonl").open("w", encoding="utf-8")

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        from rl_training.online_adaptation.flat_retention import configure_flat_retention_plane, summarize_flat_retention
        if args.flat_retention_probe:
            configure_flat_retention_plane(env_cfg)
        else:
            apply_geometry_to_env_cfg(env_cfg, geometry)
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 1
        initial_state_contract = configure_initial_state(env_cfg, args.initial_state)
        env_cfg.terminations.time_out = None
        env_cfg.terminations.terrain_out_of_bounds = None
        configure_forward_command(env_cfg.commands.base_velocity, args.forward_speed)
        sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 0.02))
        control_step_dt = sim_dt * float(getattr(env_cfg, "decimation", 1))
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg = prepare_rsl_rl_agent_cfg(agent_cfg)
        (run_dir / "resolved_env.json").write_text(json.dumps(env_cfg.to_dict(), indent=2, default=str) + "\n")
        def transform_video(env):
            if not args.video:
                return env
            return gym.wrappers.RecordVideo(
                env,
                video_folder=str(run_dir / "videos"),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length or args.max_steps,
                disable_logger=True,
            )

        wrapped_env = make_lite3_deployment_env(
            args.task,
            env_cfg,
            agent_cfg,
            gym_module=gym,
            env_kwargs={"render_mode": "rgb_array"} if args.video else {},
            env_transform=transform_video,
        )
        online_agent = None
        novelty_gate = None
        stable_manager = None
        checkpoint_saved = False
        physical_handoff_prepared = False
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            checkpoint_load = load_frozen_rsl_rl_checkpoint(runner, args.base_checkpoint)
            base_policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            freeze_rsl_rl_models(runner)

            device = as_torch_device(wrapped_env.unwrapped.device)
            normal_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
            actor_limits = torch.zeros_like(normal_limits) if args.mode == "base" else normal_limits
            adapter_step_height = (
                0.0
                if args.flat_retention_probe and args.residual_frame_dim == 59
                else (
                    None
                    if args.flat_retention_probe or args.terrain_feature_source == "proprio"
                    else geometry.step_height
                )
            )
            adapter = Lite3IsaacLabAdapter(
                wrapped_env.unwrapped,
                track_boundary_x=resolve_track_boundary_x(env_cfg),
                step_height=adapter_step_height,
                terrain_feature_source=args.terrain_feature_source,
                height_scan_config=HeightScanConfig(
                    rows=23,
                    cols=15,
                    min_valid_fraction=args.terrain_min_valid_fraction,
                ),
            )
            frame_dim = adapter.frame_dim
            if args.residual_frame_dim is not None and frame_dim != args.residual_frame_dim:
                raise ValueError(
                    "residual frame dimension differs from the requested runtime contract: "
                    f"actual={frame_dim}, requested={args.residual_frame_dim}"
                )
            if args.latent_code_dim:
                decoder_matrix = None
                if args.latent_decoder is not None:
                    decoder_matrix = load_teacher_decoder(
                        args.latent_decoder,
                        action_limits=actor_limits * args.latent_residual_fraction,
                        code_dim=args.latent_code_dim,
                    )
                actor = LatentResidualActor(
                    frame_dim, 12, code_dim=args.latent_code_dim, history_dim=64,
                    action_limits=actor_limits * args.latent_residual_fraction,
                    decoder_matrix=decoder_matrix,
                ).to(device)
                actor_architecture = (
                    "latent_residual_teacher_decoder_v1"
                    if args.latent_decoder is not None
                    else "latent_residual_v1"
                )
            else:
                actor = ResidualActor(frame_dim, 12, 64, actor_limits).to(device)
                actor_architecture = "direct_residual_v1"
            critic = TwinCritic(frame_dim, 12, 64).to(device)
            runtime_config = AgentConfig(
                frame_dim=frame_dim,
                replay_capacity=args.replay_capacity,
                min_replay_size=args.min_replay_size,
                updates_per_second=args.updates_per_second,
                residual_gain=0.0,
                n_step=args.n_step,
                flat_fraction_initial=args.flat_fraction_initial,
                flat_fraction_ready=args.flat_fraction_ready,
                flat_critic_warmup_updates=0 if task_objective is not None else args.flat_critic_warmup_updates,
                critic_replay_mode=args.critic_replay_mode,
            )
            safety_limits = SafetyLimits(
                residual_limits=normal_limits,
                residual_rate_limit=args.residual_rate_limit,
                **risk_gate_kwargs(args),
            )
            online_agent = OnlineAdaptationAgent(
                num_envs=1,
                actor=actor,
                critic=critic,
                safety_limits=safety_limits,
                config=runtime_config,
                sac_config=SACConfig(
                    target_entropy=(
                        -float(args.latent_code_dim)
                        if args.latent_code_dim
                        else -6.0
                    ),
                    flat_anchor_weight=args.flat_anchor_weight,
                    max_critic_loss=args.max_critic_loss,
                    max_mean_td_error=args.max_mean_td_error,
                    max_critic_grad_norm=args.max_critic_grad_norm,
                    target_q_min=args.target_q_min,
                    target_q_max=args.target_q_max,
                    critic_huber_delta=args.critic_huber_delta,
                    deterministic_actor_weight=args.deterministic_actor_weight,
                    initial_alpha=args.initial_alpha,
                    alpha_max=args.alpha_max,
                ),
                device=device,
            )
            if args.flat_replay is not None:
                flat_state = torch.load(
                    args.flat_replay, map_location="cpu", weights_only=False
                )
                flat_state = adapt_replay_state_frame_dim(flat_state, frame_dim)
                flat_replay = ReplayBuffer(
                    capacity=int(flat_state["capacity"]),
                    history_length=int(flat_state["history_length"]),
                    frame_dim=int(flat_state["frame_dim"]),
                    action_dim=int(flat_state["action_dim"]),
                    storage_dtype=getattr(torch, flat_state["storage_dtype"]),
                )
                flat_replay.load_state_dict(flat_state)
                if not flat_replay.frozen:
                    raise ValueError("--flat-replay must contain a frozen replay state")
                online_agent.flat_replay = flat_replay
            reward_weights = replace(
                RewardWeights(),
                post_step_stability_weight=args.post_step_stability_weight,
                post_step_progress_weight=args.post_step_progress_weight,
                stall_penalty_weight=args.stall_penalty_weight,
                completion_bonus=args.completion_bonus,
                pre_step_clearance_weight=args.pre_step_clearance_weight,
                pre_step_clearance_target=args.pre_step_clearance_target,
                pre_step_foot_forward_weight=args.pre_step_foot_forward_weight,
                pre_step_foot_forward_target=args.pre_step_foot_forward_target,
                rough_swing_lift_weight=args.rough_swing_lift_weight,
                rough_swing_lift_target=args.rough_swing_lift_target,
            )
            if task_objective is not None:
                reward_weights = RewardWeights()
            current_training_contract = training_contract(
                online_agent.updater.config, reward_weights, target_semantics=args.target_semantics,
                risk_gate=safety_limits.risk_gate_contract(),
                actor_architecture=actor_architecture,
                latent_code_dim=args.latent_code_dim or None,
                latent_residual_fraction=(
                    args.latent_residual_fraction if args.latent_code_dim else None
                ),
                task_objective=objective_metadata,
            )
            swing_lift_tracker = SwingLiftTracker()
            restored_experiment = None
            if args.experiment_checkpoint_in is not None:
                restored_experiment = load_experiment_checkpoint(
                    args.experiment_checkpoint_in,
                    expected_protocol_sha256=protocol.sha256(),
                    map_location=device,
                    restore_rng=learning_enabled,
                )
                if managed is not None and learning_enabled:
                    managed.remember_rng()
                validate_terrain_checkpoint_metadata(
                    restored_experiment.get("terrain_metadata"),
                    adapter.terrain_metadata(),
                )
                if learning_enabled:
                    validate_training_contract(
                        restored_experiment.get("training_contract"), current_training_contract
                    )
                online_agent.load_state_dict(restored_experiment["agent"])
            if (
                args.residual_checkpoint is not None
                and args.mode != "base"
                and restored_experiment is None
            ):
                online_agent.updater.load(args.residual_checkpoint, map_location=device)
                online_agent.publisher.publish(online_agent.updater.actor, online_agent.updater.version)

            metrics = RollingMetrics(window=250)
            from rl_training.online_adaptation.target_warmup import restore_target_warmup, pretrain_target_critic
            target_warmup = restore_target_warmup(args, "sac", restored_experiment)
            warmup_before = target_warmup.state_dict()
            warmup_update_records = []
            novelty_gate = NoveltyGate(
                NoveltyConfig(
                    calibration_steps=args.novelty_calibration_steps,
                    enter_threshold=args.novelty_enter_threshold,
                    exit_threshold=args.novelty_exit_threshold,
                    enter_count=args.novelty_enter_count,
                    enter_window=args.novelty_enter_window,
                    recovery_steps=args.novelty_recovery_steps,
                    cooldown_steps=args.adaptation_cooldown_steps,
                )
            )
            progress_monitor = ForwardProgressMonitor(
                min_delta_m=protocol.progress_min_delta_m,
                stall_steps=protocol.progress_stall_steps,
            )
            adaptation_performance = AdaptationPerformance(
                window=args.rollback_window,
                improvement_ratio=args.stable_improvement_ratio,
                degradation_ratio=args.rollback_degradation_ratio,
                patience=args.rollback_patience,
            )
            stable_manager = StableStateManager()
            actor_drift_monitor = ActorDriftMonitor()
            if restored_experiment is not None:
                novelty_gate.load_state_dict(restored_experiment["novelty_gate"])
                stable_manager.load_state_dict(restored_experiment["stable_manager"])
            elif args.novelty_baseline_in is not None:
                if sha256_file(args.novelty_baseline_in) != args.novelty_baseline_sha256:
                    raise ValueError("novelty baseline checksum mismatch")
                baseline_payload = json.loads(
                    Path(args.novelty_baseline_in).read_text(encoding="utf-8")
                )
                novelty_gate.load_baseline(baseline_payload)
            if learning_enabled and online_agent.flat_replay is not None:
                # Critic warmup must use the same applied residual scale as
                # the first deployment phase; the runtime config starts at
                # zero only to prevent pre-handoff control leakage.
                online_agent.set_residual_gain(resting_residual_gain)
                online_agent.prewarm_critics_from_flat()
            if restored_experiment is None:
                stable_manager.checkpoint(online_agent.updater)
            before_parameters = {
                key: value.detach().cpu().clone() for key, value in online_agent.updater.actor.state_dict().items()
            }
            torch.save(before_parameters, run_dir / "residual_before.pt")

            shared_lock = threading.Lock()
            update_lock = threading.Lock()
            control_update_lock = threading.Lock()
            paused_event = threading.Event()
            actor_gate_event = threading.Event()
            commands: queue.SimpleQueue[AdaptationCommand] = queue.SimpleQueue()
            update_wakeup = threading.Event()
            update_cycle_complete = threading.Event()
            update_errors: list[BaseException] = []
            counters = RuntimeCounters()
            update_budget = TransitionUpdateBudget(
                critic_updates_per_transition=args.critic_updates_per_transition,
                actor_updates_per_transition=args.actor_updates_per_transition,
            )
            if managed is not None and restored_experiment is not None and learning_enabled:
                update_budget.restore_for_new_trial(restored_experiment.get("update_budget"))
            adaptation_episode = 0
            active_episode = 0
            current_step = 0
            last_update: dict[str, Any] = {}
            actor_update_events_logged = 0
            last_rollback_reason = ""
            pending_rollback_completion: threading.Event | None = None
            control_step_updates = args.update_schedule == "control-step"

            def read_active_episode() -> int:
                return active_episode

            def write_active_episode(episode: int) -> None:
                nonlocal active_episode
                active_episode = episode

            def updater_loop() -> None:
                nonlocal actor_update_events_logged, last_rollback_reason
                critic_period = 1.0 / args.updates_per_second
                actor_period = 1.0 / args.actor_updates_per_second
                next_critic = time.perf_counter()
                next_actor = next_critic
                checkpointed_episode = 0
                try:
                    while not stop_event.is_set() or not commands.empty():
                        if control_step_updates:
                            if not update_wakeup.wait(timeout=0.1):
                                continue
                            update_wakeup.clear()
                        while True:
                            try:
                                command = commands.get_nowait()
                            except queue.Empty:
                                break
                            with control_update_lock, update_lock:
                                def enable_actor_if_episode_active(episode: int) -> bool:
                                    return enable_actor_gate_if_active(
                                        shared_lock=shared_lock,
                                        actor_gate_event=actor_gate_event,
                                        read_active_episode=read_active_episode,
                                        adaptation_episode=episode,
                                    )

                                checkpointed_episode, command_rollback_reason = apply_adaptation_command(
                                    command,
                                    checkpointed_episode=checkpointed_episode,
                                    updater=online_agent.updater,
                                    publisher=online_agent.publisher,
                                    stable_manager=stable_manager,
                                    drift_monitor=actor_drift_monitor,
                                    actor_gate_event=actor_gate_event,
                                    clear_adaptation_replay=online_agent.clear_adaptation_replay,
                                    enable_actor_if_episode_active=enable_actor_if_episode_active,
                                    clear_actor_update_budget=update_budget.clear_actor,
                                )
                                if command_rollback_reason is not None:
                                    with shared_lock:
                                        last_rollback_reason = command_rollback_reason

                        now = time.perf_counter()
                        with shared_lock:
                            episode_snapshot = active_episode
                            step_snapshot = current_step
                        budget_snapshot = update_budget.snapshot()
                        if not target_warmup.ready:
                            if not target_warmup.collecting:
                                def record_warmup_update(stats):
                                    with shared_lock:
                                        record_target_critic_update(
                                            stats=stats, records=warmup_update_records, last_update=last_update,
                                            counters=counters, step=step_snapshot, adaptation_episode=episode_snapshot,
                                            path=run_dir / "target_warmup_updates.jsonl",
                                            residual_gain=args.adapting_residual_gain,
                                        )
                                    if len(warmup_update_records) % 100 == 0:
                                        print(f"[target warmup] critic updates={len(warmup_update_records)}", flush=True)
                                with control_update_lock, update_lock:
                                    pretrain_target_critic(target_warmup, online_agent, on_update=record_warmup_update,
                                                          residual_gain=args.adapting_residual_gain)
                                stable_manager.checkpoint(online_agent.updater)
                            update_cycle_complete.set()
                            continue
                        actor_due = (
                            (control_step_updates or now >= next_actor)
                            and budget_snapshot["actor"] >= 1.0
                            and (not args.target_warmup_steps or budget_snapshot["critic"] >= 1.0)
                            and checkpointed_episode > 0
                            and checkpointed_episode == episode_snapshot
                            and online_agent.should_update_actor(
                                len(online_agent.adaptation_replay),
                                args.adaptation_min_replay_size,
                                paused_event.is_set(),
                                actor_gate_event.is_set(),
                            )
                        )
                        stats = None
                        policy_drift = None
                        if actor_due:
                            if not update_budget.take_actor(consume_critic=bool(args.target_warmup_steps)):
                                continue
                            critic_batch = online_agent.sample_critic_batch()
                            actor_batch = online_agent.sample_actor_batch()
                            flat_anchor_batch = (
                                None
                                if online_agent.flat_replay is None
                                else online_agent.flat_replay.sample(
                                    online_agent.config.batch_size,
                                    online_agent.device,
                                )
                            )

                            def episode_is_still_active() -> bool:
                                with shared_lock:
                                    return active_episode == episode_snapshot

                            def publish_if_episode_is_still_active(
                                candidate_actor: object, candidate_version: int
                            ) -> bool:
                                with shared_lock:
                                    if (
                                        active_episode != episode_snapshot
                                        or not actor_gate_event.is_set()
                                    ):
                                        return False
                                    online_agent.publisher.publish(candidate_actor, candidate_version)
                                    return True

                            with control_update_lock, update_lock:
                                outcome = update_and_publish_actor(
                                    updater=online_agent.updater,
                                    publisher=online_agent.publisher,
                                    drift_monitor=actor_drift_monitor,
                                    critic_batch=critic_batch,
                                    actor_batch=actor_batch,
                                    flat_anchor_batch=flat_anchor_batch,
                                    **actor_update_options(args),
                                    adaptation_episode=episode_snapshot,
                                    update_step=step_snapshot,
                                    episode_is_active=episode_is_still_active,
                                    final_publish_attempt=publish_if_episode_is_still_active,
                                    residual_gain=online_agent.residual_gain,
                                )
                            stats = replace(
                                outcome.stats,
                                flat_batch_fraction=online_agent.last_flat_batch_fraction,
                                n_step=online_agent.config.n_step,
                            )
                            policy_drift = outcome.policy_drift
                            next_actor = now + actor_period
                            next_critic = max(next_critic + critic_period, now)
                        elif (
                            (control_step_updates or now >= next_critic)
                            and budget_snapshot["critic"] >= 1.0
                            and online_agent.should_update(
                                len(online_agent.replay),
                                args.min_replay_size,
                                paused_event.is_set(),
                            )
                        ):
                            if not update_budget.take_critic():
                                continue
                            with control_update_lock, update_lock:
                                stats = online_agent.update(update_actor=False)
                            next_critic = now + critic_period

                        if stats is not None and stats.rejection_reason != "stale_adaptation_episode":
                            with shared_lock:
                                actor_update_event = record_update_result(
                                    stats=stats,
                                    policy_drift=policy_drift,
                                    last_update=last_update,
                                    counters=counters,
                                    step=step_snapshot,
                                    adaptation_episode=episode_snapshot,
                                )
                            if actor_update_event is not None:
                                write_actor_update_event(actor_updates_stream, actor_update_event)
                                with shared_lock:
                                    actor_update_events_logged += 1
                        if control_step_updates:
                            update_cycle_complete.set()
                        else:
                            stop_event.wait(0.001)
                except BaseException as error:
                    update_errors.append(error)
                    stop_event.set()
                    update_cycle_complete.set()

            if learning_enabled:
                updater_thread = threading.Thread(target=updater_loop, name="lite3-sac-updater", daemon=False)
                updater_thread.start()

            config = {
                "resting_residual_gain": resting_residual_gain,
                "arguments": vars(args),
                "task_objective": objective_metadata, "task_objective_sha256": objective_sha256,
                "risk_gate": {"semantics": "max_state_risk_residual_envelope_v1", **risk_gate_kwargs(args)},
                "critic_action_semantics": "requested_residual_before_safety_v1",
                "replay_schema_version": 2,
                "agent": asdict(runtime_config),
                "reward": asdict(reward_weights),
                "continuous_no_reset": True,
                "heldout_mask_usage": "metrics_only",
                "terrain_stage": args.terrain_stage,
                "terrain_seed": args.terrain_seed,
                "completion_qualification_version": protocol.completion_qualification_version,
                "protocol_sha256": protocol.sha256(),
                "geometry": asdict(geometry),
                "terrain_metadata": adapter.terrain_metadata(),
                "base_checkpoint_load": checkpoint_load,
                "contact_termination": CONTACT_TERMINATION_PROTOCOL,
            }
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
            )

            observation, _ = seeded_initial_reset(
                wrapped_env, args.seed if managed is None else managed.reset_seed,
                preserve_learning_rng=managed is not None,
            )
            observation = pin_forward_observation(wrapped_env, observation, args.forward_speed)
            adapter.reset_contact_diagnostics()
            generation = wrapped_env.consume_startup_generation()
            if generation is None:
                raise RuntimeError("deployment startup reset did not produce a generation")
            initial_x = float(adapter.base_x().mean().item())
            initial_y = float(adapter.base_y().mean().item())
            track_completion = None
            if args.require_track_completion and not args.flat_retention_probe:
                track_completion = build_track_completion(
                    env_cfg,
                    stage_name=args.terrain_stage,
                    initial_x=initial_x,
                    initial_y=initial_y,
                    protocol=protocol,
                )
            online_agent.reset_for_deployment_handoff()
            previous_action = torch.zeros(1, 12, device=device)
            pin_forward_command(wrapped_env.unwrapped, args.forward_speed)
            warm_up_control_path(
                base_policy=base_policy,
                online_agent=online_agent,
                adapter=adapter,
                observation=observation,
                previous_action=previous_action,
                safety_limits=safety_limits,
                safety_filter_cls=SafetyFilter,
                deterministic=not args.stochastic_control,
                device=device,
            )
            online_agent.set_residual_gain(resting_residual_gain)
            post_step_recovery = PostStepRecoveryRamp(
                transition_steps=args.post_step_recovery_steps,
                target_gain=args.post_step_residual_gain,
            )
            previous_residual = torch.zeros_like(previous_action)
            previous_x = adapter.base_x().detach().clone()
            if not hasattr(wrapped_env.unwrapped, "add_pre_reset_callback"):
                raise RuntimeError(
                    "online adaptation environment must support pre-reset terminal capture"
                )
            terminal_snapshot: PostStepSnapshot | None = None
            current_decision = None

            def capture_terminal_state(env_ids: Any) -> None:
                nonlocal terminal_snapshot
                # The deployment wrapper performs an explicit standing reset
                # after the manager environment's automatic reset.  Preserve
                # the first callback, which is the actual terminal state.
                if terminal_snapshot is not None:
                    return
                if current_decision is None:
                    raise RuntimeError(
                        "pre-reset capture occurred without an active decision"
                    )
                ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
                captured_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
                captured_mask[ids] = True
                terminal_observation = wrapped_env.unwrapped.observation_manager.compute()
                policy_observation = terminal_observation.get("policy")
                if not isinstance(policy_observation, torch.Tensor):
                    raise RuntimeError("terminal policy observation must be a tensor")
                terminal_snapshot = PostStepSnapshot(
                    mask=captured_mask,
                    policy_observation=policy_observation.detach().clone(),
                    frame=adapter.frame(current_decision.action).detach().clone(),
                    signals=clone_reward_signals(adapter.reward_signals()),
                    base_x=adapter.base_x().detach().clone(),
                    base_y=adapter.base_y().detach().clone(),
                    foot_y=adapter.foot_y().detach().clone(),
                )

            wrapped_env.unwrapped.add_pre_reset_callback(capture_terminal_state)
            safety_fallbacks = 0
            safety_violation = False
            done_count = 0
            fell = False
            heldout_steps = 0
            flat_rows = []
            executed_steps = 0
            completion_stop = False
            post_step_checkpointed = False
            novelty_decision = novelty_gate.observe(0.0, 0.0, False, 0.0, paused=True)
            trajectory = StepTrajectory(run_dir / "trajectory", vector_dtype=args.trajectory_dtype, metadata={
                "algorithm": "residual_sac", "mode": args.mode,
                "task_objective": objective_metadata, "task_objective_sha256": objective_sha256,
                "terrain_seed": args.terrain_seed, "simulation_seed": args.seed if managed is None else managed.reset_seed,
                "learner_seed": args.seed,
                "protocol_sha256": protocol.sha256(), "initial_x": initial_x,
                "step_dt": control_step_dt,
                "actuator_delays": actuator_delay_metadata(adapter.robot),
                "reset_rng_contract": "seed_before_initial_physical_reset_v1",
                "initial_state_contract": initial_state_contract,
                "initial_root_state": initial_root_state(adapter.robot),
                "critic_action_semantics": "requested_residual_before_safety_v1",
                "base_sha256": sha256_file(args.base_checkpoint),
                "actor_architecture": actor_architecture,
                "latent_decoder_sha256": None if args.latent_decoder is None else sha256_file(args.latent_decoder),
                "endings_semantics": "finite_horizon_v1" if task_objective is not None else "legacy_worker_physical_endings_with_separate_collector_cutoff",
                "policy_version_missing": -1,
            })
            trace_started = time.perf_counter()
            total_reward = 0.0
            task_boundary = None
            if managed is not None:
                managed.restore_rng()

            from rl_training.online_adaptation.control_latency import ControlLatencyProbe
            latency_probe = ControlLatencyProbe()
            for step in range(1, args.max_steps + 1):
                executed_steps = step
                terminal_snapshot = None
                with shared_lock:
                    current_step = step
                    active_episode, final_step_cutoff = cutoff_before_final_step(
                        step=step,
                        max_steps=args.max_steps,
                        active_episode=active_episode,
                        actor_gate_event=actor_gate_event,
                    )
                pending_rollback_completion, rollback_fallback_required = prepare_rollback_fallback(
                    pending_rollback_completion
                )
                loop_started = time.perf_counter()
                observation = pin_forward_observation(wrapped_env, observation, args.forward_speed)
                frame, timeout, safety_state = prepare_control_inputs(
                    adapter=adapter,
                    previous_action=previous_action,
                    device=device,
                )
                terrain_valid = adapter.terrain_valid
                counters.record_terrain_validity(terrain_valid)
                with control_phase(
                    control_update_lock=control_update_lock,
                    device=device,
                ) as control_started:
                    latency_probe.start(control_started)
                    with torch.inference_mode():
                        raw_base_action = base_policy(observation)
                        base_action = clip_lite3_policy_action(raw_base_action)
                    latency_probe.mark("base_policy")
                    warmup_step = learning_enabled and target_warmup.collecting
                    deterministic_control = warmup_step or control_is_deterministic(
                        stochastic_control=args.stochastic_control,
                        adapting_stochastic_control=args.adapting_stochastic_control,
                        learning_enabled=learning_enabled,
                        novelty_state=novelty_decision.state,
                    )
                    action_residual_gain = online_agent.residual_gain
                    filter_memory = online_agent.safety.temporal_snapshot(base_action)
                    decision = online_agent.begin_step(
                        frame,
                        base_action,
                        safety_state,
                        deterministic=deterministic_control,
                        terrain_valid=terrain_valid,
                        base_only=warmup_step,
                    )
                    latency_probe.mark("adapter_and_safety")
                    decision, rollback_fallback_used = enforce_prepared_rollback_fallback(
                        online_agent=online_agent,
                        decision=decision,
                        safety_state=safety_state,
                        fallback_required=rollback_fallback_required,
                    )
                    current_decision = decision
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    latency_probe.mark("fallback_and_device_sync")
                    control_latency = latency_probe.finish()
                    control_ms = (time.perf_counter() - control_started) * 1_000.0
                    control_timed_out = False
                    if control_ms > args.control_timeout_ms:
                        control_timed_out = True
                        timed_out_state = replace(safety_state, control_timed_out=torch.ones_like(timeout))
                        decision = enforce_control_timeout_fallback(
                            online_agent=online_agent,
                            timed_out_state=timed_out_state,
                        )
                        current_decision = decision
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        control_ms = (time.perf_counter() - control_started) * 1_000.0
                if rollback_fallback_used or control_timed_out:
                    safety_fallbacks += 1
                if update_errors:
                    raise RuntimeError("background SAC updater failed") from update_errors[0]

                paused_now = bool(decision.paused.any().item())
                update_pause_event(
                    paused_event,
                    paused=online_agent.all_environments_paused(decision.paused),
                    control_timed_out=control_timed_out,
                )
                if safety_violation_occurred(decision.paused, control_timed_out):
                    safety_violation = True

                trajectory_inputs = trajectory.snapshot(
                    **online_agent.pending_trajectory_fields(),
                    **filter_memory,
                    observation=observation["policy"], raw_base_action=raw_base_action,
                    executed_action=decision.action, executed_residual_action=decision.residual,
                    previous_action=previous_action, previous_residual_action=previous_residual,
                    root_state=adapter.robot.data.root_state_w,
                    joint_pos=adapter.robot.data.joint_pos, joint_vel=adapter.robot.data.joint_vel,
                    command=wrapped_env.unwrapped.command_manager.get_command("base_velocity"),
                    **{f"safety_{key}": value for key, value in vars(safety_state).items()},
                )
                with torch.inference_mode():
                    # Capture the last pre-action state as well as the state seen
                    # by the environment's pre-reset callback on termination.
                    adapter.observe_nonfoot_contact()
                    if managed is not None:
                        managed.reserve(step)
                    step_result = wrapped_env.step(decision.action)
                    if managed is not None:
                        managed.commit(step)
                    next_observation = step_result[0]
                    terminated, truncated, bootstrap_mask = split_worker_endings(
                        step_result, target_semantics=args.target_semantics
                    )
                    done = terminated | truncated
                    environment_done = bool(done.any().item())
                    termination_reasons = termination_reasons_from_step_result(step_result)
                    if not done.any().item():
                        adapter.observe_nonfoot_contact()
                    contact_diagnostics = adapter.contact_diagnostics()
                    post_reset_snapshot = PostStepSnapshot(
                        mask=torch.zeros_like(done, dtype=torch.bool).reshape(-1, 1),
                        policy_observation=next_observation["policy"],
                        frame=adapter.frame(decision.action),
                        signals=adapter.reward_signals(),
                        base_x=adapter.base_x(),
                        base_y=adapter.base_y(),
                        foot_y=adapter.foot_y(),
                    )
                    post_step_snapshot = select_post_step_snapshot(
                        done=done,
                        terminal=terminal_snapshot,
                        post_reset=post_reset_snapshot,
                    )
                    base_policy_observation = next_observation
                    if environment_done:
                        # RSL-RL actors consume the grouped TensorDict, while the
                        # pre-reset callback captures only the terminal policy group.
                        base_policy_observation = replace_base_policy_group(
                            next_observation,
                            post_step_snapshot.policy_observation,
                        )
                    next_base_action = clip_lite3_policy_action(base_policy(base_policy_observation))
                current_x = post_step_snapshot.base_x.detach().clone()
                base_x_value = float(current_x.mean().item())
                if track_completion is not None:
                    track_completion.observe(base_x_value)
                post_step_reached = bool(
                    track_completion is not None and track_completion.past_step_edge
                )
                if learning_enabled and post_step_reached:
                    (
                        post_step_recovery_gain,
                        post_step_recovery_progress,
                        post_step_recovery_active,
                    ) = post_step_recovery.step(
                        reached=True,
                        current_gain=online_agent.residual_gain,
                    )
                else:
                    post_step_recovery_gain = online_agent.residual_gain
                    post_step_recovery_progress = 0.0
                    post_step_recovery_active = False
                next_frame = post_step_snapshot.frame
                signals = post_step_snapshot.signals
                if args.flat_retention_probe:
                    flat_rows.append({"step": step, "base_x": base_x_value,
                                      "base_lin_vel_xy": signals.base_lin_vel_xy.reshape(-1).cpu().tolist(),
                                      "command_lin_vel_xy": signals.command_lin_vel_xy.reshape(-1).cpu().tolist(),
                                      "done": bool(done.any().item())})
                progress_valid = ~done.bool()
                if paused_now or control_timed_out:
                    progress_valid = torch.zeros_like(progress_valid)
                progress_decision = progress_monitor.observe(
                    base_x=base_x_value,
                    # The worker pins the command to --forward-speed.  Use that
                    # contract for progress gating so a transient command-manager
                    # reset cannot silently disable online adaptation.
                    commanded_forward=bool(args.forward_speed > 0.05),
                    paused=paused_now,
                )
                step_edge_x = None if track_completion is None else track_completion.step_edge_x
                pre_step_clearance_active = step_clearance_reward_active(
                    learning_enabled=learning_enabled,
                    progress_stalled=progress_decision.stalled,
                    base_x=base_x_value,
                    step_edge_x=step_edge_x,
                    interaction_distance=args.pre_step_approach_distance,
                )
                forward_progress = normalized_forward_progress(
                    current_x - previous_x,
                    forward_speed=args.forward_speed,
                    step_dt=control_step_dt,
                    valid_mask=progress_valid,
                )
                swing_lift = swing_lift_tracker.observe(signals.foot_height, signals.foot_contact)
                velocity_error = float(
                    torch.linalg.norm(signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1).mean()
                )
                tilt_rms = float(torch.linalg.norm(signals.projected_gravity_xy, dim=-1).mean())
                nonfoot_contact = int(signals.nonfoot_contact.any().item())
                completion_boundary_reached = observe_completion_snapshot(
                    track_completion,
                    post_step_snapshot,
                    swing_lift=swing_lift,
                    velocity_error=velocity_error,
                    tilt_rms=tilt_rms,
                    environment_done=environment_done,
                    safety_violation=safety_violation,
                )
                rough_swing_active = rough_swing_reward_active(
                    stage=args.terrain_stage,
                    learning_enabled=learning_enabled,
                    adapting=novelty_decision.state is NoveltyState.ADAPTING,
                )
                reward_terms = {}
                if task_objective is not None:
                    task_boundary = task_objective.boundary(
                        step=step, collector_steps=args.max_steps,
                        physical_terminated=bool(terminated.any()), physical_truncated=bool(truncated.any()),
                        safety_failure=safety_violation,
                        qualification_failure=qualification_failure(track_completion), success=completion_boundary_reached,
                    )
                    reward, reward_terms = task_objective.reward(
                        signals, decision.residual, previous_residual, forward_progress=forward_progress,
                        stalled=progress_decision.stalled, base_y=post_step_snapshot.base_y,
                        foot_y=post_step_snapshot.foot_y, initial_y=initial_y, boundary=task_boundary,
                    )
                else:
                    reward = compute_online_reward(
                        signals,
                        decision.residual,
                        previous_residual,
                        reward_weights,
                        swing_lift=swing_lift,
                        rough_swing_mask=torch.full_like(signals.base_lin_vel_z, float(rough_swing_active)),
                        forward_progress=forward_progress,
                        post_step_mask=torch.full_like(
                            forward_progress, float(post_step_reached)
                        ),
                        pre_step_clearance_mask=torch.full_like(
                            forward_progress, float(pre_step_clearance_active)
                        ),
                        progress_stalled_mask=torch.full_like(
                            forward_progress, float(progress_decision.stalled)
                        ),
                        completion_mask=torch.full_like(
                            forward_progress,
                            float(
                                track_completion is not None
                                and completion_boundary_reached
                            ),
                        ),
                        lateral_deviation=(post_step_snapshot.base_y - initial_y).abs(),
                    )
                previous_x = current_x
                rollback_reason = ""
                safety_reason = ""
                fall_reset_reason = ""
                if environment_done:
                    # A fall terminates this physical trial, but does not prove
                    # that the entire learned actor is unsafe.  Handle it before
                    # novelty observation because the wrapper has already reset
                    # the simulator state returned by step().
                    fall_reset_reason = "fall_reset"
                elif active_episode > 0 and not control_timed_out and paused_now:
                    safety_reason = decision.reasons[0] if decision.reasons else "safety_pause"
                if safety_reason:
                    online_agent.set_residual_gain(resting_residual_gain)
                    rollback_completion = threading.Event()
                    pending_rollback_completion = rollback_completion
                    rollback_command = AdaptationCommand(
                        "rollback",
                        active_episode,
                        safety_reason,
                        rollback_completion,
                    )
                    transition_episode_publication(
                        shared_lock=shared_lock,
                        actor_gate_event=actor_gate_event,
                        read_active_episode=read_active_episode,
                        write_active_episode=write_active_episode,
                        next_episode=0,
                        after_transition=lambda episode: commands.put(
                            replace(rollback_command, episode=episode)
                        ),
                    )
                    novelty_decision = novelty_gate.force_cooldown()
                    adaptation_performance.start_episode()
                    with shared_lock:
                        counters.record_rollback(safety_reason)
                        last_rollback_reason = safety_reason
                    rollback_reason = safety_reason
                elif fall_reset_reason:
                    online_agent.set_residual_gain(resting_residual_gain)
                    if active_episode > 0:
                        fall_reset_completion = threading.Event()
                        pending_rollback_completion = fall_reset_completion
                        fall_reset_command = AdaptationCommand(
                            "fall_reset",
                            active_episode,
                            fall_reset_reason,
                            fall_reset_completion,
                        )
                        transition_episode_publication(
                            shared_lock=shared_lock,
                            actor_gate_event=actor_gate_event,
                            read_active_episode=read_active_episode,
                            write_active_episode=write_active_episode,
                            next_episode=0,
                            after_transition=lambda episode: commands.put(
                                replace(fall_reset_command, episode=episode)
                            ),
                        )
                    else:
                        with shared_lock:
                            actor_gate_event.clear()
                            active_episode = 0
                    novelty_decision = novelty_gate.force_cooldown()
                    adaptation_performance.start_episode()
                    with shared_lock:
                        counters.record_fall_reset()
                        last_rollback_reason = fall_reset_reason
                    rollback_reason = fall_reset_reason
                elif completion_boundary_reached or final_step_cutoff:
                    with shared_lock:
                        actor_gate_event.clear()
                    novelty_decision = novelty_gate.force_cooldown()
                    adaptation_performance.start_episode()
                else:
                    novelty_decision = novelty_gate.observe(
                        velocity_error,
                        tilt_rms,
                        bool(nonfoot_contact),
                        float(reward.mean()),
                        paused=paused_now or control_timed_out,
                        progress_stalled=progress_decision.stalled,
                        update_baseline=not args.require_track_completion,
                    )

                if (
                    learning_enabled
                    and novelty_decision.entered_adapting
                    and not environment_done
                    and not completion_boundary_reached
                    and not final_step_cutoff
                ):
                    online_agent.set_residual_gain(args.adapting_residual_gain)
                    adaptation_episode += 1
                    transition_episode_publication(
                        shared_lock=shared_lock,
                        actor_gate_event=actor_gate_event,
                        read_active_episode=read_active_episode,
                        write_active_episode=write_active_episode,
                        next_episode=adaptation_episode,
                        after_transition=lambda _previous_episode: commands.put(
                            AdaptationCommand("start_episode", adaptation_episode)
                        ),
                    )
                    adaptation_performance.start_episode()
                    with shared_lock:
                        counters.record_novelty_entry()

                if (
                    learning_enabled
                    and post_step_reached
                    and not post_step_checkpointed
                    and active_episode > 0
                    and not final_step_cutoff
                ):
                    # Queue this before any same-step performance rollback.  The
                    # updater consumes commands FIFO, preserving the first
                    # policy that actually crossed the step edge.
                    commands.put(
                        AdaptationCommand(
                            "checkpoint", active_episode, "post_step_reached"
                        )
                    )
                    post_step_checkpointed = True

                performance_monitoring_enabled = (
                    learning_enabled
                    and novelty_decision.actor_update_enabled
                    and active_episode > 0
                    and not paused_now
                )
                # Keep stationary edge samples in critic replay so their
                # negative return is learned, but do not let thousands of
                # near-identical stalled actions dominate the actor buffer.
                actor_replay_eligible = actor_replay_transition_eligible(
                    progress_stalled=progress_decision.stalled,
                    steps_without_progress=progress_decision.steps_without_progress,
                    replay_size=len(online_agent.adaptation_replay),
                    minimum_replay_size=args.adaptation_min_replay_size,
                    stalled_replay_stride=args.stalled_actor_replay_stride,
                )
                adaptation_enabled = (
                    performance_monitoring_enabled
                    and actor_replay_eligible
                    and not final_step_cutoff
                )
                curriculum_stage_id = torch.full_like(
                    terminated,
                    next(
                        index
                        for index, stage in enumerate(protocol.stages)
                        if stage.name == args.terrain_stage
                    ),
                    dtype=torch.int64,
                )
                swing_lift_tracker.reset(done)
                actor_invalid = decision.paused.bool().reshape(-1, 1)
                if control_timed_out or rollback_fallback_used:
                    actor_invalid = torch.ones_like(actor_invalid)
                bootstrap_mask = torch.where(
                    actor_invalid, torch.zeros_like(bootstrap_mask), bootstrap_mask
                )
                learning_terminated, learning_truncated = terminated, truncated
                if task_boundary is not None:
                    learning_terminated, learning_truncated, bootstrap_mask = task_boundary.tensors(terminated)
                trajectory.append(
                    **trajectory_inputs, step=step,
                    **reward_terms,
                    policy_version=-1 if decision.actor_version is None else decision.actor_version,
                    reward=float(reward.mean()), base_x=base_x_value,
                    base_y=float(post_step_snapshot.base_y.mean()),
                    elapsed_s=step * control_step_dt, wall_elapsed_s=time.perf_counter() - trace_started,
                    control_compute_ms=control_ms, projected=bool(decision.projected.any()),
                    paused=paused_now, control_timed_out=control_timed_out,
                    terminated=bool(terminated.any()), truncated=bool(truncated.any()),
                    learning_terminated=bool(learning_terminated.any()), learning_truncated=bool(learning_truncated.any()),
                    bootstrap_mask=bootstrap_mask, completion_reached=completion_boundary_reached,
                    collector_cutoff=(task_boundary.reason == "collector_cutoff" if task_boundary is not None else
                                      step == args.max_steps and not environment_done and not completion_boundary_reached),
                    next_observation=post_step_snapshot.policy_observation, next_frame=next_frame,
                    post_reset_observation=post_reset_snapshot.policy_observation,
                    next_base_action=next_base_action, nonfoot_contact=nonfoot_contact,
                    episode_id=active_episode, reset_count=done_count + int(environment_done),
                    risk_score=decision.risk_score, risk_scale=decision.risk_scale,
                )
                total_reward += float(reward.mean())
                if managed is not None and managed.frozen:
                    online_agent.finish_inference_step(ended=learning_terminated | learning_truncated)
                else:
                    online_agent.finish_step(
                        next_frame=next_frame,
                        next_base_action=next_base_action,
                        reward=reward,
                        terminated=learning_terminated,
                        truncated=learning_truncated,
                        bootstrap_mask=bootstrap_mask,
                        discount=torch.full_like(reward, online_agent.updater.config.gamma),
                        safety_mask=(~actor_invalid).to(dtype=torch.float32),
                        curriculum_stage_id=curriculum_stage_id,
                        episode_id=torch.full_like(curriculum_stage_id, active_episode),
                        adaptation_enabled=adaptation_enabled,
                    )
                committed_global, committed_adaptation = (
                    online_agent.last_committed_transition_counts
                )
                with shared_lock:
                    counters.record_transition_commits(
                        critic_transitions=committed_global,
                        actor_transitions=committed_adaptation,
                    )
                if warmup_step:
                    target_warmup.record_transition()
                else:
                    update_budget.add(
                        critic_transitions=committed_global,
                        actor_transitions=committed_adaptation,
                    )

                if performance_monitoring_enabled:
                    with shared_lock:
                        performance_episode = active_episode
                        successful_actor_updates = counters.actor_updates_for_episode(performance_episode)
                    if successful_actor_updates > 0:
                        adaptation_performance.mark_actor_updated()
                    performance_decision = adaptation_performance.add(
                        velocity_error=velocity_error,
                        tilt_rms=tilt_rms,
                        nonfoot_contact=nonfoot_contact,
                    )
                    if performance_decision is not None:
                        if performance_decision.improved:
                            commands.put(AdaptationCommand("checkpoint", active_episode))
                        elif performance_decision.should_rollback:
                            online_agent.set_residual_gain(resting_residual_gain)
                            rollback_reason = performance_decision.rollback_reason or "performance_degradation"
                            rollback_completion = threading.Event()
                            pending_rollback_completion = rollback_completion
                            rollback_command = AdaptationCommand(
                                "rollback",
                                active_episode,
                                rollback_reason,
                                rollback_completion,
                            )
                            transition_episode_publication(
                                shared_lock=shared_lock,
                                actor_gate_event=actor_gate_event,
                                read_active_episode=read_active_episode,
                                write_active_episode=write_active_episode,
                                next_episode=0,
                                after_transition=lambda episode: commands.put(
                                    replace(rollback_command, episode=episode)
                                ),
                            )
                            novelty_decision = novelty_gate.force_cooldown()
                            with shared_lock:
                                counters.record_rollback(rollback_reason)
                                last_rollback_reason = rollback_reason
                            adaptation_performance.start_episode()
                elif novelty_decision.left_adapting and active_episode > 0:
                    online_agent.set_residual_gain(resting_residual_gain)
                    transition_episode_publication(
                        shared_lock=shared_lock,
                        actor_gate_event=actor_gate_event,
                        read_active_episode=read_active_episode,
                        write_active_episode=write_active_episode,
                        next_episode=0,
                        after_transition=lambda episode: commands.put(
                            AdaptationCommand("end_episode", episode)
                        ),
                    )
                    adaptation_performance.start_episode()

                if final_step_cutoff:
                    online_agent.set_residual_gain(resting_residual_gain)
                    boundary_decision = close_active_episode_for_worker_boundary(
                        active_episode=active_episode,
                        shared_lock=shared_lock,
                        actor_gate_event=actor_gate_event,
                        read_active_episode=read_active_episode,
                        write_active_episode=write_active_episode,
                        enqueue_command=commands.put,
                        novelty_gate=novelty_gate,
                        adaptation_performance=adaptation_performance,
                    )
                    if boundary_decision is not None:
                        novelty_decision = boundary_decision

                # Apply the ramp after novelty/performance transitions so they
                # cannot restore the larger adapting gain during recovery.
                if (
                    learning_enabled
                    and post_step_reached
                    and not safety_reason
                    and not rollback_reason
                    and not final_step_cutoff
                ):
                    online_agent.set_residual_gain(post_step_recovery_gain)

                if learning_enabled and control_step_updates:
                    update_cycle_complete.clear()
                    update_wakeup.set()
                    while not update_cycle_complete.wait(timeout=30.0):
                        if not updater_thread.is_alive() or target_warmup.ready or target_warmup.collecting:
                            raise RuntimeError("control-step SAC update did not complete")
                    if update_errors:
                        raise RuntimeError("control-step SAC updater failed") from update_errors[0]

                with shared_lock:
                    counters.record_state(novelty_decision.state.value)
                in_heldout = base_x_value >= adapter.track_boundary_x
                if in_heldout:
                    heldout_steps += 1
                metrics.update(
                    step=step,
                    in_heldout=in_heldout,
                    velocity_error=velocity_error,
                    tilt_rms=tilt_rms,
                    nonfoot_contact=nonfoot_contact,
                )

                loop_ms = (time.perf_counter() - loop_started) * 1_000.0
                with shared_lock:
                    update_snapshot = dict(last_update)
                    counters_snapshot = asdict(counters)
                    episode_snapshot = active_episode
                    last_rollback_snapshot = last_rollback_reason
                record = {
                    "control_latency": control_latency,
                    "target_warmup_collection": warmup_step,
                    "step": step,
                    "mode": args.mode,
                    "in_heldout": in_heldout,
                    "base_x": base_x_value,
                    "velocity_error": velocity_error,
                    "commanded_forward": bool(args.forward_speed > 0.05),
                    "command_x": float(signals.command_lin_vel_xy[:, 0].mean()),
                    "forward_progress": float(forward_progress.mean()),
                    "tilt_rms": tilt_rms,
                    "nonfoot_contact": nonfoot_contact,
                    "termination_reasons": list(termination_reasons),
                    **contact_diagnostics,
                    "reward": float(reward.mean()),
                    "actor_version": actor_version_for_record(decision),
                    "updates": counters_snapshot["actor_updates"],
                    "critic_updates": counters_snapshot["critic_updates"],
                    "actor_updates": counters_snapshot["actor_updates"],
                    "adaptation_state": novelty_decision.state.value,
                    "stochastic_control_active": not deterministic_control,
                    "novelty_score": novelty_decision.novelty_score,
                    "velocity_excess": novelty_decision.velocity_excess,
                    "tilt_excess": novelty_decision.tilt_excess,
                    "reward_drop": novelty_decision.reward_drop,
                    "progress_excess": novelty_decision.progress_excess,
                    "progress_stalled": progress_decision.stalled,
                    "max_progress_x": progress_decision.max_x,
                    "post_step_reached": post_step_reached,
                    "post_step_recovery_progress": post_step_recovery_progress,
                    "post_step_recovery_active": post_step_recovery_active,
                    "post_step_checkpointed": post_step_checkpointed,
                    "effective_residual_gain": online_agent.residual_gain,
                    "action_residual_gain": action_residual_gain,
                    **risk_gate_record(decision),
                    "pre_step_clearance_active": pre_step_clearance_active,
                    "rough_swing_reward_active": rough_swing_active,
                    "foot_lift_from_contact_m": swing_lift[0].tolist(),
                    "foot_contact": signals.foot_contact[0].tolist(),
                    "joint_target_residual_rad": (decision.residual * safety_state.action_scale)[0].tolist(),
                    "max_foot_height": float(signals.foot_height.max()),
                    "steps_without_progress": progress_decision.steps_without_progress,
                    "actor_update_enabled": actor_gate_event.is_set(),
                    "adaptation_replay_size": len(online_agent.adaptation_replay),
                    "actor_replay_eligible": actor_replay_eligible,
                    "adaptation_episode": episode_snapshot,
                    "performance_evaluation_enabled": adaptation_performance.evaluation_enabled,
                    "actor_updates_in_episode": counters_snapshot["actor_updates_by_episode"].get(
                        episode_snapshot, 0
                    ),
                    **rollback_record_fields(rollback_reason, last_rollback_snapshot),
                    "paused": paused_now,
                    "safety_reasons": list(decision.reasons),
                    "base_action_norm": float(base_action.norm()),
                    "residual_action_norm": float(decision.residual.norm()),
                    "base_final_action_max_abs_diff": float(
                        (decision.action - base_action).abs().max()
                    ),
                    "control_compute_ms": control_ms,
                    "control_timeout_preserved_actor": (
                        active_episode > 0 and control_timed_out
                    ),
                    "loop_ms": loop_ms,
                    "replay_size": len(online_agent.replay),
                    "replay_critic_rows_committed": committed_global,
                    "replay_actor_rows_committed": committed_adaptation,
                    "replay_safety_eligible": bool((~actor_invalid).all()),
                    "last_update": update_snapshot,
                }
                metrics_stream.write(json.dumps(record, allow_nan=True) + "\n")
                metrics_stream.flush()
                writer.add_scalar("run/velocity_error", velocity_error, step)
                writer.add_scalar("run/tilt_rms", tilt_rms, step)
                writer.add_scalar("run/forward_progress", float(forward_progress.mean()), step)
                writer.add_scalar("run/control_compute_ms", control_ms, step)
                writer.add_scalar("run/replay_size", len(online_agent.replay), step)
                writer.add_scalar(
                    "run/terrain_valid",
                    float(terrain_valid.bool().all().item()),
                    step,
                )
                writer.add_scalar(
                    "run/terrain_invalid_steps", counters.terrain_invalid_steps, step
                )
                writer.add_scalar("run/critic_updates", counters_snapshot["critic_updates"], step)
                writer.add_scalar("run/actor_updates", counters_snapshot["actor_updates"], step)

                previous_action = decision.action
                previous_residual = decision.residual
                observation = next_observation
                if task_boundary is not None and task_boundary.stop:
                    done_count += int(done.sum().item())
                    fell = bool(terminated.any())
                    boundary_decision = close_active_episode_for_worker_boundary(
                        active_episode=active_episode, shared_lock=shared_lock,
                        actor_gate_event=actor_gate_event, read_active_episode=read_active_episode,
                        write_active_episode=write_active_episode, enqueue_command=commands.put,
                        novelty_gate=novelty_gate, adaptation_performance=adaptation_performance,
                    )
                    if boundary_decision is not None:
                        novelty_decision = boundary_decision
                    completion_stop = task_boundary.success
                    break
                if done.any().item():
                    done_count += int(done.sum().item())
                    fell = True
                    break
                completion_decision = (
                    None
                    if track_completion is None
                    else evaluate_track_completion_step(
                        track_completion,
                        environment_done=False,
                        safety_violation=safety_violation,
                    )
                )
                if completion_decision is not None and completion_decision.should_stop:
                    boundary_decision = close_active_episode_for_worker_boundary(
                        active_episode=active_episode,
                        shared_lock=shared_lock,
                        actor_gate_event=actor_gate_event,
                        read_active_episode=read_active_episode,
                        write_active_episode=write_active_episode,
                        enqueue_command=commands.put,
                        novelty_gate=novelty_gate,
                        adaptation_performance=adaptation_performance,
                    )
                    if boundary_decision is not None:
                        novelty_decision = boundary_decision
                    completion_stop = True
                    break

            stop_event.set()
            update_wakeup.set()
            if updater_thread is not None:
                updater_thread.join(timeout=30.0)
                if updater_thread.is_alive():
                    raise RuntimeError("background SAC updater did not stop")
            if update_errors:
                raise RuntimeError("background SAC updater failed") from update_errors[0]

            with shared_lock:
                actor_updates_completed = counters.actor_updates
                actor_update_events_logged_snapshot = actor_update_events_logged
            if actor_update_events_logged_snapshot != actor_updates_completed:
                raise RuntimeError(
                    "Actor update event accounting mismatch: "
                    f"logged={actor_update_events_logged_snapshot}, updates={actor_updates_completed}"
                )

            with update_lock:
                online_agent.reset_for_deployment_handoff()
                physical_handoff_prepared = True
                after_parameters = {
                    key: value.detach().cpu().clone()
                    for key, value in online_agent.updater.actor.state_dict().items()
                }
                if learning_enabled:
                    save_experiment_checkpoint(
                        args.experiment_checkpoint_out,
                        {
                            "protocol_sha256": protocol.sha256(),
                            "terrain_metadata": adapter.terrain_metadata(),
                            "training_contract": current_training_contract,
                            "agent": online_agent.state_dict(),
                            "novelty_gate": novelty_gate.state_dict(),
                            "stable_manager": stable_manager.state_dict(),
                            "update_budget": update_budget.snapshot() if managed is not None else None,
                            "target_warmup": target_warmup.state_dict(),
                        },
                    )
                    checkpoint_saved = True
            torch.save(after_parameters, run_dir / "residual_after.pt")
            parameter_changed = any(
                not torch.equal(before_parameters[key], after_parameters[key]) for key in before_parameters
            )
            completion_evidence = (
                track_completion.evidence(
                    fell=fell,
                    safety_violation=safety_violation,
                )
                if track_completion is not None
                else {
                    "initial_x": initial_x,
                    "max_x": float(adapter.base_x().mean().item()),
                    "completion_target_x": None,
                    "completion_remaining_m": None,
                    "completion_reached": False,
                    "completion_mode": "disabled",
                    "progress_cost_version": protocol.progress_cost_version,
                }
            )
            summary = {
                "resting_residual_gain": resting_residual_gain,
                "target_warmup_before": warmup_before,
                "target_warmup": target_warmup.state_dict(),
                "warmup_critic_updates": len(warmup_update_records),
                "warmup_critic_diagnostics": warmup_update_records,
                "mode": args.mode,
                "task_objective": objective_metadata, "task_objective_sha256": objective_sha256,
                "task_stop_reason": None if task_boundary is None else task_boundary.reason,
                "total_reward": total_reward,
                "replay_transitions_written": counters.critic_transitions_committed,
                "steps": executed_steps,
                "reset_count": done_count,
                "fell": fell,
                "updates": counters.actor_updates,
                "critic_updates": counters.critic_updates,
                "actor_updates": counters.actor_updates,
                "update_attempts": counters.update_attempts,
                "update_rejections": dict(counters.update_rejections),
                "update_budget_remaining": update_budget.snapshot(),
                "critic_transitions_committed": counters.critic_transitions_committed,
                "actor_transitions_committed": counters.actor_transitions_committed,
                "critic_updates_per_transition_actual": (
                    counters.critic_updates / counters.critic_transitions_committed
                    if counters.critic_transitions_committed
                    else 0.0
                ),
                "actor_updates_per_transition_actual": (
                    counters.actor_updates / counters.actor_transitions_committed
                    if counters.actor_transitions_committed
                    else 0.0
                ),
                "update_schedule": args.update_schedule,
                "actor_update_events_logged": actor_update_events_logged_snapshot,
                "actor_updates_before_novelty": counters.actor_updates_before_novelty,
                "actor_updates_after_novelty": counters.actor_updates_after_novelty,
                "actor_updates_by_episode": counters.actor_updates_by_episode,
                "first_actor_update_step": counters.first_actor_update_step,
                "novelty_entries": counters.novelty_entries,
                "rollbacks": counters.rollbacks,
                "performance_rollbacks": counters.performance_rollbacks,
                "safety_rollbacks": counters.safety_rollbacks,
                "adaptation_state_durations": counters.state_durations,
                "actor_version": online_agent.updater.version,
                "safety_fallbacks": safety_fallbacks,
                "fall_resets": counters.fall_resets,
                "terrain_feature_source": adapter.terrain_feature_source,
                "terrain_feature_schema_version": adapter.terrain_metadata().get(
                    "terrain_feature_schema_version"
                ),
                "terrain_invalid_steps": counters.terrain_invalid_steps,
                "terrain_steps": counters.terrain_steps,
                "terrain_valid_fraction": (
                    1.0 - counters.terrain_invalid_steps / counters.terrain_steps
                    if counters.terrain_steps
                    else 1.0
                ),
                "safety_violation": safety_violation,
                "parameter_changed": parameter_changed,
                "last_update": dict(last_update),
                "last_rollback_reason": last_rollback_reason,
                "progress_stalled": progress_decision.stalled,
                "max_progress_x": progress_decision.max_x,
                "steps_without_progress": progress_decision.steps_without_progress,
                "terrain_stage": args.terrain_stage,
                "terrain_seed": args.terrain_seed,
                "protocol_sha256": protocol.sha256(),
                "fall_handling": "preserve_updater_reset_episode_v1",
                "contact_termination": CONTACT_TERMINATION_PROTOCOL,
                "termination_reasons": list(termination_reasons),
                **adapter.contact_diagnostics(),
                "residual_gain": online_agent.residual_gain,
                "post_step_checkpointed": post_step_checkpointed,
                **completion_evidence,
                "flat_batch_fraction": online_agent.last_flat_batch_fraction,
                "experiment_checkpoint_in_sha256": (
                    None
                    if args.experiment_checkpoint_in is None
                    else sha256_file(args.experiment_checkpoint_in)
                ),
                "experiment_checkpoint_out_sha256": (
                    None
                    if not learning_enabled
                    else sha256_file(args.experiment_checkpoint_out)
                ),
                **metrics.heldout_summary(),
            }
            trajectory_manifest = trajectory.close(complete=True)
            summary["trajectory"] = {
                "path": "trajectory/manifest.json", "steps": trajectory_manifest["steps"],
                "bytes": trajectory_manifest["bytes"],
            }
            episode_summary = {
                **summary,
                "passed": (
                    evaluate_track_completion_step(
                        track_completion,
                        environment_done=fell,
                        safety_violation=safety_violation,
                    ).passed
                    if track_completion is not None
                    else not fell and not safety_violation
                ),
                "completion_boundary": (
                    track_completion.target_x
                    if track_completion is not None
                    else resolve_track_boundary_x(env_cfg)
                ),
                "interaction_steps": executed_steps,
                "contacts": metrics.heldout_summary().get("heldout_nonfoot_contacts", 0),
                "completion_stop": completion_stop,
            }
            if args.flat_retention_probe:
                flat_summary = summarize_flat_retention(
                    flat_rows, initial_x=initial_x, step_dt=control_step_dt, max_steps=args.max_steps,
                    fell=fell, safety_violation=safety_violation, reset_count=done_count,
                )
                for target in (summary, episode_summary):
                    for key in list(target):
                        if key.startswith("completion_") or key in {"passed", "protocol_sha256", "strict_qualification_enabled"}:
                            target.pop(key, None)
                    target.update(evaluation_kind="flat_retention", flat_retention=flat_summary,
                                  terrain_geometry={"terrain_type": "plane", "terrain_generator": None})
            if managed is not None:
                summary["session_budget"] = managed.snapshot()
                episode_summary["session_budget"] = summary["session_budget"]
            write_episode_summary(run_dir / "episode_summary.json", episode_summary)
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
            )
            if args.novelty_baseline_out is not None:
                write_json_atomic(
                    args.novelty_baseline_out,
                    novelty_gate.export_baseline(
                        source={
                            "seed": args.seed,
                            "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
                        }
                    ),
                )
            writer.flush()
            if managed is not None:
                managed.finish(episode_summary, args.experiment_checkpoint_out if learning_enabled else None)
            return 0 if not fell else 1
        finally:
            stop_event.set()
            if updater_thread is not None and updater_thread.is_alive():
                updater_thread.join(timeout=30.0)
            if (
                learning_enabled
                and not checkpoint_saved
                and online_agent is not None
                and novelty_gate is not None
                and stable_manager is not None
            ):
                if not physical_handoff_prepared:
                    online_agent.reset_for_deployment_handoff()
                    physical_handoff_prepared = True
                save_experiment_checkpoint(
                    args.experiment_checkpoint_out,
                    {
                        "protocol_sha256": protocol.sha256(),
                        "terrain_metadata": adapter.terrain_metadata(),
                        "training_contract": current_training_contract,
                        "agent": online_agent.state_dict(),
                        "novelty_gate": novelty_gate.state_dict(),
                        "stable_manager": stable_manager.state_dict(),
                        "update_budget": update_budget.snapshot() if managed is not None else None,
                        "target_warmup": target_warmup.state_dict(),
                    },
                )
            actor_updates_stream.close()
            metrics_stream.close()
            writer.close()
            wrapped_env.close()
    except Exception as error:
        traceback.print_exc()
        if failure_writer is not None:
            failure_writer(Path(args.run_dir), error)
        return 2
    finally:
        if latency_probe is not None:
            latency_probe.close()
        if trajectory is not None:
            trajectory.close()
        if managed is not None:
            managed.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
