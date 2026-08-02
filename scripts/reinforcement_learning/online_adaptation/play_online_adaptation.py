"""Run one uninterrupted Lite3 pass over the held-out flat-to-box track."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from isaaclab.app import AppLauncher

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.actor_update import update_and_publish_actor
from rl_training.online_adaptation.metrics import AdaptationPerformance
from rl_training.online_adaptation.novelty import NoveltyGate
from rl_training.online_adaptation.policy_drift import ActorDriftMonitor, PolicyDrift
from rl_training.online_adaptation.rsl_rl_compat import split_rsl_rl_endings
from rl_training.online_adaptation.terrain_protocol import (
    TerrainGeometry,
    apply_geometry_to_env_cfg,
    default_flat_online_protocol,
    development_geometry,
    sealed_geometry,
)
from rl_training.online_adaptation.types import UpdateStats

from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser


@dataclass(frozen=True)
class AdaptationCommand:
    kind: Literal["start_episode", "checkpoint", "rollback", "end_episode"]
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
    state_durations: dict[str, int] = field(default_factory=dict)
    actor_updates_by_episode: dict[int, int] = field(default_factory=dict)

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


@dataclass(frozen=True)
class RuntimeStepResult:
    adaptation_enabled: bool
    clear_adaptation_replay: bool
    commands: tuple[AdaptationCommand, ...]
    record_fields: dict[str, Any]


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


def write_actor_update_event(stream: Any, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()


def resolve_worker_geometry(
    terrain_stage: str, terrain_seed: int, *, evaluation_only: bool
) -> TerrainGeometry:
    if evaluation_only:
        return sealed_geometry(terrain_seed)
    return development_geometry(terrain_stage, terrain_seed)


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
) -> tuple[int, str | None]:
    if command.kind == "start_episode":
        clear_adaptation_replay()
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
        clear_adaptation_replay()
        checkpointed_episode = 0
        actor_gate_event.clear()
        if command.completion is not None:
            command.completion.set()
        return checkpointed_episode, command.reason
    elif command.kind == "end_episode":
        if command.episode != checkpointed_episode:
            return checkpointed_episode, None
        drift_monitor.clear_episode(adaptation_episode=command.episode)
        clear_adaptation_replay()
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
    update_record = asdict(stats)
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


def resolve_track_boundary_x(env_cfg: Any) -> float:
    if hasattr(env_cfg, "track_boundary_x"):
        return float(env_cfg.track_boundary_x)
    return float(getattr(env_cfg, "heldout_start_x", -7.0))


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
        safety_reason = ""

        if self.active_episode > 0:
            if control_timed_out:
                safety_reason = "control_timeout"
            elif paused:
                safety_reason = safety_reasons[0] if safety_reasons else "safety_pause"
            elif done:
                safety_reason = "environment_termination"

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
        else:
            novelty_decision = self.novelty_gate.observe(
                velocity_error,
                tilt_rms,
                bool(nonfoot_contact),
                reward,
                paused=paused,
            )

        if self.mode == "online" and novelty_decision.entered_adapting:
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
            },
        )

    def record_critic_update(self) -> None:
        self.counters.record_critic_update()

    def record_actor_update(self, *, step: int) -> None:
        self.counters.record_actor_update(step=step, adaptation_episode=self.active_episode)


def main() -> int:
    args = build_play_parser().parse_args()
    default_protocol = default_flat_online_protocol()
    protocol = replace(
        default_protocol,
        stages=tuple(
            replace(stage, max_episodes=args.protocol_max_stage_episodes)
            for stage in default_protocol.stages
        ),
        max_total_episodes=args.protocol_max_episodes,
        max_total_steps=args.protocol_max_total_steps,
    )
    if args.protocol_sha256 != protocol.sha256():
        raise ValueError("--protocol-sha256 does not match the runtime protocol")
    geometry = (
        TerrainGeometry(step_height=0.0, rough_amplitude=0.0, seed=args.terrain_seed)
        if args.flat_retention_probe
        else resolve_worker_geometry(
            args.terrain_stage, args.terrain_seed, evaluation_only=args.evaluation_only
        )
    )
    learning_enabled = args.mode == "online" and not args.evaluation_only
    if (
        args.mode != "base"
        and args.residual_checkpoint is None
        and args.experiment_checkpoint_in is None
        and not (args.mode == "online" and args.flat_replay is not None)
        and not (args.mode == "residual_frozen" and args.residual_gain == 0.0)
    ):
        raise ValueError(
            f"--residual-checkpoint or --experiment-checkpoint-in is required for mode {args.mode}"
        )
    if args.task != "FlatOnline-Deeprobotics-Lite3-v0":
        raise ValueError("the curriculum worker must use the FlatOnline Lite3 task")
    if args.updates_per_second <= 0.0 or args.actor_updates_per_second <= 0.0:
        raise ValueError("critic and actor update frequencies must be positive")
    if args.video:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    updater_thread: threading.Thread | None = None
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
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.deployment_startup import make_lite3_deployment_env
        from rl_training.online_adaptation.metrics import AdaptationPerformance, RollingMetrics
        from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
        from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate, NoveltyState
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.replay_buffer import ReplayBuffer
        from rl_training.online_adaptation.rsl_rl_compat import (
            freeze_rsl_rl_models,
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
        from rl_training.online_adaptation.provenance import sha256_file

        failure_writer = write_failure_record

        torch.manual_seed(args.seed)
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir))
        metrics_stream = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")
        actor_updates_stream = (run_dir / "actor_updates.jsonl").open("w", encoding="utf-8")

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        apply_geometry_to_env_cfg(env_cfg, geometry)
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.terminations.terrain_out_of_bounds = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.forward_speed, args.forward_speed)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg = prepare_rsl_rl_agent_cfg(agent_cfg)
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
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(args.base_checkpoint)
            base_policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            freeze_rsl_rl_models(runner)

            device = as_torch_device(wrapped_env.unwrapped.device)
            normal_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
            actor_limits = torch.zeros_like(normal_limits) if args.mode == "base" else normal_limits
            actor = ResidualActor(49, 12, 64, actor_limits).to(device)
            critic = TwinCritic(49, 12, 64).to(device)
            runtime_config = AgentConfig(
                replay_capacity=args.replay_capacity,
                min_replay_size=args.min_replay_size,
                updates_per_second=args.updates_per_second,
                residual_gain=0.0,
                n_step=args.n_step,
                flat_fraction_initial=args.flat_fraction_initial,
                flat_fraction_ready=args.flat_fraction_ready,
                flat_critic_warmup_updates=args.flat_critic_warmup_updates,
                critic_replay_mode=args.critic_replay_mode,
            )
            safety_limits = SafetyLimits(residual_limits=normal_limits)
            online_agent = OnlineAdaptationAgent(
                num_envs=1,
                actor=actor,
                critic=critic,
                safety_limits=safety_limits,
                config=runtime_config,
                sac_config=SACConfig(flat_anchor_weight=args.flat_anchor_weight),
                device=device,
            )
            if args.flat_replay is not None:
                flat_state = torch.load(
                    args.flat_replay, map_location="cpu", weights_only=False
                )
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
            restored_experiment = None
            if args.experiment_checkpoint_in is not None:
                restored_experiment = load_experiment_checkpoint(
                    args.experiment_checkpoint_in,
                    expected_protocol_sha256=protocol.sha256(),
                    map_location=device,
                )
                online_agent.load_state_dict(restored_experiment["agent"])
            if (
                args.residual_checkpoint is not None
                and args.mode != "base"
                and restored_experiment is None
            ):
                online_agent.updater.load(args.residual_checkpoint, map_location=device)
                online_agent.publisher.publish(online_agent.updater.actor, online_agent.updater.version)

            adapter = Lite3IsaacLabAdapter(
                wrapped_env.unwrapped,
                track_boundary_x=resolve_track_boundary_x(env_cfg),
            )
            reward_weights = RewardWeights()
            metrics = RollingMetrics(window=250)
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
            if learning_enabled and online_agent.flat_replay is not None:
                online_agent.prewarm_critics_from_flat()
            if restored_experiment is None:
                stable_manager.checkpoint(online_agent.updater)
            before_parameters = {
                key: value.detach().cpu().clone() for key, value in online_agent.updater.actor.state_dict().items()
            }
            torch.save(before_parameters, run_dir / "residual_before.pt")

            shared_lock = threading.Lock()
            update_lock = threading.Lock()
            paused_event = threading.Event()
            actor_gate_event = threading.Event()
            commands: queue.SimpleQueue[AdaptationCommand] = queue.SimpleQueue()
            update_errors: list[BaseException] = []
            counters = RuntimeCounters()
            adaptation_episode = 0
            active_episode = 0
            current_step = 0
            last_update: dict[str, Any] = {}
            actor_update_events_logged = 0
            last_rollback_reason = ""
            pending_rollback_completion: threading.Event | None = None

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
                        while True:
                            try:
                                command = commands.get_nowait()
                            except queue.Empty:
                                break
                            with update_lock:
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
                                )
                                if command_rollback_reason is not None:
                                    with shared_lock:
                                        last_rollback_reason = command_rollback_reason

                        now = time.perf_counter()
                        with shared_lock:
                            episode_snapshot = active_episode
                            step_snapshot = current_step
                        actor_due = (
                            now >= next_actor
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

                            with update_lock:
                                outcome = update_and_publish_actor(
                                    updater=online_agent.updater,
                                    publisher=online_agent.publisher,
                                    drift_monitor=actor_drift_monitor,
                                    critic_batch=critic_batch,
                                    actor_batch=actor_batch,
                                    flat_anchor_batch=flat_anchor_batch,
                                    adaptation_episode=episode_snapshot,
                                    update_step=step_snapshot,
                                    episode_is_active=episode_is_still_active,
                                    final_publish_attempt=publish_if_episode_is_still_active,
                                )
                            stats = replace(
                                outcome.stats,
                                flat_batch_fraction=online_agent.last_flat_batch_fraction,
                                n_step=online_agent.config.n_step,
                            )
                            policy_drift = outcome.policy_drift
                            next_actor = now + actor_period
                            next_critic = max(next_critic + critic_period, now)
                        elif now >= next_critic and online_agent.should_update(
                            len(online_agent.replay), args.min_replay_size, paused_event.is_set()
                        ):
                            with update_lock:
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
                        stop_event.wait(0.001)
                except BaseException as error:
                    update_errors.append(error)
                    stop_event.set()

            if learning_enabled:
                updater_thread = threading.Thread(target=updater_loop, name="lite3-sac-updater", daemon=False)
                updater_thread.start()

            config = {
                "arguments": vars(args),
                "agent": asdict(runtime_config),
                "reward": asdict(reward_weights),
                "continuous_no_reset": True,
                "heldout_mask_usage": "metrics_only",
                "terrain_stage": args.terrain_stage,
                "terrain_seed": args.terrain_seed,
                "protocol_sha256": protocol.sha256(),
                "geometry": asdict(geometry),
            }
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
            )

            observation, _ = wrapped_env.reset()
            generation = wrapped_env.consume_startup_generation()
            if generation is None:
                raise RuntimeError("deployment startup reset did not produce a generation")
            online_agent.reset_for_deployment_handoff()
            previous_action = torch.zeros(1, 12, device=device)
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
            online_agent.set_residual_gain(args.residual_gain)
            previous_residual = torch.zeros_like(previous_action)
            safety_fallbacks = 0
            safety_violation = False
            done_count = 0
            fell = False
            heldout_steps = 0
            executed_steps = 0
            novelty_decision = novelty_gate.observe(0.0, 0.0, False, 0.0, paused=True)

            for step in range(1, args.max_steps + 1):
                executed_steps = step
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
                control_started = time.perf_counter()
                frame = adapter.frame(previous_action)
                with torch.inference_mode():
                    base_action = clip_lite3_policy_action(base_policy(observation))
                timeout = torch.zeros(1, 1, dtype=torch.bool, device=device)
                safety_state = adapter.safety_state(timeout)
                decision = online_agent.begin_step(
                    frame,
                    base_action,
                    safety_state,
                    deterministic=not args.stochastic_control,
                )
                decision, rollback_fallback_used = enforce_prepared_rollback_fallback(
                    online_agent=online_agent,
                    decision=decision,
                    safety_state=safety_state,
                    fallback_required=rollback_fallback_required,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                control_ms = (time.perf_counter() - control_started) * 1_000.0
                control_timed_out = False
                if control_ms > args.control_timeout_ms:
                    control_timed_out = True
                    timed_out_state = replace(safety_state, control_timed_out=torch.ones_like(timeout))
                    decision = enforce_control_timeout_fallback(
                        online_agent=online_agent,
                        timed_out_state=timed_out_state,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    control_ms = (time.perf_counter() - control_started) * 1_000.0
                if rollback_fallback_used or control_timed_out:
                    safety_fallbacks += 1
                if update_errors:
                    raise RuntimeError("background SAC updater failed") from update_errors[0]

                paused_now = bool(decision.paused.any().item())
                if online_agent.all_environments_paused(decision.paused):
                    paused_event.set()
                else:
                    paused_event.clear()
                if safety_violation_occurred(decision.paused, control_timed_out):
                    safety_violation = True

                with torch.inference_mode():
                    step_result = wrapped_env.step(decision.action)
                    next_observation = step_result[0]
                    terminated, truncated, bootstrap_mask = split_worker_endings(
                        step_result, target_semantics=args.target_semantics
                    )
                    done = terminated | truncated
                    next_base_action = clip_lite3_policy_action(base_policy(next_observation))
                next_frame = adapter.frame(decision.action)
                signals = adapter.reward_signals()
                reward = compute_online_reward(signals, decision.residual, previous_residual, reward_weights)
                velocity_error = float(
                    torch.linalg.norm(signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1).mean()
                )
                tilt_rms = float(torch.linalg.norm(signals.projected_gravity_xy, dim=-1).mean())
                nonfoot_contact = int(signals.nonfoot_contact.any().item())
                rollback_reason = ""
                safety_reason = ""
                if active_episode > 0:
                    if control_timed_out:
                        safety_reason = "control_timeout"
                    elif paused_now:
                        safety_reason = decision.reasons[0] if decision.reasons else "safety_pause"
                    elif done.any().item():
                        safety_reason = "environment_termination"
                if safety_reason:
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
                else:
                    novelty_decision = novelty_gate.observe(
                        velocity_error,
                        tilt_rms,
                        bool(nonfoot_contact),
                        float(reward.mean()),
                        paused=paused_now,
                    )

                if (
                    learning_enabled
                    and novelty_decision.entered_adapting
                    and not final_step_cutoff
                ):
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

                performance_monitoring_enabled = (
                    learning_enabled
                    and novelty_decision.actor_update_enabled
                    and active_episode > 0
                    and not paused_now
                )
                adaptation_enabled = performance_monitoring_enabled and not final_step_cutoff
                curriculum_stage_id = torch.full_like(
                    terminated,
                    next(
                        index
                        for index, stage in enumerate(protocol.stages)
                        if stage.name == args.terrain_stage
                    ),
                    dtype=torch.int64,
                )
                actor_invalid = decision.paused.bool().reshape(-1, 1)
                if control_timed_out or rollback_fallback_used:
                    actor_invalid = torch.ones_like(actor_invalid)
                bootstrap_mask = torch.where(
                    actor_invalid, torch.zeros_like(bootstrap_mask), bootstrap_mask
                )
                online_agent.finish_step(
                    next_frame=next_frame,
                    next_base_action=next_base_action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    bootstrap_mask=bootstrap_mask,
                    discount=torch.full_like(reward, online_agent.updater.config.gamma),
                    safety_mask=(~actor_invalid).to(dtype=torch.float32),
                    curriculum_stage_id=curriculum_stage_id,
                    episode_id=torch.full_like(curriculum_stage_id, active_episode),
                    adaptation_enabled=adaptation_enabled,
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

                if final_step_cutoff and active_episode > 0:
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

                with shared_lock:
                    counters.record_state(novelty_decision.state.value)
                in_heldout = bool(adapter.heldout_mask().item())
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
                    "step": step,
                    "mode": args.mode,
                    "in_heldout": in_heldout,
                    "base_x": float(adapter.base_x().mean()),
                    "velocity_error": velocity_error,
                    "tilt_rms": tilt_rms,
                    "nonfoot_contact": nonfoot_contact,
                    "reward": float(reward.mean()),
                    "actor_version": actor_version_for_record(decision),
                    "updates": counters_snapshot["actor_updates"],
                    "critic_updates": counters_snapshot["critic_updates"],
                    "actor_updates": counters_snapshot["actor_updates"],
                    "adaptation_state": novelty_decision.state.value,
                    "novelty_score": novelty_decision.novelty_score,
                    "velocity_excess": novelty_decision.velocity_excess,
                    "tilt_excess": novelty_decision.tilt_excess,
                    "reward_drop": novelty_decision.reward_drop,
                    "actor_update_enabled": actor_gate_event.is_set(),
                    "adaptation_replay_size": len(online_agent.adaptation_replay),
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
                    "loop_ms": loop_ms,
                    "replay_size": len(online_agent.replay),
                    "last_update": update_snapshot,
                }
                metrics_stream.write(json.dumps(record, allow_nan=True) + "\n")
                metrics_stream.flush()
                writer.add_scalar("run/velocity_error", velocity_error, step)
                writer.add_scalar("run/tilt_rms", tilt_rms, step)
                writer.add_scalar("run/control_compute_ms", control_ms, step)
                writer.add_scalar("run/replay_size", len(online_agent.replay), step)
                writer.add_scalar("run/critic_updates", counters_snapshot["critic_updates"], step)
                writer.add_scalar("run/actor_updates", counters_snapshot["actor_updates"], step)

                previous_action = decision.action
                previous_residual = decision.residual
                observation = next_observation
                if done.any().item():
                    done_count += int(done.sum().item())
                    fell = True
                    break

            stop_event.set()
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
                after_parameters = {
                    key: value.detach().cpu().clone()
                    for key, value in online_agent.updater.actor.state_dict().items()
                }
                if learning_enabled:
                    save_experiment_checkpoint(
                        args.experiment_checkpoint_out,
                        {
                            "protocol_sha256": protocol.sha256(),
                            "agent": online_agent.state_dict(),
                            "novelty_gate": novelty_gate.state_dict(),
                            "stable_manager": stable_manager.state_dict(),
                        },
                    )
                    checkpoint_saved = True
            torch.save(after_parameters, run_dir / "residual_after.pt")
            parameter_changed = any(
                not torch.equal(before_parameters[key], after_parameters[key]) for key in before_parameters
            )
            summary = {
                "mode": args.mode,
                "steps": executed_steps,
                "reset_count": done_count,
                "fell": fell,
                "updates": counters.actor_updates,
                "critic_updates": counters.critic_updates,
                "actor_updates": counters.actor_updates,
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
                "safety_violation": safety_violation,
                "parameter_changed": parameter_changed,
                "last_update": dict(last_update),
                "last_rollback_reason": last_rollback_reason,
                "terrain_stage": args.terrain_stage,
                "terrain_seed": args.terrain_seed,
                "protocol_sha256": protocol.sha256(),
                "residual_gain": online_agent.residual_gain,
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
            episode_summary = {
                **summary,
                "passed": not fell and not safety_violation,
                "completion_boundary": resolve_track_boundary_x(env_cfg),
                "interaction_steps": executed_steps,
                "contacts": metrics.heldout_summary().get("heldout_nonfoot_contacts", 0),
            }
            write_episode_summary(run_dir / "episode_summary.json", episode_summary)
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
            )
            writer.flush()
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
                save_experiment_checkpoint(
                    args.experiment_checkpoint_out,
                    {
                        "protocol_sha256": protocol.sha256(),
                        "agent": online_agent.state_dict(),
                        "novelty_gate": novelty_gate.state_dict(),
                        "stable_manager": stable_manager.state_dict(),
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
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
