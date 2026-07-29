"""Run one uninterrupted Lite3 pass over the held-out flat-to-box track."""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from isaaclab.app import AppLauncher

from rl_training.online_adaptation.actor_update import update_and_publish_actor
from rl_training.online_adaptation.metrics import AdaptationPerformance
from rl_training.online_adaptation.novelty import NoveltyGate
from rl_training.online_adaptation.policy_drift import ActorDriftMonitor, PolicyDrift
from rl_training.online_adaptation.types import UpdateStats

from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser


@dataclass(frozen=True)
class AdaptationCommand:
    kind: Literal["start_episode", "checkpoint", "rollback", "end_episode"]
    episode: int
    reason: str = ""


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


def apply_adaptation_command(
    command: AdaptationCommand,
    *,
    checkpointed_episode: int,
    updater: Any,
    publisher: Any,
    stable_manager: Any,
    drift_monitor: ActorDriftMonitor,
    actor_gate_event: Any,
    episode_is_active: Callable[[int], bool],
) -> tuple[int, str | None]:
    if command.kind == "start_episode":
        stable_manager.checkpoint(updater)
        drift_monitor.start_episode(updater.actor, adaptation_episode=command.episode)
        checkpointed_episode = command.episode
        if episode_is_active(command.episode):
            actor_gate_event.set()
    elif command.kind == "checkpoint":
        if command.episode == checkpointed_episode:
            stable_manager.checkpoint(updater)
    elif command.kind == "rollback":
        if command.episode != checkpointed_episode:
            return checkpointed_episode, None
        if stable_manager.rollback(updater):
            publisher.publish(updater.actor, updater.version)
        drift_monitor.clear_episode(adaptation_episode=command.episode)
        checkpointed_episode = 0
        actor_gate_event.clear()
        return checkpointed_episode, command.reason
    elif command.kind == "end_episode":
        if command.episode != checkpointed_episode:
            return checkpointed_episode, None
        drift_monitor.clear_episode(adaptation_episode=command.episode)
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
) -> bool:
    if stats.rejection_reason == "stale_adaptation_episode":
        return False
    last_update.clear()
    last_update.update(asdict(stats))
    if policy_drift is not None:
        last_update.update(policy_drift_record(policy_drift))
    if stats.accepted and stats.critic_updated:
        counters.record_critic_update()
    if stats.accepted and stats.actor_updated:
        counters.record_actor_update(step=step, adaptation_episode=adaptation_episode)
    return True


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
        base_action = base_policy(observation)
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
                "rollback_reason": rollback_reason or self.last_rollback_reason,
                "performance_evaluation_enabled": self.adaptation_performance.evaluation_enabled,
            },
        )

    def record_critic_update(self) -> None:
        self.counters.record_critic_update()

    def record_actor_update(self, *, step: int) -> None:
        self.counters.record_actor_update(step=step, adaptation_episode=self.active_episode)


def main() -> int:
    args = build_play_parser().parse_args()
    if args.mode != "base" and args.residual_checkpoint is None:
        raise ValueError(f"--residual-checkpoint is required for mode {args.mode}")
    if "HeldoutBoxes" not in args.task:
        raise ValueError("the online demonstration must use the held-out task")
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

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.metrics import AdaptationPerformance, RollingMetrics
        from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
        from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate, NoveltyState
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.rsl_rl_compat import freeze_rsl_rl_models, prepare_rsl_rl_agent_cfg
        from rl_training.online_adaptation.runtime import (
            as_torch_device,
            safety_violation_occurred,
            write_failure_record,
        )
        from rl_training.online_adaptation.sac import SACConfig, StableStateManager
        from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits

        failure_writer = write_failure_record

        torch.manual_seed(args.seed)
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir))
        metrics_stream = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
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
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(run_dir / "videos"),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length or args.max_steps,
                disable_logger=True,
            )
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
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
            )
            safety_limits = SafetyLimits(residual_limits=normal_limits)
            online_agent = OnlineAdaptationAgent(
                num_envs=1,
                actor=actor,
                critic=critic,
                safety_limits=safety_limits,
                config=runtime_config,
                sac_config=SACConfig(),
                device=device,
            )
            if args.residual_checkpoint is not None and args.mode != "base":
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
            last_rollback_reason = ""

            def updater_loop() -> None:
                nonlocal last_rollback_reason
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
                                def command_episode_is_active(episode: int) -> bool:
                                    with shared_lock:
                                        return active_episode == episode

                                checkpointed_episode, command_rollback_reason = apply_adaptation_command(
                                    command,
                                    checkpointed_episode=checkpointed_episode,
                                    updater=online_agent.updater,
                                    publisher=online_agent.publisher,
                                    stable_manager=stable_manager,
                                    drift_monitor=actor_drift_monitor,
                                    actor_gate_event=actor_gate_event,
                                    episode_is_active=command_episode_is_active,
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

                            def episode_is_still_active() -> bool:
                                with shared_lock:
                                    return active_episode == episode_snapshot

                            with update_lock:
                                outcome = update_and_publish_actor(
                                    updater=online_agent.updater,
                                    publisher=online_agent.publisher,
                                    drift_monitor=actor_drift_monitor,
                                    critic_batch=critic_batch,
                                    actor_batch=actor_batch,
                                    adaptation_episode=episode_snapshot,
                                    update_step=step_snapshot,
                                    episode_is_active=episode_is_still_active,
                                )
                            stats = outcome.stats
                            policy_drift = outcome.policy_drift
                            next_actor = now + actor_period
                            next_critic = max(next_critic + critic_period, now)
                        elif now >= next_critic and online_agent.should_update(
                            len(online_agent.replay), args.min_replay_size, paused_event.is_set()
                        ):
                            with update_lock:
                                stats = online_agent.updater.update(
                                    critic_batch=online_agent.sample_critic_batch(),
                                    update_actor=False,
                                )
                            next_critic = now + critic_period

                        if stats is not None and stats.rejection_reason != "stale_adaptation_episode":
                            with shared_lock:
                                record_update_result(
                                    stats=stats,
                                    policy_drift=policy_drift,
                                    last_update=last_update,
                                    counters=counters,
                                    step=step_snapshot,
                                    adaptation_episode=episode_snapshot,
                                )
                        stop_event.wait(0.001)
                except BaseException as error:
                    update_errors.append(error)
                    stop_event.set()

            if args.mode == "online":
                updater_thread = threading.Thread(target=updater_loop, name="lite3-sac-updater", daemon=False)
                updater_thread.start()

            config = {
                "arguments": vars(args),
                "agent": asdict(runtime_config),
                "reward": asdict(reward_weights),
                "continuous_no_reset": True,
                "heldout_mask_usage": "metrics_only",
            }
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
            )

            observation, _ = wrapped_env.reset()
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
                if update_errors:
                    raise RuntimeError("background SAC updater failed") from update_errors[0]
                loop_started = time.perf_counter()
                control_started = time.perf_counter()
                frame = adapter.frame(previous_action)
                with torch.inference_mode():
                    base_action = base_policy(observation)
                timeout = torch.zeros(1, 1, dtype=torch.bool, device=device)
                safety_state = adapter.safety_state(timeout)
                decision = online_agent.begin_step(
                    frame,
                    base_action,
                    safety_state,
                    deterministic=not args.stochastic_control,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                control_ms = (time.perf_counter() - control_started) * 1_000.0
                control_timed_out = False
                if control_ms > args.control_timeout_ms:
                    control_timed_out = True
                    timed_out_state = replace(safety_state, control_timed_out=torch.ones_like(timeout))
                    decision = online_agent.fallback_pending(timed_out_state)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    control_ms = (time.perf_counter() - control_started) * 1_000.0
                    safety_fallbacks += 1

                paused_now = bool(decision.paused.any().item())
                if online_agent.all_environments_paused(decision.paused):
                    paused_event.set()
                else:
                    paused_event.clear()
                if safety_violation_occurred(decision.paused, control_timed_out):
                    safety_violation = True

                with torch.inference_mode():
                    next_observation, _, done, _ = wrapped_env.step(decision.action)
                    next_base_action = base_policy(next_observation)
                next_frame = adapter.frame(decision.action)
                signals = adapter.reward_signals()
                reward = compute_online_reward(signals, decision.residual, previous_residual, reward_weights)
                done_column = done.reshape(-1, 1)
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
                    episode_to_rollback = active_episode
                    novelty_decision = novelty_gate.force_cooldown()
                    actor_gate_event.clear()
                    online_agent.clear_adaptation_replay()
                    adaptation_performance.start_episode()
                    with shared_lock:
                        active_episode = 0
                        counters.record_rollback(safety_reason)
                    commands.put(AdaptationCommand("rollback", episode_to_rollback, safety_reason))
                    rollback_reason = safety_reason
                else:
                    novelty_decision = novelty_gate.observe(
                        velocity_error,
                        tilt_rms,
                        bool(nonfoot_contact),
                        float(reward.mean()),
                        paused=paused_now,
                    )

                if args.mode == "online" and novelty_decision.entered_adapting:
                    actor_gate_event.clear()
                    online_agent.clear_adaptation_replay()
                    adaptation_performance.start_episode()
                    adaptation_episode += 1
                    with shared_lock:
                        active_episode = adaptation_episode
                        counters.record_novelty_entry()
                    commands.put(AdaptationCommand("start_episode", adaptation_episode))

                adaptation_enabled = (
                    args.mode == "online"
                    and novelty_decision.actor_update_enabled
                    and active_episode > 0
                    and not paused_now
                )
                online_agent.finish_step(
                    next_frame,
                    next_base_action,
                    reward,
                    done_column,
                    adaptation_enabled=adaptation_enabled,
                )

                if adaptation_enabled:
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
                            episode_to_rollback = active_episode
                            rollback_reason = performance_decision.rollback_reason or "performance_degradation"
                            novelty_decision = novelty_gate.force_cooldown()
                            actor_gate_event.clear()
                            online_agent.clear_adaptation_replay()
                            with shared_lock:
                                active_episode = 0
                                counters.record_rollback(rollback_reason)
                            commands.put(
                                AdaptationCommand("rollback", episode_to_rollback, rollback_reason)
                            )
                            adaptation_performance.start_episode()
                elif novelty_decision.left_adapting and active_episode > 0:
                    episode_to_end = active_episode
                    actor_gate_event.clear()
                    online_agent.clear_adaptation_replay()
                    adaptation_performance.start_episode()
                    with shared_lock:
                        active_episode = 0
                    commands.put(AdaptationCommand("end_episode", episode_to_end))

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
                    rollback_snapshot = rollback_reason or last_rollback_reason
                record = {
                    "step": step,
                    "mode": args.mode,
                    "in_heldout": in_heldout,
                    "base_x": float(adapter.base_x().mean()),
                    "velocity_error": velocity_error,
                    "tilt_rms": tilt_rms,
                    "nonfoot_contact": nonfoot_contact,
                    "reward": float(reward.mean()),
                    "actor_version": online_agent.publisher.current()[1],
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
                    "rollback_reason": rollback_snapshot,
                    "paused": paused_now,
                    "safety_reasons": list(decision.reasons),
                    "base_action_norm": float(base_action.norm()),
                    "residual_action_norm": float(decision.residual.norm()),
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

            with update_lock:
                after_parameters = {
                    key: value.detach().cpu().clone()
                    for key, value in online_agent.updater.actor.state_dict().items()
                }
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
                **metrics.heldout_summary(),
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
            )
            writer.flush()
            return 0 if not fell else 1
        finally:
            stop_event.set()
            if updater_thread is not None and updater_thread.is_alive():
                updater_thread.join(timeout=30.0)
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
