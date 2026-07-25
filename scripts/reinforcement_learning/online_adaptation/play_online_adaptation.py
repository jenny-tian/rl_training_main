"""Run one uninterrupted Lite3 pass over the held-out flat-to-box track."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from isaaclab.app import AppLauncher

from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser


def main() -> int:
    args = build_play_parser().parse_args()
    if args.mode != "base" and args.residual_checkpoint is None:
        raise ValueError(f"--residual-checkpoint is required for mode {args.mode}")
    if "HeldoutBoxes" not in args.task:
        raise ValueError("the online demonstration must use the held-out task")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    updater_thread: threading.Thread | None = None
    stop_event = threading.Event()
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from torch.utils.tensorboard import SummaryWriter

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.metrics import RollingMetrics
        from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.sac import SACConfig, StableStateManager
        from rl_training.online_adaptation.safety import SafetyLimits

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
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(args.base_checkpoint)
            base_policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            policy_module = getattr(runner.alg, "policy", None)
            if policy_module is None:
                policy_module = getattr(runner.alg, "actor_critic")
            policy_module.requires_grad_(False)
            if any(parameter.requires_grad for parameter in policy_module.parameters()):
                raise RuntimeError("base PPO policy must remain frozen")

            device = wrapped_env.unwrapped.device
            normal_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
            actor_limits = torch.zeros_like(normal_limits) if args.mode == "base" else normal_limits
            actor = ResidualActor(49, 12, 64, actor_limits).to(device)
            critic = TwinCritic(49, 12, 64).to(device)
            runtime_config = AgentConfig(
                replay_capacity=args.replay_capacity,
                min_replay_size=args.min_replay_size,
                updates_per_second=args.updates_per_second,
            )
            online_agent = OnlineAdaptationAgent(
                num_envs=1,
                actor=actor,
                critic=critic,
                safety_limits=SafetyLimits(residual_limits=normal_limits),
                config=runtime_config,
                sac_config=SACConfig(),
                device=device,
            )
            if args.residual_checkpoint is not None and args.mode != "base":
                online_agent.updater.load(args.residual_checkpoint, map_location=device)
                online_agent.publisher.publish(online_agent.updater.actor, online_agent.updater.version)

            adapter = Lite3IsaacLabAdapter(
                wrapped_env.unwrapped,
                track_boundary_x=float(getattr(env_cfg, "heldout_start_x", -7.0)),
            )
            reward_weights = RewardWeights()
            metrics = RollingMetrics(window=250)
            stable_manager = StableStateManager()
            before_parameters = {
                key: value.detach().cpu().clone() for key, value in online_agent.updater.actor.state_dict().items()
            }
            torch.save(before_parameters, run_dir / "residual_before.pt")

            shared_lock = threading.Lock()
            update_lock = threading.Lock()
            paused_event = threading.Event()
            update_errors: list[BaseException] = []
            accepted_updates = 0
            rollbacks = 0
            last_update: dict[str, float | int | bool] = {}

            def updater_loop() -> None:
                nonlocal accepted_updates
                period = 1.0 / max(args.updates_per_second, 1e-6)
                try:
                    while not stop_event.is_set():
                        started = time.perf_counter()
                        if online_agent.should_update(
                            len(online_agent.replay), args.min_replay_size, paused_event.is_set()
                        ):
                            with update_lock:
                                stats = online_agent.updater.update(online_agent.sample_update_batch())
                                if stats.accepted:
                                    online_agent.publisher.publish(online_agent.updater.actor, stats.version)
                            with shared_lock:
                                last_update.clear()
                                last_update.update(asdict(stats))
                                if stats.accepted:
                                    accepted_updates += 1
                        stop_event.wait(max(0.0, period - (time.perf_counter() - started)))
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
            previous_residual = torch.zeros_like(previous_action)
            safety_fallbacks = 0
            safety_violation = False
            done_count = 0
            fell = False
            heldout_steps = 0
            executed_steps = 0

            for step in range(1, args.max_steps + 1):
                executed_steps = step
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
                    deterministic=args.mode != "online",
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                control_ms = (time.perf_counter() - control_started) * 1_000.0
                if control_ms > args.control_timeout_ms:
                    timed_out_state = replace(safety_state, control_timed_out=torch.ones_like(timeout))
                    decision = online_agent.fallback_pending(timed_out_state)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    control_ms = (time.perf_counter() - control_started) * 1_000.0
                    safety_fallbacks += 1

                paused_event.set() if decision.paused.any().item() else paused_event.clear()
                residual_limit = normal_limits.unsqueeze(0)
                if (decision.residual.abs() > residual_limit + 1e-6).any().item():
                    safety_violation = True

                with torch.inference_mode():
                    next_observation, _, done, _ = wrapped_env.step(decision.action)
                    next_base_action = base_policy(next_observation)
                next_frame = adapter.frame(decision.action)
                signals = adapter.reward_signals()
                reward = compute_online_reward(signals, decision.residual, previous_residual, reward_weights)
                done_column = done.reshape(-1, 1)
                online_agent.finish_step(next_frame, next_base_action, reward, done_column)

                velocity_error = float(
                    torch.linalg.norm(signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1).mean()
                )
                tilt_rms = float(torch.linalg.norm(signals.projected_gravity_xy, dim=-1).mean())
                nonfoot_contact = int(signals.nonfoot_contact.any().item())
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

                if args.mode == "online" and in_heldout and heldout_steps % 250 == 0:
                    recent = metrics.rolling_records()
                    score = -sum(
                        float(record["velocity_error"]) + float(record["tilt_rms"]) for record in recent
                    ) / max(len(recent), 1)
                    with update_lock:
                        if stable_manager.observe(score, online_agent.updater):
                            online_agent.publisher.publish(
                                online_agent.updater.actor, online_agent.updater.version
                            )
                            rollbacks += 1

                loop_ms = (time.perf_counter() - loop_started) * 1_000.0
                with shared_lock:
                    update_snapshot = dict(last_update)
                    updates_snapshot = accepted_updates
                record = {
                    "step": step,
                    "mode": args.mode,
                    "in_heldout": in_heldout,
                    "velocity_error": velocity_error,
                    "tilt_rms": tilt_rms,
                    "nonfoot_contact": nonfoot_contact,
                    "reward": float(reward.mean()),
                    "actor_version": online_agent.publisher.current()[1],
                    "updates": updates_snapshot,
                    "paused": bool(decision.paused.any().item()),
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
                writer.add_scalar("run/updates", updates_snapshot, step)

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
                "updates": accepted_updates,
                "rollbacks": rollbacks,
                "actor_version": online_agent.updater.version,
                "safety_fallbacks": safety_fallbacks,
                "safety_violation": safety_violation,
                "parameter_changed": parameter_changed,
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
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
