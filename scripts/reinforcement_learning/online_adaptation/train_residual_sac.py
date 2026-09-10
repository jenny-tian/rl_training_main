"""Pretrain the Lite3 residual SAC policy without held-out terrain exposure."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

from scripts.reinforcement_learning.online_adaptation.cli import build_train_parser


def main() -> int:
    args = build_train_parser().parse_args()
    if "HeldoutBoxes" in args.task:
        raise ValueError("held-out terrain is forbidden during residual pretraining")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import torch
        import yaml
        from torch.utils.tensorboard import SummaryWriter

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.terrain_observation import HeightScanConfig
        from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.rsl_rl_compat import (
            freeze_rsl_rl_models,
            load_frozen_rsl_rl_checkpoint,
            prepare_rsl_rl_agent_cfg,
            split_rsl_rl_endings,
        )
        from rl_training.online_adaptation.replay_buffer import (
            ReplayBuffer,
            adapt_replay_state_frame_dim,
        )
        from rl_training.online_adaptation.sac import SACConfig
        from rl_training.online_adaptation.safety import SafetyLimits
        from rl_training.online_adaptation.terrain_guards import assert_heldout_terrains_disabled

        torch.manual_seed(args.seed)
        output_dir = Path(args.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.forward_speed, args.forward_speed)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        assert_heldout_terrains_disabled(env_cfg.scene.terrain.terrain_generator)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg = prepare_rsl_rl_agent_cfg(agent_cfg)
        env = gym.make(args.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        writer = SummaryWriter(log_dir=str(output_dir))
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            checkpoint_load = load_frozen_rsl_rl_checkpoint(runner, args.base_checkpoint)
            policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            freeze_rsl_rl_models(runner)

            device = wrapped_env.unwrapped.device
            action_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
            adapter = Lite3IsaacLabAdapter(
                wrapped_env.unwrapped,
                step_height=args.terrain_step_height,
                terrain_feature_source=args.terrain_feature_source,
                height_scan_config=HeightScanConfig(
                    rows=23,
                    cols=15,
                    min_valid_fraction=args.terrain_min_valid_fraction,
                ),
            )
            if not hasattr(wrapped_env.unwrapped, "add_pre_reset_callback"):
                raise RuntimeError("training environment must support pre-reset terminal capture")
            frame_dim = adapter.frame_dim
            actor = ResidualActor(frame_dim, 12, 64, action_limits).to(device)
            critic = TwinCritic(frame_dim, 12, 64).to(device)
            runtime_config = AgentConfig(
                frame_dim=frame_dim,
                replay_capacity=args.replay_capacity,
                min_replay_size=args.min_replay_size,
                residual_gain=args.residual_gain,
                n_step=args.n_step,
                flat_fraction_initial=args.flat_fraction_initial,
                flat_fraction_ready=args.flat_fraction_ready,
                flat_critic_warmup_updates=args.flat_critic_warmup_updates,
                critic_replay_mode="dual",
            )
            online_agent = OnlineAdaptationAgent(
                num_envs=args.num_envs,
                actor=actor,
                critic=critic,
                safety_limits=SafetyLimits(residual_limits=action_limits),
                config=runtime_config,
                sac_config=SACConfig(flat_anchor_weight=args.flat_anchor_weight),
                device=device,
            )
            if args.flat_replay is not None:
                flat_state = torch.load(args.flat_replay, map_location="cpu", weights_only=False)
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
                online_agent.prewarm_critics_from_flat()
            if args.resume_checkpoint is not None:
                online_agent.updater.load(args.resume_checkpoint, map_location=device)
                online_agent.publisher.publish(online_agent.updater.actor, online_agent.updater.version)
            reward_weights = RewardWeights(residual_weight=0.25, residual_rate_weight=0.10)

            config_record = {
                "arguments": vars(args),
                "agent": asdict(runtime_config),
                "sac": asdict(online_agent.updater.config),
                "reward": asdict(reward_weights),
                "terrain_leakage_assertion": "passed",
                "terrain_metadata": adapter.terrain_metadata(),
                "base_checkpoint_load": checkpoint_load,
                "timeout_target_observation": "pre_reset_terminal",
            }
            (output_dir / "config.json").write_text(
                json.dumps(config_record, indent=2, default=str) + "\n", encoding="utf-8"
            )
            (output_dir / "config.yaml").write_text(
                yaml.safe_dump(config_record, sort_keys=False), encoding="utf-8"
            )

            observation, _ = wrapped_env.reset()
            previous_action = torch.zeros(args.num_envs, 12, device=device)
            previous_residual = torch.zeros_like(previous_action)
            last_step = 0
            latest_update = None
            terminal_snapshot: dict[str, torch.Tensor] = {}
            current_decision = None

            def capture_terminal_state(env_ids) -> None:
                if current_decision is None:
                    raise RuntimeError("pre-reset capture occurred without an active decision")
                ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
                captured_mask = torch.zeros(args.num_envs, 1, dtype=torch.bool, device=device)
                captured_mask[ids] = True
                terminal_observation = wrapped_env.unwrapped.observation_manager.compute()
                policy_observation = terminal_observation.get("policy")
                if not isinstance(policy_observation, torch.Tensor):
                    raise RuntimeError("terminal policy observation must be a tensor")
                terminal_signals = adapter.reward_signals()
                terminal_snapshot.update(
                    mask=captured_mask,
                    policy_observation=policy_observation.detach().clone(),
                    frame=adapter.frame(current_decision.action).detach().clone(),
                    reward=compute_online_reward(
                        terminal_signals,
                        current_decision.residual,
                        previous_residual,
                        reward_weights,
                    ).detach().clone(),
                    velocity_error=torch.linalg.norm(
                        terminal_signals.base_lin_vel_xy
                        - terminal_signals.command_lin_vel_xy,
                        dim=-1,
                    ).detach().clone(),
                    tilt=torch.linalg.norm(
                        terminal_signals.projected_gravity_xy, dim=-1
                    ).detach().clone(),
                )

            wrapped_env.unwrapped.add_pre_reset_callback(capture_terminal_state)

            def save_checkpoint(step: int) -> None:
                checkpoint = output_dir / f"residual_{step:08d}.pt"
                online_agent.updater.save(checkpoint)
                online_agent.updater.save(output_dir / "latest.pt")

            for step in range(1, args.max_steps + 1):
                last_step = step
                terminal_snapshot.clear()
                frame = adapter.frame(previous_action)
                terrain_valid = adapter.terrain_valid
                timeout = torch.zeros(args.num_envs, 1, dtype=torch.bool, device=device)
                with torch.inference_mode():
                    base_action = clip_lite3_policy_action(policy(observation))
                decision = online_agent.begin_step(
                    frame,
                    base_action,
                    adapter.safety_state(timeout),
                    deterministic=False,
                    terrain_valid=terrain_valid,
                )
                current_decision = decision
                with torch.inference_mode():
                    next_observation, _, done, extras = wrapped_env.step(decision.action)
                done_column = done.bool().reshape(-1, 1)
                post_reset_frame = adapter.frame(decision.action)
                signals = adapter.reward_signals()
                post_reset_reward = compute_online_reward(
                    signals, decision.residual, previous_residual, reward_weights
                )
                velocity_error_rows = torch.linalg.norm(
                    signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1
                )
                tilt_rows = torch.linalg.norm(signals.projected_gravity_xy, dim=-1)
                bootstrap_observation = next_observation.clone()
                next_frame = post_reset_frame
                reward = post_reset_reward
                if bool(done_column.any().item()):
                    captured_mask = terminal_snapshot.get("mask")
                    if not isinstance(captured_mask, torch.Tensor) or not torch.equal(
                        captured_mask, done_column
                    ):
                        raise RuntimeError("pre-reset capture mask does not match ended environments")
                    next_frame = torch.where(
                        done_column, terminal_snapshot["frame"], post_reset_frame
                    )
                    reward = torch.where(
                        done_column, terminal_snapshot["reward"], post_reset_reward
                    )
                    policy_mask = done_column.expand_as(terminal_snapshot["policy_observation"])
                    bootstrap_observation["policy"] = torch.where(
                        policy_mask,
                        terminal_snapshot["policy_observation"],
                        next_observation["policy"],
                    )
                    row_mask = done_column.squeeze(-1)
                    velocity_error_rows = torch.where(
                        row_mask,
                        terminal_snapshot["velocity_error"],
                        velocity_error_rows,
                    )
                    tilt_rows = torch.where(
                        row_mask, terminal_snapshot["tilt"], tilt_rows
                    )
                with torch.inference_mode():
                    next_base_action = clip_lite3_policy_action(policy(bootstrap_observation))
                terminated, truncated, bootstrap_mask = split_rsl_rl_endings(
                    done, extras, bootstrap_timeouts=True
                )
                curriculum_stage_id = torch.zeros_like(terminated, dtype=torch.int64)
                online_agent.finish_step(
                    next_frame=next_frame,
                    next_base_action=next_base_action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    bootstrap_mask=bootstrap_mask,
                    discount=torch.full_like(reward, online_agent.updater.config.gamma),
                    safety_mask=(~decision.paused.bool()).reshape(-1, 1),
                    curriculum_stage_id=curriculum_stage_id,
                    episode_id=torch.zeros_like(curriculum_stage_id),
                )

                paused = online_agent.all_environments_paused(decision.paused)
                if online_agent.should_update(len(online_agent.replay), args.min_replay_size, paused):
                    critic_batch = online_agent.sample_critic_batch()
                    actor_batch = online_agent.replay.sample(
                        online_agent.config.batch_size, online_agent.device
                    )
                    flat_anchor_batch = (
                        None
                        if online_agent.flat_replay is None
                        else online_agent.flat_replay.sample(
                            online_agent.config.batch_size, online_agent.device
                        )
                    )
                    stats = online_agent.updater.update(
                        critic_batch=critic_batch,
                        actor_batch=actor_batch,
                        flat_anchor_batch=flat_anchor_batch,
                        residual_gain=online_agent.residual_gain,
                    )
                    latest_update = stats
                    writer.add_scalar("loss/critic", stats.critic_loss, step)
                    writer.add_scalar("loss/actor", stats.actor_loss, step)
                    writer.add_scalar("loss/alpha", stats.alpha_loss, step)
                    writer.add_scalar("train/alpha", stats.alpha, step)
                    writer.add_scalar("train/version", stats.version, step)
                    if stats.accepted:
                        online_agent.publisher.publish(online_agent.updater.actor, stats.version)

                velocity_error = velocity_error_rows.mean()
                tilt = tilt_rows.mean()
                writer.add_scalar("source/velocity_error", float(velocity_error), step)
                writer.add_scalar("source/tilt", float(tilt), step)
                writer.add_scalar("runtime/replay_size", len(online_agent.replay), step)
                writer.add_scalar("runtime/terminated_fraction", float(terminated.float().mean()), step)
                writer.add_scalar("runtime/truncated_fraction", float(truncated.float().mean()), step)
                writer.add_scalar("runtime/bootstrap_fraction", float(bootstrap_mask.float().mean()), step)

                if step == 1 or step % 250 == 0:
                    update_text = (
                        "none"
                        if latest_update is None
                        else (
                            f"version={latest_update.version} "
                            f"critic_loss={latest_update.critic_loss:.5f} "
                            f"actor_loss={latest_update.actor_loss:.5f}"
                        )
                    )
                    print(
                        f"[residual-pretrain] step={step}/{args.max_steps} "
                        f"velocity_error={float(velocity_error):.5f} "
                        f"tilt={float(tilt):.5f} replay={len(online_agent.replay)} "
                        f"timeouts={int(truncated.sum())} "
                        f"bootstrap={int(bootstrap_mask.sum())} "
                        f"{update_text}",
                        flush=True,
                    )

                if step % args.save_interval == 0:
                    with torch.inference_mode():
                        deterministic_residual, _, _ = online_agent.updater.actor.sample(
                            online_agent.history.snapshot(), next_frame, next_base_action, deterministic=True
                        )
                    residual_norm = deterministic_residual.norm(dim=-1).mean()
                    writer.add_scalar("eval/deterministic_residual_norm_mean", float(residual_norm), step)
                    writer.add_scalar(
                        "eval/applied_deterministic_residual_norm_mean",
                        float(residual_norm * args.residual_gain),
                        step,
                    )
                    save_checkpoint(step)

                done_mask = terminated | truncated
                previous_residual = torch.where(done_mask, torch.zeros_like(decision.residual), decision.residual)
                previous_action = torch.where(done_mask, torch.zeros_like(decision.action), decision.action)
                observation = next_observation

            save_checkpoint(last_step)
            writer.flush()
            return 0
        finally:
            writer.close()
            wrapped_env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
