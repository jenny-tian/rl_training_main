"""Pretrain the Lite3 residual SAC policy without held-out terrain exposure."""

from __future__ import annotations

import json
from collections import deque
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
        from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.rsl_rl_compat import freeze_rsl_rl_models, prepare_rsl_rl_agent_cfg
        from rl_training.online_adaptation.sac import SACConfig, StableStateManager
        from rl_training.online_adaptation.safety import SafetyLimits
        from rl_training.online_adaptation.terrain_guards import assert_heldout_terrains_disabled

        torch.manual_seed(args.seed)
        output_dir = Path(args.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        assert_heldout_terrains_disabled(env_cfg.scene.terrain.terrain_generator)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg = prepare_rsl_rl_agent_cfg(agent_cfg)
        env = gym.make(args.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        writer = SummaryWriter(log_dir=str(output_dir))
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(args.base_checkpoint)
            policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            freeze_rsl_rl_models(runner)

            device = wrapped_env.unwrapped.device
            action_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=device)
            actor = ResidualActor(49, 12, 64, action_limits).to(device)
            critic = TwinCritic(49, 12, 64).to(device)
            runtime_config = AgentConfig(
                replay_capacity=args.replay_capacity,
                min_replay_size=args.min_replay_size,
            )
            online_agent = OnlineAdaptationAgent(
                num_envs=args.num_envs,
                actor=actor,
                critic=critic,
                safety_limits=SafetyLimits(residual_limits=action_limits),
                config=runtime_config,
                sac_config=SACConfig(),
                device=device,
            )
            if args.resume_checkpoint is not None:
                online_agent.updater.load(args.resume_checkpoint, map_location=device)
                online_agent.publisher.publish(online_agent.updater.actor, online_agent.updater.version)
            adapter = Lite3IsaacLabAdapter(wrapped_env.unwrapped)
            stable_manager = StableStateManager()
            reward_weights = RewardWeights(residual_weight=0.25, residual_rate_weight=0.10)
            source_scores: deque[float] = deque(maxlen=250)

            config_record = {
                "arguments": vars(args),
                "agent": asdict(runtime_config),
                "sac": asdict(online_agent.updater.config),
                "reward": asdict(reward_weights),
                "terrain_leakage_assertion": "passed",
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

            def save_checkpoint(step: int) -> None:
                checkpoint = output_dir / f"residual_{step:08d}.pt"
                online_agent.updater.save(checkpoint)
                online_agent.updater.save(output_dir / "latest.pt")

            for step in range(1, args.max_steps + 1):
                last_step = step
                frame = adapter.frame(previous_action)
                timeout = torch.zeros(args.num_envs, 1, dtype=torch.bool, device=device)
                with torch.inference_mode():
                    base_action = policy(observation)
                decision = online_agent.begin_step(
                    frame,
                    base_action,
                    adapter.safety_state(timeout),
                    deterministic=False,
                )
                with torch.inference_mode():
                    next_observation, _, done, _ = wrapped_env.step(decision.action)
                    next_base_action = policy(next_observation)
                next_frame = adapter.frame(decision.action)
                signals = adapter.reward_signals()
                reward = compute_online_reward(signals, decision.residual, previous_residual, reward_weights)
                done_column = done.reshape(-1, 1)
                online_agent.finish_step(next_frame, next_base_action, reward, done_column)

                paused = online_agent.all_environments_paused(decision.paused)
                if online_agent.should_update(len(online_agent.replay), args.min_replay_size, paused):
                    stats = online_agent.updater.update(online_agent.sample_update_batch())
                    writer.add_scalar("loss/critic", stats.critic_loss, step)
                    writer.add_scalar("loss/actor", stats.actor_loss, step)
                    writer.add_scalar("loss/alpha", stats.alpha_loss, step)
                    writer.add_scalar("train/alpha", stats.alpha, step)
                    writer.add_scalar("train/version", stats.version, step)
                    if stats.accepted:
                        online_agent.publisher.publish(online_agent.updater.actor, stats.version)

                velocity_error = torch.linalg.norm(
                    signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1
                ).mean()
                tilt = torch.linalg.norm(signals.projected_gravity_xy, dim=-1).mean()
                score = -float(velocity_error + tilt)
                source_scores.append(score)
                writer.add_scalar("source/velocity_error", float(velocity_error), step)
                writer.add_scalar("source/tilt", float(tilt), step)
                writer.add_scalar("runtime/replay_size", len(online_agent.replay), step)

                if step % 250 == 0:
                    rolled_back = stable_manager.observe(
                        sum(source_scores) / max(len(source_scores), 1), online_agent.updater
                    )
                    if rolled_back:
                        online_agent.publisher.publish(online_agent.updater.actor, online_agent.updater.version)
                        save_checkpoint(step)
                        writer.add_scalar("train/rollback", 1, step)

                if step % args.save_interval == 0:
                    with torch.inference_mode():
                        deterministic_residual, _, _ = online_agent.updater.actor.sample(
                            online_agent.history.snapshot(), next_frame, next_base_action, deterministic=True
                        )
                    writer.add_scalar("eval/deterministic_residual_norm", float(deterministic_residual.norm()), step)
                    save_checkpoint(step)

                done_mask = done_column.bool()
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
