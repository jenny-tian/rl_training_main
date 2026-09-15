"""Experimental low-rank PPO worker for Lite3 online terrain adaptation.

The script is deliberately separate from ``play_online_adaptation.py``.  That
worker remains the SAC reference path; this entrypoint makes a paired PPO run
possible without changing its replay or safety semantics.  It uses the same
RSL-RL checkpoint and task, injects zero-output LoRA adapters into the actor,
and updates only after complete on-policy blocks.
"""

from __future__ import annotations

import copy
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run low-rank online PPO adaptation from a frozen Lite3 PPO checkpoint.",
        allow_abbrev=False,
    )
    # AppLauncher inspects the parser while adding its own flags, so a
    # required application argument would be rejected before parse_args gets
    # the actual command line.  Validate it in main after parsing instead.
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--task", default="FlatOnline-Deeprobotics-Lite3-v0")
    parser.add_argument(
        "--terrain-stage",
        choices=("step_2cm", "step_4cm", "step_6cm", "rough_2cm", "rough_4cm"),
        default="rough_4cm",
    )
    parser.add_argument("--terrain-seed", type=int, default=101)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--forward-speed", type=float, default=0.3)
    parser.add_argument("--max-steps", type=_positive_int, default=2_000)
    parser.add_argument("--rollout-steps", type=_positive_int, default=256)
    parser.add_argument("--output", default="logs/online_adaptation/lora_ppo")
    parser.add_argument("--rank", type=_positive_int, default=2)
    parser.add_argument("--lora-alpha", type=_positive_float, default=None)
    parser.add_argument("--action-std", type=_positive_float, default=0.1)
    parser.add_argument("--residual-rate-limit", type=_positive_float, default=0.03)
    parser.add_argument("--actor-lr", type=_positive_float, default=1e-4)
    parser.add_argument("--critic-lr", type=_positive_float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=_positive_int, default=2)
    parser.add_argument("--minibatch-size", type=_positive_int, default=64)
    parser.add_argument("--clip-eps", type=_positive_float, default=0.1)
    parser.add_argument("--target-kl", type=_positive_float, default=0.01)
    parser.add_argument("--gamma", type=_positive_float, default=0.99)
    parser.add_argument("--gae-lambda", type=_positive_float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    parser.add_argument("--max-parameter-abs", type=_positive_float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.base_checkpoint:
        raise SystemExit("--base-checkpoint is required")
    simulation_app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch
        from torch import nn
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        from rl_training.online_adaptation.lora import inject_lora, lora_parameter_count, mark_only_lora_trainable
        from rl_training.online_adaptation.online_session import OnlinePPOSession
        from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
        from rl_training.online_adaptation.ppo_rollout import PPORollout
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.reward import RewardWeights, compute_online_reward
        from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
        from rl_training.online_adaptation.rsl_rl_compat import (
            RslMLPPolicyAdapter,
            freeze_rsl_rl_models,
            load_frozen_rsl_rl_checkpoint,
            prepare_rsl_rl_agent_cfg,
            split_rsl_rl_endings,
        )
        from rl_training.online_adaptation.terrain_protocol import (
            apply_geometry_to_env_cfg,
            development_geometry,
        )

        torch.manual_seed(args.seed)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 1
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.forward_speed, args.forward_speed)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.terminations.time_out = None
        geometry = development_geometry(args.terrain_stage, args.terrain_seed)
        apply_geometry_to_env_cfg(env_cfg, geometry)
        agent_cfg = prepare_rsl_rl_agent_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"))
        env = gym.make(args.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            checkpoint_load = load_frozen_rsl_rl_checkpoint(runner, args.base_checkpoint)
            source_model = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            # Freeze the source model first, then inject adapters.  The RSL-RL
            # runner is not used for learning after this point.
            freeze_rsl_rl_models(runner)
            adapted_model = copy.deepcopy(source_model)
            adapters = inject_lora(adapted_model, rank=args.rank, alpha=args.lora_alpha)
            distribution = getattr(adapted_model, "distribution", None)
            if hasattr(distribution, "log_std_param"):
                distribution.log_std_param.data.fill_(math.log(args.action_std))
            elif hasattr(distribution, "std_param"):
                distribution.std_param.data.fill_(args.action_std)
            else:
                raise RuntimeError("RSL actor must expose a configurable Gaussian action standard deviation")
            mark_only_lora_trainable(adapted_model)
            # The adapter owns the trainable LoRA parameters while the
            # publisher holds a frozen inference copy.  Keep both on the same
            # device so publication is a pure state-dict swap.
            policy = RslMLPPolicyAdapter(adapted_model).to(args.device)
            probe_obs, _ = wrapped_env.reset()
            if not isinstance(probe_obs, dict) and not hasattr(probe_obs, "get"):
                raise RuntimeError("RSL-RL online PPO requires grouped policy observations")
            policy_obs = probe_obs["policy"]
            if not isinstance(policy_obs, torch.Tensor) or policy_obs.ndim != 2:
                raise RuntimeError("policy observation must have shape [1, obs_dim]")
            obs_dim = int(policy_obs.shape[-1])
            critic = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)
            ).to(args.device)
            updater = OnlinePPOUpdater(
                policy,
                critic,
                PPOConfig(
                    clip_eps=args.clip_eps,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    max_grad_norm=args.max_grad_norm,
                    target_kl=args.target_kl,
                    max_parameter_abs=args.max_parameter_abs,
                ),
            )
            from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher

            publisher = DoubleBufferedActorPublisher(policy)
            session = OnlinePPOSession(updater, rollout_capacity=args.rollout_steps, publisher=publisher)
            reward_adapter = Lite3IsaacLabAdapter(wrapped_env.unwrapped, terrain_feature_source="proprio")
            if not hasattr(wrapped_env.unwrapped, "add_pre_reset_callback"):
                raise RuntimeError("online PPO task must support pre-reset terminal capture")
            reward_weights = RewardWeights(residual_weight=0.0, residual_rate_weight=0.0)
            safety_filter = SafetyFilter(SafetyLimits(
                residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=args.device),
                residual_rate_limit=args.residual_rate_limit,
            ))
            observation = probe_obs
            terminal_capture = {}
            current_residual = torch.zeros(1, 12, device=args.device)
            previous_residual = torch.zeros_like(current_residual)

            def capture_terminal_state(env_ids):
                ids = torch.as_tensor(env_ids, dtype=torch.long, device=args.device)
                terminal_observation = wrapped_env.unwrapped.observation_manager.compute()["policy"]
                terminal_capture["ids"] = ids
                terminal_capture["observation"] = terminal_observation.detach().clone()
                terminal_capture["reward"] = compute_online_reward(
                    reward_adapter.reward_signals(),
                    current_residual,
                    previous_residual,
                    reward_weights,
                ).detach().clone()

            wrapped_env.unwrapped.add_pre_reset_callback(capture_terminal_state)
            rows = []
            update_records = []
            updates_stream = (output / "updates.jsonl").open("w", encoding="utf-8")
            for step in range(args.max_steps):
                previous_residual = current_residual.detach().clone()
                terminal_capture.clear()
                obs_tensor = observation["policy"]
                control_policy, version = publisher.current()
                with torch.no_grad():
                    distribution = control_policy.get_distribution(obs_tensor)
                    requested_action = distribution.sample()
                    old_log_prob = distribution.log_prob(requested_action)
                    if old_log_prob.ndim > 1:
                        old_log_prob = old_log_prob.sum(dim=-1)
                    old_value = critic(obs_tensor)
                    base_action = clip_lite3_policy_action(source_model(observation))
                    requested_residual = requested_action - base_action
                    safety_decision = safety_filter.project(
                        base_action,
                        requested_residual,
                        reward_adapter.safety_state(torch.zeros(1, 1, dtype=torch.bool, device=args.device)),
                    )
                    executed_action = safety_decision.action
                    current_residual = safety_decision.residual.detach().clone()
                result = wrapped_env.step(executed_action)
                next_observation, _, done, extras = result
                terminated, truncated, _ = split_rsl_rl_endings(done, extras)
                next_obs_tensor = next_observation["policy"]
                environment_done = bool((terminated | truncated).any().item())
                value_observation = next_obs_tensor
                reward_tensor = compute_online_reward(
                    reward_adapter.reward_signals(),
                    current_residual,
                    previous_residual,
                    reward_weights,
                ).to(dtype=torch.float32)
                if environment_done:
                    if "observation" not in terminal_capture or "reward" not in terminal_capture:
                        raise RuntimeError("ended online PPO row has no pre-reset terminal capture")
                    value_observation = terminal_capture["observation"]
                    reward_tensor = terminal_capture["reward"].to(dtype=torch.float32)
                with torch.no_grad():
                    next_value_raw = critic(value_observation)
                    next_value = next_value_raw * (~terminated).to(dtype=next_value_raw.dtype)
                session.append(PPORollout(
                    observations=obs_tensor.detach(),
                    actions=requested_action.detach(),
                    old_log_probs=old_log_prob.detach(),
                    old_values=old_value.detach(),
                    rewards=reward_tensor.detach(),
                    terminated=terminated.detach(),
                    truncated=truncated.detach(),
                    next_values=next_value.detach(),
                    policy_version=version,
                )
                )
                rows.append({
                    "step": step + 1,
                    "reward": float(reward_tensor.mean()),
                    "done": bool(done.any()),
                    "policy_version": version,
                    "safety_projected": bool(safety_decision.projected.any()),
                    "safety_paused": bool(safety_decision.paused.any()),
                })
                observation = next_observation
                if environment_done:
                    safety_filter.reset_temporal_state()
                    current_residual.zero_()
                    previous_residual.zero_()
                if session.ready:
                    stats = session.update()
                    if stats is not None:
                        update_records.append(stats.__dict__)
                        updates_stream.write(json.dumps(stats.__dict__, sort_keys=True) + "\n")
                        updates_stream.flush()
                        print(
                            f"[LoRA-PPO] step={step + 1} version={updater.policy_version} "
                            f"accepted={stats.accepted} kl={stats.approx_kl:.6g}",
                            flush=True,
                        )
            if session.pending_transitions:
                stats = session.update(force=True)
                if stats is not None:
                    update_records.append(stats.__dict__)
                    updates_stream.write(json.dumps(stats.__dict__, sort_keys=True) + "\n")
                    updates_stream.flush()
            updates_stream.close()
            torch.save(
                {
                    "schema_version": 1,
                    "base_checkpoint": str(args.base_checkpoint),
                    "checkpoint_load": checkpoint_load,
                    "terrain_geometry": {
                        "step_height": geometry.step_height,
                        "rough_amplitude": geometry.rough_amplitude,
                        "seed": geometry.seed,
                    },
                    "lora_rank": args.rank,
                    "lora_parameter_count": lora_parameter_count(adapted_model),
                    "session": session.state_dict(),
                    "rows": rows,
                    "updates": update_records,
                },
                output / "latest.pt",
            )
            (output / "config.json").write_text(
                json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
            )
        finally:
            wrapped_env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
