"""Qualify a frozen Lite3 PPO checkpoint on source-only terrains."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.app import AppLauncher

from scripts.reinforcement_learning.online_adaptation.cli import build_qualify_parser


def main() -> int:
    args = build_qualify_parser().parse_args()
    if "HeldoutBoxes" in args.task:
        raise ValueError("base qualification must use the source-only task")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import torch

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter

        records: list[dict[str, float | int]] = []
        for seed in args.seeds:
            env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
            env_cfg.seed = seed
            env_cfg.terminations.time_out = None
            agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
            agent_cfg.seed = seed
            env = gym.make(args.task, cfg=env_cfg)
            wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            try:
                runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
                runner.load(args.base_checkpoint)
                policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
                adapter = Lite3IsaacLabAdapter(wrapped_env.unwrapped)
                observation, _ = wrapped_env.reset()
                velocity_error_sum = 0.0
                sample_count = 0
                falls = 0
                for _ in range(args.steps_per_seed):
                    with torch.inference_mode():
                        action = policy(observation)
                        observation, _, done, _ = wrapped_env.step(action)
                    signals = adapter.reward_signals()
                    velocity_error_sum += float(
                        torch.linalg.norm(signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1).mean()
                    )
                    sample_count += 1
                    if done.any().item():
                        falls += int(done.sum().item())
                        break
                records.append(
                    {
                        "seed": seed,
                        "falls": falls,
                        "steps": sample_count,
                        "mean_velocity_error": velocity_error_sum / max(sample_count, 1),
                    }
                )
            finally:
                wrapped_env.close()

        aggregate_error = sum(float(record["mean_velocity_error"]) for record in records) / len(records)
        aggregate_falls = sum(int(record["falls"]) for record in records)
        passed = aggregate_falls == 0 and aggregate_error <= args.max_mean_velocity_error
        output = {
            "task": args.task,
            "base_checkpoint": args.base_checkpoint,
            "records": records,
            "aggregate": {
                "falls": aggregate_falls,
                "mean_velocity_error": aggregate_error,
                "max_mean_velocity_error": args.max_mean_velocity_error,
                "passed": passed,
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        return 0 if passed else 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
