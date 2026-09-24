"""Run a minimal Isaac Lab reset/step smoke check for a Lite3 task."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="FlatOnline-Deeprobotics-Lite3-v0")
    parser.add_argument("--seed", type=int, default=101)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import rl_training.tasks  # noqa: F401

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        try:
            observation, reset_info = env.reset()
            unwrapped = env.unwrapped
            action_dim = int(unwrapped.action_manager.total_action_dim)
            action = torch.zeros(
                (int(unwrapped.num_envs), action_dim),
                device=unwrapped.device,
                dtype=torch.float32,
            )
            next_observation, reward, terminated, truncated, step_info = env.step(action)
            print(
                json.dumps(
                    {
                        "task": args.task,
                        "seed": args.seed,
                        "device": str(unwrapped.device),
                        "num_envs": int(unwrapped.num_envs),
                        "action_dim": action_dim,
                        "observation_type": type(observation).__name__,
                        "next_observation_type": type(next_observation).__name__,
                        "reward": reward.detach().cpu().tolist(),
                        "terminated": terminated.detach().cpu().tolist(),
                        "truncated": truncated.detach().cpu().tolist(),
                        "reset_info_keys": sorted(reset_info),
                        "step_info_keys": sorted(step_info),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        finally:
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
