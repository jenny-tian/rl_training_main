"""Qualify a frozen Lite3 PPO on the deterministic deployment-matched flat task."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.app import AppLauncher

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.lite3_stability_contract import LITE3_MAX_PROJECTED_GRAVITY_XY
from rl_training.online_adaptation.rsl_rl_compat import prepare_rsl_rl_agent_cfg, split_rsl_rl_endings
from rl_training.online_adaptation.terrain_guards import assert_flat_online_deployment_config
from rl_training.online_adaptation.deployment_qualification import build_deployment_qualification_report
from scripts.reinforcement_learning.online_adaptation.cli import (
    FLAT_ONLINE_TASK,
    build_deployment_qualify_parser,
)


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_deployment_qualify_parser().parse_args(argv)
    if args.task != FLAT_ONLINE_TASK:
        raise ValueError(f"deployment qualification requires task {FLAT_ONLINE_TASK!r}")
    if args.max_projected_gravity_xy != LITE3_MAX_PROJECTED_GRAVITY_XY:
        raise ValueError("deployment qualification does not allow changing the tilt threshold")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.isaaclab_adapter import Lite3IsaacLabAdapter
        from rl_training.online_adaptation.deployment_startup import make_lite3_deployment_env
        from rl_training.online_adaptation.rsl_rl_compat import load_frozen_rsl_rl_checkpoint
        from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.seed = args.seed
        assert_flat_online_deployment_config(env_cfg)
        agent_cfg = prepare_rsl_rl_agent_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"))
        wrapped_env = make_lite3_deployment_env(args.task, env_cfg, agent_cfg, gym_module=gym)
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            load_frozen_rsl_rl_checkpoint(runner, args.base_checkpoint)
            policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            adapter = Lite3IsaacLabAdapter(wrapped_env.unwrapped)
            limits = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4, device=wrapped_env.unwrapped.device))
            safety = SafetyFilter(limits)
            observation, reset_info = wrapped_env.reset()
            generation = wrapped_env.consume_startup_generation()
            if generation is None or reset_info.get("startup_generation") != generation:
                raise RuntimeError("deployment startup handoff generation was not acknowledged")
            attempted = completed = timeouts = falls = premature = paused = projected = nonfinite = 0
            velocity_sum = 0.0
            velocity_samples = 0
            max_tilt = 0.0
            min_height = float("inf")
            while attempted < args.expected_steps and completed < args.expected_episodes:
                attempted += 1
                with torch.inference_mode():
                    raw_action = policy(observation)
                finite = torch.isfinite(raw_action).all(dim=-1, keepdim=True)
                nonfinite += int((~finite).sum().item())
                action = clip_lite3_policy_action(raw_action)
                timeout = torch.zeros(action.shape[0], 1, dtype=torch.bool, device=action.device)
                state = adapter.safety_state(timeout)
                max_tilt = max(max_tilt, float(state.projected_gravity_xy.norm(dim=-1).amax().item()))
                min_height = min(min_height, float(state.base_height.amin().item()))
                decision = safety.project(action, torch.zeros_like(action), state, force_zero_residual=True)
                paused += int(decision.paused.sum().item())
                projected += int(decision.projected.sum().item())
                safety_mask = finite & ~decision.paused.bool() & ~decision.projected.bool()
                if not bool(safety_mask.all().item()):
                    break
                with torch.inference_mode():
                    observation, _, done, extras = wrapped_env.step(decision.action)
                terminated, truncated, _ = split_rsl_rl_endings(done, extras)
                signals = adapter.reward_signals()
                velocity_sum += float(torch.linalg.norm(signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1).mean().item())
                velocity_samples += 1
                if bool(terminated.any().item()):
                    falls += int(terminated.sum().item())
                    premature += int(terminated.sum().item())
                if bool(truncated.any().item()):
                    timeouts += int(truncated.sum().item())
                if bool((terminated | truncated).any().item()):
                    completed += int((terminated | truncated).sum().item())
                    safety.reset_temporal_state()
                    generation = wrapped_env.consume_startup_generation()
                    if generation is None:
                        raise RuntimeError("deployment startup reset did not produce a generation")
            record = {
                "task": args.task, "seed": args.seed, "attempted_rows": attempted,
                "completed_episodes": completed, "timeouts": timeouts, "falls": falls,
                "premature_terminations": premature, "paused_rows": paused,
                "projected_rows": projected, "nonfinite_rows": nonfinite,
                "max_projected_gravity_xy": max_tilt, "min_base_height": min_height,
                "mean_velocity_error": velocity_sum / max(velocity_samples, 1),
            }
            report = build_deployment_qualification_report(
                task=args.task, base_checkpoint=args.base_checkpoint, record=record,
                expected_seed=args.seed, expected_steps=args.expected_steps,
                expected_episodes=args.expected_episodes,
                max_mean_velocity_error=args.max_mean_velocity_error,
                max_projected_gravity_xy=args.max_projected_gravity_xy,
            )
            _write_report(Path(args.output), report)
            return 0 if report["passed"] else 1
        finally:
            wrapped_env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
