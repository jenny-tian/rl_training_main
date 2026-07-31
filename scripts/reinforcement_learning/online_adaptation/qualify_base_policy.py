"""Qualify a frozen Lite3 PPO checkpoint on source-only terrains."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from scripts.reinforcement_learning.online_adaptation.cli import build_qualify_parser
from rl_training.online_adaptation.terrain_guards import assert_plane_only_terrain, is_strict_flat_baseline_task


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _run_multi_seed(args) -> int:
    from rl_training.online_adaptation.qualification import build_qualification_report

    if not is_strict_flat_baseline_task(args.task):
        raise ValueError("base qualification requires task 'Flat-Deeprobotics-Lite3-v0'")

    output_path = Path(args.output)
    records: list[dict[str, float | int]] = []
    with tempfile.TemporaryDirectory(prefix="lite3-base-qualification-") as temporary_dir:
        for seed in args.seeds:
            worker_output = Path(temporary_dir) / f"seed_{seed}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--base-checkpoint",
                args.base_checkpoint,
                "--task",
                args.task,
                "--seeds",
                str(seed),
                "--steps-per-seed",
                str(args.steps_per_seed),
                "--max-mean-velocity-error",
                str(args.max_mean_velocity_error),
                "--output",
                str(worker_output),
                "--device",
                args.device,
            ]
            if args.headless:
                command.append("--headless")
            completed = subprocess.run(command, check=False)
            if not worker_output.is_file():
                raise RuntimeError(
                    f"qualification worker for seed {seed} produced no report "
                    f"(exit code {completed.returncode})"
                )
            worker_report = json.loads(worker_output.read_text(encoding="utf-8"))
            worker_records = worker_report.get("records", [])
            if len(worker_records) != 1 or int(worker_records[0]["seed"]) != seed:
                raise RuntimeError(f"qualification worker returned an invalid record for seed {seed}")
            records.append(worker_records[0])

    report = build_qualification_report(
        args.task,
        args.base_checkpoint,
        records,
        args.max_mean_velocity_error,
    )
    _write_report(output_path, report)
    return 0 if report["aggregate"]["passed"] else 1


def _run_single_seed(args) -> int:
    from rl_training.online_adaptation.qualification import build_qualification_report

    if not is_strict_flat_baseline_task(args.task):
        raise ValueError("base qualification requires task 'Flat-Deeprobotics-Lite3-v0'")
    if len(args.seeds) != 1:
        raise ValueError("single-seed worker requires exactly one seed")

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
        from rl_training.online_adaptation.rsl_rl_compat import prepare_rsl_rl_agent_cfg
        from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits

        seed = args.seeds[0]
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        assert_plane_only_terrain(env_cfg.scene.terrain)
        env_cfg.seed = seed
        env_cfg.terminations.time_out = None
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = seed
        agent_cfg = prepare_rsl_rl_agent_cfg(agent_cfg)
        env = gym.make(args.task, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(args.base_checkpoint)
            policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
            adapter = Lite3IsaacLabAdapter(wrapped_env.unwrapped)
            action_limits = torch.tensor([0.24, 0.20, 0.20] * 4, device=wrapped_env.unwrapped.device)
            safety = SafetyFilter(SafetyLimits(residual_limits=action_limits))
            observation, _ = wrapped_env.reset()
            velocity_error_sum = 0.0
            sample_count = 0
            falls = 0
            paused_rows = 0
            projected_rows = 0
            nonfinite_rows = 0
            for _ in range(args.steps_per_seed):
                with torch.inference_mode():
                    raw_action = policy(observation)
                finite_rows = torch.isfinite(raw_action).all(dim=-1, keepdim=True)
                nonfinite_rows += int((~finite_rows).sum().item())
                if not bool(finite_rows.all().item()):
                    break
                action = clip_lite3_policy_action(raw_action)
                timeout = torch.zeros(
                    action.shape[0], 1, dtype=torch.bool, device=action.device
                )
                decision = safety.project(
                    action,
                    torch.zeros_like(action),
                    adapter.safety_state(timeout),
                    force_zero_residual=True,
                )
                paused_rows += int(decision.paused.sum().item())
                projected_rows += int(decision.projected.sum().item())
                with torch.inference_mode():
                    observation, _, done, _ = wrapped_env.step(decision.action)
                signals = adapter.reward_signals()
                velocity_error_sum += float(
                    torch.linalg.norm(signals.base_lin_vel_xy - signals.command_lin_vel_xy, dim=-1).mean()
                )
                sample_count += 1
                if done.any().item():
                    falls += int(done.sum().item())
                    break
            record = {
                "seed": seed,
                "falls": falls,
                "paused_rows": paused_rows,
                "projected_rows": projected_rows,
                "nonfinite_rows": nonfinite_rows,
                "steps": sample_count,
                "mean_velocity_error": velocity_error_sum / max(sample_count, 1),
            }
            report = build_qualification_report(
                args.task,
                args.base_checkpoint,
                [record],
                args.max_mean_velocity_error,
            )
            _write_report(Path(args.output), report)
            return 0 if report["aggregate"]["passed"] else 1
        finally:
            wrapped_env.close()
    finally:
        simulation_app.close()


def main() -> int:
    args = build_qualify_parser().parse_args()
    if not is_strict_flat_baseline_task(args.task):
        raise ValueError("base qualification requires task 'Flat-Deeprobotics-Lite3-v0'")
    if len(args.seeds) > 1:
        return _run_multi_seed(args)
    return _run_single_seed(args)


if __name__ == "__main__":
    raise SystemExit(main())
