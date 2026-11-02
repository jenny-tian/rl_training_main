"""Two-environment Source training wiring probe, without PPO construction.

This is a short diagnostic, not G3/G4 qualification or a terrain success test.
It uses the first registered training seed, shared-bank layout and frozen actor
mean plus independent unit Gaussian noise. The last step injects an episode
timeout in environment zero to check pre-reset capture and selective reset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))

from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter, finalize_and_close


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app, env = None, None
    started = time.monotonic()
    report = {"status": "starting", "simulation_control_steps": 0, "ppo_updates": 0,
              "training_ready": False, "g3_g4_qualified": False,
              "diagnostic_only": True, "injected_timeout": True,
              "attempt_id": args.attempt_id,
              "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    writer = SmokeArtifactWriter(args.output, report)
    try:
        print("[source-training-smoke] launching simulator", flush=True)
        app = AppLauncher(args).app
        import gymnasium as gym
        import torch
        import rl_training.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.source_scan.frozen_actor import load_frozen_source_actor
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.source_scan.training_lifecycle import SourceRunSeeds
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import (
            DeeproboticsLite3SourceTrainingEnvCfg,
        )

        seeds = SourceRunSeeds(1)
        cfg = DeeproboticsLite3SourceTrainingEnvCfg(
            source_training_run=1, source_parallel_sampler_enabled=True,
            source_parallel_sampler_num_envs=2, source_shared_bank_rows=1)
        cfg.sim.device = args.device
        report.update(ppo_seed=seeds.ppo, geometry_seed=seeds.geometry, sampler_seed=seeds.sampler,
                      reset_state_id=seeds.reset_state_id, reset_randomized=False,
                      num_envs=2, shared_bank_rows=1, stochastic_std=1.0,
                      noise_rng_seed=132001, rsl_rl=importlib.metadata.version("rsl-rl-lib"))
        actor = load_frozen_source_actor(args.checkpoint, device=args.device)
        print("[source-training-smoke] constructing two environments", flush=True)
        base_env = gym.make("SourceTerrain-Deeprobotics-Lite3-v0", cfg=cfg)
        env = base_env
        wrapped = RslRlVecEnvWrapper(base_env, clip_actions=None)
        env = SourceTrainingVecEnv(wrapped)
        obs = env.get_observations()
        rng = torch.Generator(device=args.device).manual_seed(132001)
        report.update(status="running", critic_width=obs["critic"].shape[1])
        writer.persist(report)
        for index in range(24):
            if not app.is_running():
                raise RuntimeError("simulator stopped before diagnostic limit")
            if index == 23:
                # Diagnostic injection, never a natural 2000-step episode.
                env.unwrapped.episode_length_buf[0] = 1999
            columns_before = env.unwrapped.scene.terrain.terrain_types.clone()
            with torch.inference_mode():
                raw = actor(obs) + torch.randn((2, 12), device=args.device, generator=rng)
                obs, rewards, dones, extras = env.step(raw)
            report["simulation_control_steps"] = index + 1
            terminal = extras["source_terminal"]
            terminal_row = None
            if terminal is not None:
                terminal_row = {name: terminal[name].cpu().tolist() for name in (
                    "ids", "steps", "hold", "completed", "terminated", "truncated")}
                terminal_row["observations"] = {
                    name: value.cpu().tolist() for name, value in terminal["observations"].items()}
                terminal_row["terms"] = {name: value.cpu().tolist() for name, value in terminal["terms"].items()}
            writer.append_step({"step": index + 1, "done": dones.cpu().tolist(),
                         "raw_action": raw.cpu().tolist(),
                         "base_action": env.last_trace.base_action.cpu().tolist(),
                         "requested_action": env.last_trace.requested_action.cpu().tolist(),
                         "executed_action": env.last_trace.executed_action.cpu().tolist(),
                         "projected": env.last_trace.decision.projected.cpu().tolist(),
                         "paused": env.last_trace.decision.paused.cpu().tolist(),
                         "time_outs": extras["time_outs"].cpu().tolist(),
                         "episode_counters": env.unwrapped.episode_length_buf.cpu().tolist(),
                         "terrain_columns_before": columns_before.cpu().tolist(),
                         "terrain_columns_after": env.unwrapped.scene.terrain.terrain_types.cpu().tolist(),
                         "terminal": terminal_row})
            report.update(exposure=env.exposure.report(), total_transitions=2 * (index + 1))
            writer.persist(report)
            if index < 23 and bool(dones.any()):
                raise RuntimeError("unexpected early episode ending; stop without retry")
            if index == 23:
                if dones.tolist() != [1, 0] or terminal is None:
                    raise RuntimeError("injected timeout did not reset exactly environment zero")
                if terminal["terminated"].any() or extras["time_outs"].tolist() != [True, False]:
                    raise RuntimeError("injected timeout was confused with a task failure")
                if env.unwrapped.episode_length_buf.tolist() != [0, 24]:
                    raise RuntimeError("asynchronous timeout reset changed unaffected episode counter")
        report.update(status="passed", exposure=env.exposure.report(),
                      total_transitions=48, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
        print("[source-training-smoke] simulation assertions completed; parent acceptance pending; PPO updates=0", flush=True)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds_before_shutdown"] = time.monotonic() - started
        finalize_and_close(writer, report, environment_close=env.close if env is not None else None,
                           application_close=app.close if app is not None else None)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
