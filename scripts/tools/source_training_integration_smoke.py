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
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output before launch, so a killed process cannot look like an
    # absent run and be retried accidentally under the same artifact name.
    stream = args.output.open("x", encoding="utf-8")
    app, env = None, None
    started = time.monotonic()
    report = {"status": "starting", "simulation_control_steps": 0, "ppo_updates": 0,
              "training_ready": False, "g3_g4_qualified": False,
              "diagnostic_only": True, "injected_timeout": True,
              "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    stream.write(json.dumps(report) + "\n")
    stream.flush()
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
        rows = []
        for index in range(24):
            if not app.is_running():
                raise RuntimeError("simulator stopped before diagnostic limit")
            if index == 23:
                # Diagnostic injection, never a natural 2000-step episode.
                env.unwrapped.episode_length_buf[0] = 1999
            with torch.inference_mode():
                raw = actor(obs) + torch.randn((2, 12), device=args.device, generator=rng)
                obs, rewards, dones, extras = env.step(raw)
            report["simulation_control_steps"] = index + 1
            terminal = extras["source_terminal"]
            rows.append({"step": index + 1, "done": dones.cpu().tolist(),
                         "raw_action": raw.cpu().tolist(),
                         "executed_action": env.last_trace.executed_action.cpu().tolist(),
                         "projected": env.last_trace.decision.projected.cpu().tolist(),
                         "terminal_ids": terminal["ids"].cpu().tolist() if terminal else [],
                         "terminal_steps": terminal["steps"].cpu().tolist() if terminal else [],
                         "active_terms": {name: value.cpu().tolist()
                                          for name, value in terminal["terms"].items()} if terminal else {}})
            report["steps"] = rows
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
        print("[source-training-smoke] wiring probe passed; PPO updates=0", flush=True)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            report.update(status="failed", environment_cleanup_error=traceback.format_exc())
        finally:
            try:
                if app is not None:
                    app.close()
            except Exception:
                report.update(status="failed", simulator_cleanup_error=traceback.format_exc())
            report["elapsed_seconds"] = time.monotonic() - started
            stream.seek(0)
            stream.truncate()
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
