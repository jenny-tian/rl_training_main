"""Paired B2 development evaluation: first episodes, 32 starts per column."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
TASK = "SourceBlock-B2-Deeprobotics-Lite3-v0"
SEED = 20_261_112


def evaluate(args, app):
    import gymnasium as gym
    import torch
    import rl_training.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_training.source_scan.frozen_actor import load_frozen_source_actor

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=640)
    cfg.seed = SEED
    cfg.observations.policy.enable_corruption = False
    # Equal coverage is evaluation-only; training keeps the B2 weighted mix.
    cfg.scene.terrain.source_curriculum_allocation = None
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    obs, _ = env.reset()
    actor = load_frozen_source_actor(args.checkpoint, device=base.device, expected_sha256=args.sha256)
    terrain, robot, state = base.scene.terrain, base.scene["robot"], base.source_block_state
    columns = terrain.terrain_types.clone()
    if torch.bincount(columns, minlength=20).tolist() != [32] * 20:
        raise ValueError("development evaluation requires 32 starts per column")
    initial = torch.cat((robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel,
                         base.command_manager.get_command("base_velocity"),
                         terrain.terrain_levels[:, None].float(), state.axes[:, None].float(), obs["policy"]), -1)
    hashes = [hashlib.sha256(initial[columns == c].cpu().numpy().tobytes()).hexdigest() for c in range(20)]
    alive = torch.ones(640, dtype=torch.bool, device=base.device)
    records = []
    with torch.inference_mode():
        for step in range(1, base.max_episode_length + 2):
            if not app.is_running():
                raise RuntimeError("simulator stopped during development evaluation")
            obs, reward, terminated, truncated, _ = env.step(actor(obs).clamp(-100, 100))
            if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                raise FloatingPointError("nonfinite development rollout")
            ended = alive & (terminated | truncated)
            for i in torch.nonzero(ended).flatten().tolist():
                terms = [name for name in base.termination_manager.active_terms
                         if bool(base.termination_manager.get_term(name)[i])]
                failed = bool(terminated[i])
                complete = not failed and "source_curriculum_completion" in terms
                contact = state.contact_diagnostics
                records.append(dict(env_id=i, column=int(columns[i]), step=step,
                    outcome="failure" if failed else "complete" if complete else "timeout",
                    terms=terms, terminal_contacts={n: float(f) for n, f in
                        zip(contact.names, contact.last_terminal[i].tolist()) if f > 5.},
                    episode_peak_contacts={n: float(f) for n, f in
                        zip(contact.names, contact.last_peak[i].tolist()) if f > 5.}))
            alive &= ~ended
            if step % 250 == 0:
                print(f"[block-development] step={step} first_episodes={len(records)}/640", flush=True)
            if not alive.any():
                break
    if alive.any() or len(records) != 640:
        raise RuntimeError("not every first episode ended")
    summary = []
    for c, name in enumerate(state.stats.column_names):
        rows = [r for r in records if r["column"] == c]
        summary.append(dict(column=c, name=name, episodes=len(rows),
                            **{o: sum(r["outcome"] == o for r in rows) for o in ("complete", "failure", "timeout")}))
    return dict(status="passed", task=TASK, seed=SEED, checkpoint=str(args.checkpoint),
                checkpoint_sha256=args.sha256, num_envs=640, policy_observation_corruption=False,
                initial_state_hashes_by_column=hashes, columns=summary,
                records=sorted(records, key=lambda r: r["env_id"]),
                contact_note="sensor history at terminal step; multiple body contacts may overlap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("checkpoint identity mismatch")
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), "--child", "--checkpoint", str(args.checkpoint),
                    "--sha256", args.sha256, "--output", str(args.output), "--gpu", str(args.gpu)],
                   gpu=args.gpu, output=args.output, timeout=1200)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0").app
        report = evaluate(args, app)
    except Exception:
        traceback.print_exc()
        report = dict(status="failed", error=traceback.format_exc())
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("DEVELOPMENT " + report["status"], flush=True)
    os._exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    sys.exit(main())
