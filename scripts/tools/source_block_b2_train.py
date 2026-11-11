"""At most 300 PPO updates, with synchronous independent checks every 100."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
from source_block_b2_protocol import TASK, EVIDENCE, checkpoint_for, decision, evaluations, save_json


class DevelopmentStop(Exception):
    pass


def assert_equal_state(a, b, path="state"):
    import torch
    if isinstance(a, torch.Tensor):
        if not torch.equal(a.cpu(), b.cpu()):
            raise ValueError(f"restored tensor mismatch: {path}")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError(f"restored keys mismatch: {path}")
        for key in a:
            assert_equal_state(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"restored length mismatch: {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_equal_state(x, y, f"{path}/{i}")
    elif a != b:
        raise ValueError(f"restored value mismatch: {path}")


def train(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    import gymnasium as gym
    import torch
    import rl_training.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner
    from rl_training.source_curriculum.learning import wrap_source_curriculum_learning

    checkpoint, sha = checkpoint_for(args.start)
    baseline_dir = EVIDENCE / "baseline" / args.start
    baseline = json.loads((baseline_dir / "blocks/evaluation.json").read_text())
    if not json.loads((baseline_dir / "decision.json").read_text())["continue_training"]:
        raise ValueError("baseline failed Q4; do not start training")
    other = "terrain3497" if args.start == "flat1998" else "flat1998"
    paired = json.loads((EVIDENCE / "baseline" / other / "blocks/evaluation.json").read_text())
    if baseline["initial_state_hashes_by_column"] != paired["initial_state_hashes_by_column"]:
        raise ValueError("starting policies were not evaluated on paired initial states")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=4096)
    cfg.seed = args.seed
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent.seed, agent.device, agent.max_iterations = args.seed, "cuda:0", 300
    agent.logger = "tensorboard"
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
    log_dir = args.output / "training"
    log_dir.mkdir()
    dump_yaml(str(log_dir / "params/env.yaml"), cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent)
    raw = gym.make(TASK, cfg=cfg)
    env = wrap_source_curriculum_learning(RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions))
    from rl_training.source_curriculum.allocation import b2_column_counts
    actual_counts = torch.bincount(raw.unwrapped.scene.terrain.terrain_types, minlength=20).cpu().tolist()
    if actual_counts != b2_column_counts(4096):
        raise ValueError("training allocation differs from registered B2 counts")
    dimensions = raw.unwrapped.observation_manager.group_obs_dim
    if tuple(dimensions["policy"]) != (45,) or tuple(dimensions["critic"]) != (48,):
        raise ValueError("training observation dimensions changed")
    original_step = env.step

    def finite_step(actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("nonfinite training actions")
        observations, rewards, dones, extras = original_step(actions)
        if not torch.isfinite(rewards).all() or not all(torch.isfinite(v).all() for v in observations.values()):
            raise FloatingPointError("nonfinite training observations or rewards")
        return observations, rewards, dones, extras

    env.step = finite_step
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log_dir), device="cuda:0")
    runner.add_git_repo_to_log(__file__)
    runner.load(str(checkpoint))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = runner.alg.save()
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert_equal_state(saved[key], restored[key], key)
    initial_steps = {k: int(v["step"]) for k, v in saved["optimizer_state_dict"]["state"].items()}
    report = dict(status="running", start=args.start, seed=args.seed, checkpoint=str(checkpoint), sha256=sha,
                  load_rule="full actor, critic, learned action std and Adam; adaptive LR starts from common config",
                  restored_exact=True, initial_iteration=runner.current_learning_iteration,
                  initial_algorithm_learning_rate=runner.alg.learning_rate,
                  initial_optimizer_group_lrs=[g["lr"] for g in runner.alg.optimizer.param_groups],
                  max_updates=300, completed_updates=0, checks=[])
    report["environment_column_counts"] = actual_counts
    original_log = runner.logger.log

    def checked_log(**kwargs):
        import math
        if not all(math.isfinite(float(v)) for v in kwargs["loss_dict"].values()):
            raise FloatingPointError("nonfinite training loss")
        original_log(**kwargs)
        report["completed_updates"] += 1
        updates = report["completed_updates"]
        if updates % 100:
            return
        check_dir = args.output / f"update_{updates:04d}"
        check_dir.mkdir()
        path = check_dir / "model.pt"
        runner.save(str(path), infos=dict(b2_updates=updates, adaptive_learning_rate=runner.alg.learning_rate))
        current = runner.alg.save()
        for name in ("actor_state_dict", "critic_state_dict"):
            if not all(torch.isfinite(v).all() for v in current[name].values()):
                raise FloatingPointError("nonfinite model parameters")
            if not any(not torch.equal(saved[name][k], v.cpu()) for k, v in current[name].items()):
                raise ValueError(f"{name} did not change")
        increments = {int(v["step"]) - initial_steps[k] for k, v in current["optimizer_state_dict"]["state"].items()}
        if increments != {updates * 20}:
            raise ValueError(f"unexpected optimizer steps: {increments}")
        state = raw.unwrapped.source_block_state
        stats = state.stats.snapshot()
        stats["contact_diagnostics"] = state.contact_diagnostics.snapshot()
        stats["environment_column_counts"] = torch.bincount(raw.unwrapped.scene.terrain.terrain_types, minlength=20).cpu().tolist()
        save_json(check_dir / "statistics.json", stats)
        current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"[B2] pause at {updates}; checking {path}", flush=True)
        block, flat = evaluations(path, current_sha, check_dir / "evaluation", args.gpu)
        gate = decision(block, flat, baseline)
        gate.update(updates=updates, checkpoint=str(path), sha256=current_sha,
                    adaptive_learning_rate=runner.alg.learning_rate, optimizer_step_increase=updates * 20)
        save_json(check_dir / "decision.json", gate)
        report["checks"].append(gate)
        save_json(args.output / "training_report.json", report)
        print(f"[B2] update={updates} continue={gate['continue_training']} reasons={gate['reasons']}", flush=True)
        if not gate["continue_training"]:
            raise DevelopmentStop()

    runner.logger.log = checked_log
    try:
        runner.learn(num_learning_iterations=300, init_at_random_ep_len=True)
        report["status"] = "budget_completed"
    except DevelopmentStop:
        runner.logger.stop_logging_writer()
        report["status"] = "stopped_by_development_rule"
    report["training_transitions"] = report["completed_updates"] * 4096 * 24
    save_json(args.output / "training_report.json", report)
    print(f"[B2] FINISHED {args.start} seed={args.seed} {report['status']} updates={report['completed_updates']}", flush=True)
    # Same bounded worker-exit convention as block validation/evaluation.
    sys.stdout.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=("flat1998", "terrain3497"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = EVIDENCE / "training" / f"{args.start}_seed{args.seed}"
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), "--child", "--start", args.start,
                    "--seed", str(args.seed), "--gpu", str(args.gpu)], gpu=args.gpu, output=args.output, timeout=10800)
    try:
        train(args)
    except Exception:
        traceback.print_exc()
        save_json(args.output / "training_error.json", dict(error=traceback.format_exc()))
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
