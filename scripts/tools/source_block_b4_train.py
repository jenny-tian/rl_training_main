"""Full-state paired PPO trial. Exactly 25 updates, then save and exit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import sys
import traceback

from source_block_b2_protocol import checkpoint_for, save_json
from source_block_b2_train import assert_equal_state
from source_block_b4_protocol import NUM_ENVS, RAW, SEEDS, STEPS, TASKS, UPDATES, config_audit


def train(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    import gymnasium as gym
    import torch
    import rl_training.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner
    from rl_training.source_curriculum.allocation import b2_column_counts
    from rl_training.source_curriculum.learning import wrap_source_curriculum_learning

    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

    checkpoint, sha = checkpoint_for("terrain3497")
    configs, agents, audit = config_audit(args.seed, NUM_ENVS)
    save_json(args.output / "config_audit.json", audit)
    cfg, agent = configs[args.branch], agents[args.branch]
    agent.seed, agent.device, agent.max_iterations = args.seed, "cuda:0", UPDATES
    agent.logger = "tensorboard"
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log_dir = args.output / "training"
    log_dir.mkdir()
    dump_yaml(str(log_dir / "params/env.yaml"), cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent)
    raw = gym.make(TASKS[args.branch], cfg=cfg)
    env = wrap_source_curriculum_learning(RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions))
    base = raw.unwrapped
    actual_counts = torch.bincount(base.scene.terrain.terrain_types, minlength=20).cpu().tolist()
    if actual_counts != b2_column_counts(NUM_ENVS):
        raise ValueError("training allocation differs from B2")
    dims = base.observation_manager.group_obs_dim
    if tuple(dims["policy"]) != (45,) or tuple(dims["critic"]) != (48,):
        raise ValueError("observation dimensions changed")
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log_dir), device="cuda:0")
    runner.add_git_repo_to_log(__file__)
    runner.load(str(checkpoint))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = runner.alg.save()
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert_equal_state(saved[key], restored[key], key)
    if runner.alg.learning_rate != .0003844:
        raise ValueError("initial algorithm LR changed")
    if not all(p.requires_grad for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()):
        raise ValueError("actor or critic parameters are unexpectedly frozen")
    initial_steps = {k: int(v["step"]) for k, v in saved["optimizer_state_dict"]["state"].items()}
    robot = base.scene["robot"]
    initial = dict(root=digest(robot.data.root_state_w), joints=digest(robot.data.joint_pos),
        joint_velocity=digest(robot.data.joint_vel), commands=digest(base.command_manager.get_command("base_velocity")),
        materials=digest(robot.root_physx_view.get_material_properties()),
        terrain_columns=digest(base.scene.terrain.terrain_types),
        torch_cpu_rng=digest(torch.get_rng_state()), torch_cuda_rng=digest(torch.cuda.get_rng_state()))
    report = dict(status="running", branch=args.branch, seed=args.seed, task=TASKS[args.branch],
        checkpoint=str(checkpoint), checkpoint_sha256=sha, restored_exact=True,
        restore_fields=["actor", "critic", "learned_std", "Adam"], all_parameters_trainable=True,
        initial_iteration=runner.current_learning_iteration, initial_algorithm_learning_rate=runner.alg.learning_rate,
        initial_optimizer_group_lrs=[g["lr"] for g in runner.alg.optimizer.param_groups],
        initial_hashes=initial, environment_column_counts=actual_counts,
        max_updates=UPDATES, completed_updates=0, environment_steps=0, update_metrics=[],
        evaluation_automatic_extension=False)
    save_json(args.output / "training_report.json", report)
    original_step, original_log = env.step, runner.logger.log

    def finite_step(actions):
        if report["environment_steps"] >= UPDATES * STEPS:
            raise RuntimeError("training interaction budget exceeded")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("nonfinite training actions")
        if not report["environment_steps"]:
            report["first_action_sha256"] = digest(actions)
            report["first_episode_length_sha256"] = digest(env.episode_length_buf)
        observations, rewards, dones, extras = original_step(actions)
        report["environment_steps"] += 1
        if not torch.isfinite(rewards).all() or not all(torch.isfinite(v).all() for v in observations.values()):
            raise FloatingPointError("nonfinite training observations or rewards")
        return observations, rewards, dones, extras

    def checked_log(**kwargs):
        if not all(math.isfinite(float(v)) for v in kwargs["loss_dict"].values()):
            raise FloatingPointError("nonfinite training loss")
        report["completed_updates"] += 1
        if report["completed_updates"] > UPDATES:
            raise RuntimeError("training update budget exceeded")
        report["update_metrics"].append(dict(update=report["completed_updates"],
            learning_rate=runner.alg.learning_rate, losses={k: float(v) for k, v in kwargs["loss_dict"].items()},
            mean_action_std=float(kwargs["action_std"].mean())))
        original_log(**kwargs)
        save_json(args.output / "training_report.json", report)
        print(f"B4 {args.branch} seed={args.seed} {report['completed_updates']}/{UPDATES}", flush=True)

    env.step, runner.logger.log = finite_step, checked_log
    runner.learn(num_learning_iterations=UPDATES, init_at_random_ep_len=True)
    if report["completed_updates"] != UPDATES or report["environment_steps"] != UPDATES * STEPS:
        raise ValueError("incomplete training budget")
    current = runner.alg.save()
    for name in ("actor_state_dict", "critic_state_dict"):
        if not all(torch.isfinite(v).all() for v in current[name].values()):
            raise FloatingPointError("nonfinite model parameters")
        if not any(not torch.equal(saved[name][k], v.cpu()) for k, v in current[name].items()):
            raise ValueError(f"{name} did not change")
    increments = {int(v["step"]) - initial_steps[k] for k, v in current["optimizer_state_dict"]["state"].items()}
    if increments != {UPDATES * 20}:
        raise ValueError(f"unexpected optimizer step increments: {increments}")
    path = args.output / "model_0025.pt"
    runner.save(str(path), infos=dict(b4_updates=UPDATES, branch=args.branch, seed=args.seed,
                                   adaptive_learning_rate=runner.alg.learning_rate))
    state = base.source_block_state
    stats = state.stats.snapshot()
    stats["contact_diagnostics"] = state.contact_diagnostics.snapshot()
    save_json(args.output / "statistics.json", stats)
    report.update(status="budget_completed", output_checkpoint=str(path),
        output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_transitions=UPDATES * STEPS * NUM_ENVS, optimizer_step_increase=UPDATES * 20,
        final_algorithm_learning_rate=runner.alg.learning_rate,
        final_optimizer_group_lrs=[g["lr"] for g in runner.alg.optimizer.param_groups],
        final_iteration=runner.current_learning_iteration)
    save_json(args.output / "training_report.json", report)
    print(f"B4 FINISHED {args.branch} seed={args.seed} updates={UPDATES}", flush=True)
    # Bound worker teardown as in the existing B2 training worker.
    sys.stdout.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=tuple(TASKS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = RAW / "training" / f"{args.branch}_seed{args.seed}"
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
                   gpu=args.gpu, output=args.output, timeout=3600)
    try:
        train(args)
    except Exception:
        traceback.print_exc()
        save_json(args.output / "training_error.json", dict(error=traceback.format_exc()))
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
