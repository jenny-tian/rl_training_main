"""Real 120/4096-environment B1 verification; no policy training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]
TASK = "SourceBlock-L2-Deeprobotics-Lite3-v0"
REFERENCE_TASK = "SourceBlock-ReferenceRough-Deeprobotics-Lite3-v0"
B2_TASK = "SourceBlock-B2-Deeprobotics-Lite3-v0"
CHECKPOINT = ROOT / "logs/rsl_rl/deeprobotics_lite3_source_curriculum/2026-11-10_16-00-10_full_seed42/model_3497.pt"
CHECKPOINT_SHA = "47889ee59d07416287332164a38566a812d6ebbc0be637e13752829f8891d49c"


def check(args, checks, report):
    import gymnasium as gym
    import torch
    import rl_training
    import rl_training.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import SceneEntityCfg
    from rl_training.source_curriculum.allocation import b2_column_counts, targeted_column_counts
    from rl_training.source_curriculum.block import axis_vectors
    from rl_training.source_curriculum.rows import COLUMN_SPECS
    from rl_training.source_scan.frozen_actor import load_frozen_source_actor
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.source_block import source_block_bounds, source_block_completion
    from source_curriculum_check import audit_collision_groups, install_collision_recorder
    if not Path(rl_training.__file__).resolve().is_relative_to(ROOT / "source/rl_training"):
        raise ValueError("wrong rl_training package")
    install_collision_recorder()
    b2 = getattr(args, "b2", False)
    reference = getattr(args, "reference_rough", False) or b2
    task = B2_TASK if b2 else REFERENCE_TASK if reference else TASK
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs)
    cfg.seed = 42
    cfg.observations.policy.enable_corruption = False
    started = time.monotonic()
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    robot, terrain = base.scene["robot"], base.scene.terrain
    report["gym_make_s"] = time.monotonic() - started
    checks.add("startup_height", bool((robot.data.root_link_pos_w[:, 2] > 2.9).all()), "pre-reset warmup above map")
    report["collision_groups"] = audit_collision_groups(base.scene)
    checks.add("collision_groups", True, "robot environments isolated; shared ground")
    checks.add("block_geometry_profile", cfg.source_fixed_difficulty == "L2" and tuple(terrain.terrain_origins.shape) == (10, 20, 3)
               and tuple(cfg.scene.terrain.terrain_generator.size) == (8., 8.)
               and cfg.source_block_rough_profile == ("reference_uniform" if reference else "source_l2"),
               "200 blocks, 8x8m; " + ("reference rough, other families L2" if reference else "all families L2"))
    checks.add("observation_dimensions", tuple(base.observation_manager.group_obs_dim["policy"]) == (45,)
               and tuple(base.observation_manager.group_obs_dim["critic"]) == (48,), str(base.observation_manager.group_obs_dim))
    obs, _ = env.reset()
    state = base.source_block_state
    rough_probe = None
    if reference:
        from source_block_reference_rough_check import ReferenceRoughProbe
        rough_probe = ReferenceRoughProbe(base, checks, report)
    counts = torch.bincount(terrain.terrain_types, minlength=20).tolist()
    report["environment_column_counts"] = counts
    checks.add("allocation", counts == (b2_column_counts if b2 else targeted_column_counts)(args.num_envs), str(counts))
    command = base.command_manager.get_term("base_velocity")
    tile = ~state.omni
    def check_turn():
        turn = terrain.terrain_types == 9
        c = command.command[turn]
        return (bool((c[:, :2] == 0).all()) and bool((c[:, 2].abs() >= .2).all())
                and bool((c[:, 2].abs() <= .8).all()) and not bool(command.is_heading_env[turn].any())
                and not bool(command.is_standing_env[turn].any()))
    turn_ok = True
    checks.add("exit_heading", bool((torch.remainder(command.heading_target[tile] - state.axes[tile] * (torch.pi / 2) + torch.pi,
               2 * torch.pi) - torch.pi).abs().max() < 1e-5), "commands follow selected world axis")
    checks.add("tile_velocity", bool(((command.command[tile, 0] >= .25) & (command.command[tile, 0] <= .5)).all())
               and bool((command.command[tile, 1] == 0).all()), "tile vx .25-.5, vy 0; omni retains original distribution")
    seen = torch.zeros(10, 20, dtype=torch.bool, device=base.device)
    clearance_min, clearance_max = float("inf"), -float("inf")
    scan_ok = True
    for k in range(10):
        if k:
            obs, _ = env.reset()
        if b2:
            turn_ok &= check_turn()
        seen[terrain.terrain_levels, terrain.terrain_types] = True
        hits = base.scene["height_scanner_base"].data.ray_hits_w
        clearance = robot.data.root_link_pos_w[:, 2] - hits[..., 2].amax(-1)
        clearance_min = min(clearance_min, float(clearance.min()))
        clearance_max = max(clearance_max, float(clearance.max()))
        expected = ((clearance - .375).abs() < 2e-4)
        if rough_probe is not None:
            expected[rough_probe.mask] = ((clearance[rough_probe.mask] >= .375 - 2e-4)
                                           & (clearance[rough_probe.mask] <= .425 + 2e-4))
            rough_probe.observe_reset()
        scan_ok &= bool(torch.isfinite(hits).all()) and bool(expected.all())
    checks.add("all_replicas_reachable", bool(seen.all()), f"visited={int(seen.sum())}/200 in ten resets")
    checks.add("reset_clearance", scan_ok, f"root minus scan=[{clearance_min},{clearance_max}] "
               + ("rough .375-.425m; other tiles .375m" if reference else "expected .375m"))
    if rough_probe is not None:
        rough_probe.finish_resets()
    checks.add("manual_resets_not_outcomes", int(state.stats.outcomes.sum()) == 0, f"endings={int(state.stats.outcomes.sum())}")
    body_cfg = SceneEntityCfg("robot", body_names=".*")
    body_cfg.resolve(base.scene)
    endpoint_mask = (terrain.terrain_types == 4) | (terrain.terrain_types == 5)
    if rough_probe is not None:
        endpoint_mask |= rough_probe.mask
    ids = torch.nonzero(endpoint_mask).flatten()
    pose = robot.data.root_state_w[ids, :7].clone()
    # A point just beyond the threshold avoids floating-point cancellation
    # at distant world origins (3.4 can reconstruct as 3.3999996).
    pose[:, :2] = terrain.env_origins[ids, :2] + axis_vectors(state.axes[ids]) * 3.41
    pose[:, 2] = .375  # the outer platform of both native stairs is z=0
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=base.device), env_ids=ids)
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(base.step_dt)
    complete = source_block_completion(base)
    outside = source_block_bounds(base, body_cfg)
    root, feet = state.progress()
    report["endpoint_probe"] = dict(root_progress_range=[float(root[ids].min()), float(root[ids].max())],
                                     min_foot_progress=float(feet[ids].min()), completed=int(complete[ids].sum()),
                                     outside=int(outside[ids].sum()), tested=int(ids.numel()))
    checks.add("completion_before_boundary", bool(complete[ids].all()) and not bool(outside[ids].any()), json.dumps(report["endpoint_probe"]))
    # Invalid state must fail locally; probe the pure boundary using one genuine
    # outward pose and check separate failure/truncation semantics.
    all_ids = torch.arange(base.num_envs, device=base.device)
    pose = robot.data.root_state_w[:, :7].clone()
    pose[:, :2] = terrain.env_origins[:, :2] + axis_vectors(state.axes) * 3.95
    robot.write_root_pose_to_sim(pose, env_ids=all_ids)
    robot.write_root_velocity_to_sim(torch.zeros(base.num_envs, 6, device=base.device), env_ids=all_ids)
    base.scene.write_data_to_sim(); base.sim.forward(); base.scene.update(base.step_dt)
    failure = source_block_bounds(base, body_cfg, omni=False)
    truncation = source_block_bounds(base, body_cfg, omni=True)
    checks.add("boundary_semantics", torch.equal(failure, tile) and torch.equal(truncation, state.omni), "tile failure; omni truncation")
    obs, _ = env.reset()
    actor = load_frozen_source_actor(CHECKPOINT, device=base.device, expected_sha256=CHECKPOINT_SHA)
    before = int(state.stats.transitions.sum())
    started = time.monotonic()
    finite = True
    for _ in range(100):
        obs, reward, terminated, truncated, _ = env.step(actor(obs).clamp(-100, 100))
        if b2:
            turn_ok &= check_turn()
        finite &= bool(torch.isfinite(reward).all()) and all(bool(torch.isfinite(v).all()) for v in obs.values())
        if rough_probe is not None:
            rough_probe.observe_step()
    torch.cuda.synchronize()
    report["rollout_seconds"] = time.monotonic() - started
    checks.add("finite_rollout", finite, "100 frozen-policy steps, finite observations/rewards")
    checks.add("exact_transition_counts", int(state.stats.transitions.sum()) - before == args.num_envs * 100,
               f"new transitions={int(state.stats.transitions.sum()) - before}")
    checks.add("all_conditions_sampled", bool((state.stats.transitions.sum(1) > 0).all()), "all 20 conditions sampled")
    if rough_probe is not None:
        rough_probe.finish_rollout()
    if b2:
        command._resample_command(torch.arange(base.num_envs, device=base.device))
        command._update_command()
        checks.add("persistent_pure_turn", turn_ok and check_turn(), "ten resets, 100 steps, resample and heading update")
        checks.add("contact_diagnostics", state.contact_diagnostics is not None,
                   "terminal history saved before automatic reset")
        report["contact_diagnostics"] = state.contact_diagnostics.snapshot()
    report["statistics"] = state.stats.snapshot()
    report["physical_row_histogram"] = torch.bincount(terrain.terrain_levels, minlength=10).tolist()
    report["checkpoint"] = str(CHECKPOINT)
    report["checkpoint_sha256"] = CHECKPOINT_SHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, choices=(120, 4096), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--reference-rough", action="store_true")
    parser.add_argument("--b2", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--num-envs", str(args.num_envs),
                   "--output", str(args.output)]
        if args.reference_rough:
            command.append("--reference-rough")
        if args.b2:
            command.append("--b2")
        return run(command, gpu=args.gpu, output=args.output)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    from source_curriculum_check import Checks
    checks = Checks()
    report = dict(task=B2_TASK if args.b2 else REFERENCE_TASK if args.reference_rough else TASK,
                  num_envs=args.num_envs, checks=checks.results)
    try:
        check(args, checks, report)
    except Exception as exc:
        traceback.print_exc()
        checks.add("execution", False, f"{type(exc).__name__}: {exc}")
    report["all_passed"] = checks.all_passed
    (args.output / "checks.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"SUMMARY {sum(c['passed'] for c in checks.results)}/{len(checks.results)}", flush=True)
    os._exit(0 if checks.all_passed else 1)


if __name__ == "__main__":
    sys.exit(main())
