"""GPU check for SourceCurriculum-Deeprobotics-Lite3-v0 (stage 1, Task 4).

Builds the 4096-env task once in Isaac Sim, prints one ``PASS/FAIL name detail``
line per check, writes a JSON report and exits 0 only if every check passes.
Run on one GPU with at least 12 GiB free.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training"), str(ROOT / "scripts/tools")]

TASK = "SourceCurriculum-Deeprobotics-Lite3-v0"
TARGETED_TASK = "SourceCurriculum-Targeted-Deeprobotics-Lite3-v0"
NUM_ENVS = 4096
FROZEN_STEPS = 50
EXPECTED_TERMINATIONS = {"time_out", "illegal_contact", "bad_orientation_2", "source_curriculum_safety",
                         "source_curriculum_tile_bounds", "source_curriculum_completion",
                         "source_curriculum_omni_truncation"}
EXPECTED_CURRICULUM = {"source_curriculum_levels"}
FLAT_TILE_COLUMNS = (7, 8, 13, 14)  # flat geometry: injected poses need no terrain height lookup


def runtime_failure(line):
    """PhysX can log a fatal GPU error while returning finite, stale tensors."""
    markers = ("PhysX error:", "Scene state is corrupted", "CUDA error", "CUDA_ERROR_",
               "failed to allocate", "out of memory", "buffer overflow", "buffer capacity exceeded")
    return any(marker.lower() in line.lower() for marker in markers)


def supervise(args):
    """Inspect native simulator output, including errors before gym.make returns."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if visible not in ("0", "1"):
        raise ValueError("CUDA_VISIBLE_DEVICES must select physical GPU 0 or 1")
    free = int(subprocess.check_output([
        "nvidia-smi", "--id=" + visible, "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True).strip())
    if free < 12288:
        raise RuntimeError(f"GPU{visible} admission rejected: free={free} MiB < 12288 MiB")
    log_path = args.output.with_suffix(".sim.log")
    failures = []
    geometry = None
    child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "PYTHONUNBUFFERED": "1"}
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--num-envs", str(args.num_envs),
               "--output", str(args.output)]
    if getattr(args, "targeted", False):
        command.append("--targeted")
    started = time.monotonic()
    memory_samples = []
    monitor_stop = threading.Event()

    def monitor_memory():
        while not monitor_stop.is_set():
            try:
                used = int(subprocess.check_output([
                    "nvidia-smi", "--id=" + visible, "--query-gpu=memory.used", "--format=csv,noheader,nounits"
                ], text=True).strip())
                memory_samples.append(used)
            except (ValueError, subprocess.SubprocessError):
                pass
            monitor_stop.wait(0.5)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    with log_path.open("w") as log, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=child_env,
        start_new_session=True,
    ) as child:
        for line in child.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            if line.startswith("GEOMETRY "):
                geometry = json.loads(line[len("GEOMETRY "):])
            if runtime_failure(line):
                failures.append(line.strip())
                # Stop only our simulator child. Results from a corrupted scene have no value.
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
        returncode = child.wait()
    monitor_stop.set()
    monitor.join(timeout=5)
    child_report_exists = args.output.exists()
    report = json.loads(args.output.read_text()) if child_report_exists else {
        "task": TASK, "num_envs": args.num_envs, "checks": []}
    healthy = not failures and returncode == 0 and child_report_exists and bool(report.get("checks"))
    result = {"name": "runtime_simulation_health", "passed": healthy,
              "detail": f"child_exit={returncode} fatal_messages={failures[:3]}"}
    report["checks"].append(result)
    report["runtime"] = {"physical_gpu": int(visible), "admission_free_mib": free,
                         "simulator_log": str(log_path), "child_exit": returncode,
                         # nvidia-smi reports the whole physical GPU, including other processes.
                         "peak_total_gpu_used_mib_sampled": max(memory_samples, default=None),
                         "gpu_memory_sample_interval_seconds": 0.5,
                         "gpu_memory_sample_count": len(memory_samples),
                         "fatal_messages": failures, "wall_seconds": time.monotonic() - started}
    report["collision_geometry"] = geometry
    report["all_passed"] = healthy and all(item["passed"] for item in report["checks"])
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"{'PASS' if healthy else 'FAIL'} {result['name']} {result['detail']}", flush=True)
    return 0 if report["all_passed"] else 1


class Checks:
    def __init__(self):
        self.results = []

    def add(self, name, passed, detail):
        passed = bool(passed)
        self.results.append({"name": name, "passed": passed, "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'} {name} {detail}", flush=True)
        return passed

    def guard(self, name, fn):
        try:
            fn()
        except Exception as exc:  # a crashing check is a failing check; keep running the others
            traceback.print_exc()
            self.add(name, False, f"exception: {type(exc).__name__}: {exc}")

    @property
    def all_passed(self):
        return bool(self.results) and all(r["passed"] for r in self.results)


def install_row_recorder():
    """Record, per generated tile, the row the terrain function derived from its difficulty (check 9)."""
    from rl_training.tasks.manager_based.locomotion.velocity.terrains import source_curriculum_terrain as sct
    from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_terrain_generator import (
        SourceTerrainGenerator)
    pending, tiles = [], []
    real_row = sct.row_from_difficulty
    real_add = SourceTerrainGenerator._add_sub_terrain

    def row_from_difficulty(difficulty, num_rows):
        row = real_row(difficulty, num_rows)
        pending.append(row)
        return row

    def add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        tiles.append({"grid_row": int(row), "grid_col": int(col), "built_rows": list(pending)})
        pending.clear()
        return real_add(self, mesh, origin, row, col, sub_terrain_cfg)

    sct.row_from_difficulty = row_from_difficulty
    SourceTerrainGenerator._add_sub_terrain = add_sub_terrain
    return tiles


def install_collision_recorder():
    """Audit generated collider extents before PhysX startup can allocate midphase memory."""
    import numpy as np
    import re
    from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_terrain_importer import (
        SourceTerrainImporter)
    parts = []
    real_part = SourceTerrainImporter._import_mesh_part
    real_mesh = SourceTerrainImporter.import_mesh

    def part(self, path, mesh, material):
        match = re.search(r"tile_r(\d+)_c(\d+)", path)
        if match is None:
            raise ValueError(f"collider has no tile ownership: {path}")
        row, col = map(int, match.groups())
        cfg = self.cfg.terrain_generator
        lower = np.array([row * cfg.size[0] - cfg.num_rows * cfg.size[0] / 2,
                          col * cfg.size[1] - cfg.num_cols * cfg.size[1] / 2])
        within = bool(np.all(mesh.bounds[0, :2] >= lower - 1e-6)
                      and np.all(mesh.bounds[1, :2] <= lower + cfg.size + np.array([1e-6, 1e-6])))
        if not within or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"non-finite or out-of-tile collider: {path} bounds={mesh.bounds}")
        parts.append({"path": path, "triangles": len(mesh.faces), "bounds": mesh.bounds.tolist()})
        return real_part(self, path, mesh, material)

    def mesh(self, name, terrain_mesh):
        out = real_mesh(self, name, terrain_mesh)
        print("GEOMETRY " + json.dumps({"parts": len(parts), "total_triangles": sum(p["triangles"] for p in parts),
              "max_part_triangles": max(p["triangles"] for p in parts),
              "all_finite_within_tile": True, "colliders": parts}), flush=True)
        return out

    SourceTerrainImporter._import_mesh_part = part
    SourceTerrainImporter.import_mesh = mesh


def timed_curriculum(base, timings):
    import torch
    manager = base.curriculum_manager
    index = manager._term_names.index("source_curriculum_levels")
    term_cfg = manager._term_cfgs[index]
    term = term_cfg.func

    class Timed:  # the manager also calls term.reset(), so delegate every other attribute
        def __getattr__(self, name):
            return getattr(term, name)

        def __call__(self, env, env_ids, **params):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = term(env, env_ids, **params)
            torch.cuda.synchronize()
            n = env.num_envs if env_ids is None or isinstance(env_ids, slice) else len(env_ids)
            timings.append({"seconds": time.perf_counter() - start, "num_reset_envs": int(n)})
            return out

    term_cfg.func = Timed()
    return term


def scan_height_check(base, ids):
    """Root z minus 9-ray max hit z in [0.25, 0.45]; rays finite and within 0.08 m in xy of the root."""
    import torch
    if ids.numel() == 0:
        return False, "envs=0 (no env left to check)", None
    root = base.scene["robot"].data.root_pos_w[ids]
    hits = base.scene.sensors["height_scanner_base"].data.ray_hits_w[ids]
    finite = torch.isfinite(hits).all(-1).all(-1)
    clearance = root[:, 2] - hits[..., 2].amax(-1)
    xy = (hits[..., :2] - root[:, None, :2]).norm(dim=-1).amax(-1)
    in_range = (clearance >= 0.25) & (clearance <= 0.45)
    ok = finite & in_range & (xy <= 0.08)
    c = clearance[torch.isfinite(clearance)]
    height_range = f"[{float(c.min()):.3f},{float(c.max()):.3f}]" if c.numel() else "no_finite_values"
    detail = (f"envs={ids.numel()} rays={hits.shape[1]} bad={int((~ok).sum())} non_finite={int((~finite).sum())} "
              f"clearance={height_range} max_xy={float(xy.max()):.4f}")
    return bool(ok.all()) and hits.shape[1] == 9, detail, ok


def audit_collision_groups(scene):
    """Require the installed Cloner's inverted self-and-ground-only graph."""
    if (scene.cfg.replicate_physics or scene.cfg.clone_in_fabric or not scene.cfg.filter_collisions
            or scene.cfg.robot.collision_group != 0 or set(scene._global_prim_paths) != {"/World/ground"}):
        raise ValueError("Source curriculum requires USD filtering and ground-only global collisions")
    stage = scene.stage
    if stage.GetPrimAtPath(scene.physics_scene_path).GetAttribute("physxScene:invertCollisionGroupFilter").Get() is not True:
        raise ValueError("collision filter must be inverted")
    global_group = "/World/collisions/global_group"
    groups = [f"/World/collisions/group{i}" for i in range(scene.cfg.num_envs)]
    if {str(p.GetPath()) for p in stage.GetPrimAtPath("/World/collisions").GetChildren()} != {*groups, global_group}:
        raise ValueError("collision group inventory mismatch")
    records = [(global_group, "/World/ground", {*groups, global_group})]
    records += [(group, path, {group, global_group}) for group, path in zip(groups, scene.env_prim_paths)]
    for group, path, allowed in records:
        prim = stage.GetPrimAtPath(group)
        includes = {str(p) for p in prim.GetRelationship("collection:colliders:includes").GetTargets()}
        actual = {str(p) for p in prim.GetRelationship("physics:filteredGroups").GetTargets()}
        if (prim.GetTypeName() != "PhysicsCollisionGroup" or includes != {path} or actual != allowed
                or prim.GetRelationship("collection:colliders:excludes").GetTargets()
                or prim.GetAttribute("collection:colliders:expansionRule").Get() != "expandPrims"
                or prim.GetAttribute("collection:colliders:includeRoot").Get() is True):
            raise ValueError(f"collision group contract violated: {group}")
    return {"inverted": True, "isolated_environment_groups": len(groups), "ground_global": True}


def pick(mask, count, generator_seed):
    import torch
    ids = mask.nonzero().flatten()
    if ids.numel() < count:
        raise RuntimeError(f"need {count} candidate envs, found {ids.numel()}")
    g = torch.Generator(device="cpu").manual_seed(generator_seed)
    return [int(i) for i in ids[torch.randperm(ids.numel(), generator=g)[:count].to(ids.device)]]


def inject(base, ids, pos, roll=None):
    """Teleport envs to world positions with zero velocity; optional roll (rad), yaw 0."""
    import torch
    robot = base.scene["robot"]
    ids_t = torch.as_tensor(ids, device=base.device, dtype=torch.long)
    pose = torch.zeros(len(ids), 7, device=base.device)
    pose[:, :3] = torch.as_tensor(pos, device=base.device, dtype=torch.float32)
    rolls = torch.zeros(len(ids), device=base.device) if roll is None else torch.as_tensor(
        roll, device=base.device, dtype=torch.float32)
    pose[:, 3] = torch.cos(rolls / 2)
    pose[:, 4] = torch.sin(rolls / 2)
    robot.write_root_pose_to_sim(pose, env_ids=ids_t)
    robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=base.device), env_ids=ids_t)


def build_and_check(app, checks, report, num_envs, *, targeted=False):
    import gymnasium as gym
    import torch
    import rl_training
    import rl_training.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_training.source_curriculum.rows import NUM_COLUMNS, NUM_ROWS, OMNI_COLUMNS
    from rl_training.source_scan.frozen_actor import load_frozen_source_actor
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as reward_mdp
    from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_terrain_importer import (
        SourceTerrainImporter)
    from source_adaptation_runtime import OLD

    source_root = ROOT / "source/rl_training"
    if not Path(rl_training.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError(f"rl_training imported from {rl_training.__file__}, expected {source_root}")
    tiles = install_row_recorder()
    install_collision_recorder()
    task = TARGETED_TASK if targeted else TASK
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    env_cfg.seed = 42
    env_cfg.observations.policy.enable_corruption = False  # obs noise off for the check only
    print("SCENE " + json.dumps({"replicate_physics": env_cfg.scene.replicate_physics,
          "filter_collisions": env_cfg.scene.filter_collisions,
          "robot_collision_group": env_cfg.scene.robot.collision_group,
          "num_envs": env_cfg.scene.num_envs}), flush=True)
    setup_start = time.perf_counter()
    env = gym.make(task, cfg=env_cfg)
    report["gym_make_seconds"] = time.perf_counter() - setup_start
    base = env.unwrapped
    robot = base.scene["robot"]
    startup_z = robot.data.root_pos_w[:, 2]
    default_z = robot.data.default_root_state[:, 2]
    checks.add("0_startup_placement", bool((startup_z > 2.9).all()) and bool((default_z == 0.375).all()),
               f"startup_root_z=[{float(startup_z.min()):.3f},{float(startup_z.max()):.3f}] "
               f"default_root_z=[{float(default_z.min()):.3f},{float(default_z.max()):.3f}]")
    def check_collision_groups():
        report["usd_collision_graph"] = audit_collision_groups(base.scene)
        checks.add("0_collision_groups", True, json.dumps(report["usd_collision_graph"]))
    checks.guard("0_collision_groups", check_collision_groups)
    terrain = base.scene.terrain
    command = base.command_manager.get_term("base_velocity")
    curriculum_timings = []
    curriculum = timed_curriculum(base, curriculum_timings)
    report["curriculum_timings"] = curriculum_timings

    # 9. tile rows (the grid is generated inside gym.make)
    def check_tile_rows():
        bad = [t for t in tiles if t["built_rows"] != [t["grid_row"]]]
        cells = {(t["grid_row"], t["grid_col"]) for t in tiles}
        ok = not bad and len(tiles) == NUM_ROWS * NUM_COLUMNS and len(cells) == NUM_ROWS * NUM_COLUMNS
        checks.add("9_tile_rows", ok, f"tiles={len(tiles)} distinct={len(cells)} mismatched={bad[:5]}")
    checks.guard("9_tile_rows", check_tile_rows)

    # 1. active term sets
    def check_terms():
        terms = set(base.termination_manager.active_terms)
        curr = set(base.curriculum_manager.active_terms)
        ok = terms == EXPECTED_TERMINATIONS and curr == EXPECTED_CURRICULUM
        checks.add("1_active_terms", ok, f"terminations={sorted(terms)} curriculum={sorted(curr)}")
    checks.guard("1_active_terms", check_terms)

    # 2. obs dims, gait level, single friction palette
    def check_dims():
        dims = base.observation_manager.group_obs_dim
        palette = terrain._configured_friction_candidates()
        ok = (tuple(dims["policy"]) == (45,) and tuple(dims["critic"]) == (48,) and reward_mdp.gait_level == 1.0
              and isinstance(terrain, SourceTerrainImporter) and len(palette) > 0)
        checks.add("2_obs_dims_gait_palette", ok,
                   f"policy={tuple(dims['policy'])} critic={tuple(dims['critic'])} gait_level={reward_mdp.gait_level} "
                   f"importer={type(terrain).__name__} palette={[round(float(v), 4) for v in palette]}")
    checks.guard("2_obs_dims_gait_palette", check_dims)

    obs, _ = env.reset()
    types = terrain.terrain_types
    omni = torch.isin(types, torch.as_tensor(OMNI_COLUMNS, device=types.device, dtype=types.dtype))
    tile = ~omni
    all_ids = torch.arange(base.num_envs, device=base.device)

    # 3. levels, column balance, origins
    def check_levels():
        levels = terrain.terrain_levels
        counts = torch.bincount(types.long(), minlength=NUM_COLUMNS)
        mean = base.num_envs / NUM_COLUMNS
        expected_origins = terrain.terrain_origins[levels.long(), types.long()]
        origin_err = float((terrain.env_origins - expected_origins).abs().max())
        tl, ol = levels[tile], levels[omni]
        if targeted:
            from rl_training.source_curriculum.allocation import TARGETED_PROFILE, targeted_column_counts
            expected_counts = torch.tensor(targeted_column_counts(base.num_envs), device=types.device)
            allocation_ok = (torch.equal(counts, expected_counts)
                             and terrain.cfg.source_curriculum_allocation == TARGETED_PROFILE
                             and torch.equal(command.tile_mask, tile)
                             and torch.equal(curriculum.omni, omni))
        else:
            allocation_ok = (float((counts.float() - mean).abs().max()) <= 5
                             and abs(int(omni.sum()) - base.num_envs * len(OMNI_COLUMNS) / NUM_COLUMNS) <= 20)
        report["environment_column_counts"] = counts.tolist()
        ok = (int(tl.min()) >= 0 and int(tl.max()) <= 4 and int(ol.min()) >= 1 and int(ol.max()) <= 8
              and allocation_ok and bool((counts > 0).all())
              and counts.numel() == NUM_COLUMNS and origin_err <= 1e-6)
        checks.add("3_levels_columns_origins", ok,
                   f"tile_levels=[{int(tl.min())},{int(tl.max())}] omni_levels=[{int(ol.min())},{int(ol.max())}] "
                   f"column_counts=[{int(counts.min())},{int(counts.max())}] mean={mean:.1f} omni={int(omni.sum())} "
                   f"origin_err={origin_err:.2e}")
    checks.guard("3_levels_columns_origins", check_levels)

    if targeted:
        def check_initial_counts():
            stats = curriculum.column_stats
            checks.add("12_initial_column_counts", int(stats.outcomes.sum()) == 0
                       and int(stats.starts.sum()) == base.num_envs,
                       f"starts={int(stats.starts.sum())} endings={int(stats.outcomes.sum())}")
        checks.guard("12_initial_column_counts", check_initial_counts)

    # 4. clearance over the 9-ray base scan after reset
    def check_scan_reset():
        ok, detail, _ = scan_height_check(base, all_ids)
        checks.add("4_scan_after_reset", ok, detail)
    checks.guard("4_scan_after_reset", check_scan_reset)

    # 5. tile commands vs omni command mix
    def check_commands():
        vel = command.vel_command_b
        t = tile.nonzero().flatten()
        o = omni.nonzero().flatten()
        tile_ok = (bool(((vel[t, 0] >= 0.25) & (vel[t, 0] <= 0.5)).all()) and bool((vel[t, 1] == 0).all())
                   and bool(command.is_heading_env[t].all()) and not bool(command.is_standing_env[t].any()))
        standing = int(command.is_standing_env[o].sum())
        lateral = int((vel[o, 1].abs() > 0).sum())
        backward = int((vel[o, 0] < 0).sum())
        tile_like = int(((vel[o, 0] >= 0.25) & (vel[o, 0] <= 0.5) & (vel[o, 1] == 0)).sum())
        omni_ok = standing > 0 and lateral > 0 and backward > 0 and tile_like < o.numel()
        checks.add("5_commands", tile_ok and omni_ok,
                   f"tile_vx=[{float(vel[t, 0].min()):.3f},{float(vel[t, 0].max()):.3f}] "
                   f"tile_|vy|max={float(vel[t, 1].abs().max()):.3f} tile_heading={bool(command.is_heading_env[t].all())} "
                   f"tile_standing={int(command.is_standing_env[t].sum())} omni n={o.numel()} standing={standing} "
                   f"lateral={lateral} backward={backward} tile_like={tile_like}")
    checks.guard("5_commands", check_commands)

    # 6 + 10. frozen actor rollout
    actor = load_frozen_source_actor(OLD, device="cuda:0")
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rl_training.source_curriculum.learning import wrap_source_curriculum_learning
    learning_env = wrap_source_curriculum_learning(RslRlVecEnvWrapper(env, clip_actions=100.0))
    obs = learning_env.get_observations()
    ever_reset = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    non_finite = {"obs": 0, "reward": 0, "contact": 0}
    step_seconds = []
    contact = base.scene.sensors["contact_forces"]
    first_curriculum_call = len(curriculum_timings)
    term_counts = {name: [] for name in base.termination_manager.active_terms}
    for _ in range(FROZEN_STEPS):
        with torch.inference_mode():
            action = actor(obs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        obs, reward, terminated, truncated, _ = env.step(action)
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - start)
        ever_reset |= terminated | truncated
        for name in term_counts:
            term_counts[name].append(int(base.termination_manager.get_term(name).sum()))
        non_finite["obs"] += sum(int((~torch.isfinite(v)).sum()) for v in obs.values())
        non_finite["reward"] += int((~torch.isfinite(reward)).sum())
        non_finite["contact"] += int((~torch.isfinite(contact.data.net_forces_w)).sum())
    survivors = (~ever_reset).nonzero().flatten()
    report["rollout_termination_counts_per_step"] = term_counts
    print("TERMS " + json.dumps({k: v[:12] for k, v in term_counts.items()}), flush=True)

    def check_rollout():
        ok, detail, _ = scan_height_check(base, survivors)
        nan_ok = not any(non_finite.values())
        checks.add("6_frozen_rollout", ok and nan_ok,
                   f"steps={FROZEN_STEPS} reset_envs={int(ever_reset.sum())} non_finite={non_finite} {detail}")
    checks.guard("6_frozen_rollout", check_rollout)

    default_z = base.scene["robot"].data.default_root_state[:, 2]
    level = terrain.terrain_levels
    flat_tile = torch.isin(types, torch.as_tensor(FLAT_TILE_COLUMNS, device=types.device, dtype=types.dtype))
    tm = base.termination_manager

    def alive():  # well into an episode: not reset during the rollout or by an injected step
        return base.episode_length_buf >= 40

    def term(name, ids):
        return [bool(v) for v in tm.get_term(name)[ids].tolist()]

    # 7a. far outside both the tile and the flat region
    def check_far_outside():
        a, = pick(tile & alive(), 1, 1)
        b, = pick(omni & alive(), 1, 2)
        inject(base, [a, b], [[500.0, 500.0, 1.0], [500.0, 500.0, 1.0]])
        env.step(actor(obs))
        bounds, trunc = term("source_curriculum_tile_bounds", [a, b]), term("source_curriculum_omni_truncation", [a, b])
        ok = bounds == [True, False] and trunc == [False, True]
        checks.add("7a_far_outside_masks", ok, f"tile_env={a} omni_env={b} tile_bounds={bounds} omni_trunc={trunc}")
    checks.guard("7a_far_outside_masks", check_far_outside)

    # 7b + 8. progress >= 9.25 masks and curriculum moves, one injected step
    def check_progress_and_moves():
        at2 = flat_tile & alive() & (level == 2)
        c, e, f, g = pick(at2, 4, 3)
        d, = pick(omni & alive(), 1, 4)
        origins = terrain.env_origins
        ids = [c, d, e, f, g]

        def at(i, dx, dz=0.0):
            o = origins[i]
            return [float(o[0]) + dx, float(o[1]), float(o[2] + default_z[i]) + dz]
        pos = [at(c, 9.30), at(d, 9.30), at(e, 9.30), at(f, 2.0, 0.05), at(g, 9.30, 0.05)]
        inject(base, ids, pos, roll=[0.0, 0.0, 0.0, 1.2, 1.2])
        before = level[ids].tolist()
        _, _, learning_dones, learning_extras = learning_env.step(actor(obs))
        completion = term("source_curriculum_completion", ids)
        trunc = term("source_curriculum_omni_truncation", [d])
        bounds = term("source_curriculum_tile_bounds", [c, e])
        ok7 = completion[:2] == [True, False] and trunc == [False] and bounds == [False, False]
        checks.add("7b_completion_masks", ok7,
                   f"tile_env={c} omni_env={d} completion={completion[:2]} omni_trunc={trunc} tile_bounds={bounds}")
        after = [int(level[i]) for i in (e, f, g)]
        reset = [int(base.episode_length_buf[i]) == 0 for i in (e, f, g)]
        failed = [bool(v) for v in tm.terminated[[e, f, g]].tolist()]
        origin_ok = all(torch.equal(origins[i], terrain.terrain_origins[level[i], types[i]]) for i in (e, f, g))
        ok8 = (after == [3, 1, 1] and all(reset) and failed == [False, True, True]
               and completion[2:] == [True, False, True] and origin_ok)
        checks.add("8_curriculum_moves", ok8,
                   f"envs complete/fail/both={[e, f, g]} levels {before[2:]}->{after} reset={reset} "
                   f"terminated={failed} completion={completion[2:]} origins_follow={origin_ok}")
        bootstrap = learning_extras['time_outs'][[e, f, g]].tolist()
        raw_timeouts = base.reset_time_outs[[e, f, g]].tolist()
        checks.add('11_failure_priority_bootstrap',
                   bootstrap == [True, False, False] and raw_timeouts == [True, False, True]
                   and learning_dones[[e, f, g]].tolist() == [1, 1, 1],
                   f'complete/fail/both bootstrap={bootstrap} raw_timeouts={raw_timeouts}')
    checks.guard("7b_8_progress_and_moves", check_progress_and_moves)

    if targeted:
        def check_anchor_counts():
            # Use real termination/manager flow on flat columns at rows 4/9;
            # stairs' physical contact is covered by the unchanged evaluator.
            ids = pick(flat_tile & alive(), 4, 20261111)
            rows = torch.tensor([4, 4, 9, 9], device=base.device)
            terrain.terrain_levels[ids] = rows
            terrain.env_origins[ids] = terrain.terrain_origins[rows, types[ids]]
            positions = []
            for i in ids:
                origin = terrain.env_origins[i]
                positions.append([float(origin[0]) + 9.30, float(origin[1]), float(origin[2] + default_z[i]) + .05])
            previous = curriculum.column_stats.outcomes.clone()
            inject(base, ids, positions, roll=[0., 1.2, 0., 1.2])
            learning_env.step(actor(obs))
            delta = curriculum.column_stats.outcomes - previous
            raw_c = term("source_curriculum_completion", ids)
            raw_f = tm.terminated[ids].tolist()
            ok = (raw_c == [True] * 4 and raw_f == [False, True, False, True]
                  and all(int(delta[int(types[i]), int(row), outcome]) >= 1
                          for i, row, outcome in zip(ids, rows, [0, 1, 0, 1])))
            checks.add("13_anchor_outcome_counts", ok,
                       f"envs={ids} before_rows={rows.tolist()} completion={raw_c} failure={raw_f}")
            report["column_statistics"] = curriculum.column_stats.snapshot()
        checks.guard("13_anchor_outcome_counts", check_anchor_counts)

    # 10. timing (report only)
    rollout_calls = curriculum_timings[first_curriculum_call:]
    triggered = [t for t in rollout_calls if t["num_reset_envs"] > 0]
    mean_step = sum(step_seconds) / len(step_seconds)
    timing = {"mean_env_step_s": mean_step, "env_step_s_min": min(step_seconds), "env_step_s_max": max(step_seconds),
              "curriculum_calls": len(curriculum_timings),
              "curriculum_call_s_mean": (sum(t["seconds"] for t in triggered) / len(triggered)) if triggered else None,
              "curriculum_reset_env_call_s": curriculum_timings[0] if curriculum_timings else None,
              "curriculum_call_s_max": max((t["seconds"] for t in curriculum_timings), default=None)}
    report["timing"] = timing
    checks.add("10_step_time", True,
               f"mean_env_step={mean_step:.4f}s (n={len(step_seconds)}, {base.num_envs} envs) "
               f"curriculum_mean={timing['curriculum_call_s_mean']} over {len(triggered)} rollout resets, "
               f"curriculum_full_reset={timing['curriculum_reset_env_call_s']}, "
               f"curriculum_max={timing['curriculum_call_s_max']}")
    report["frozen_actor_checkpoint"] = str(OLD)
    if not math.isfinite(mean_step):
        raise RuntimeError("non-finite step time")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, choices=(2048, 4096), default=NUM_ENVS,
                        help="2048 is the research plan's memory-pressure fallback")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--targeted", action="store_true", help="check the approved 30/20/50 retraining profile")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.child:
        if args.output.exists():
            raise FileExistsError(f"preserve existing evidence: {args.output}")
        sys.exit(supervise(args))
    faulthandler.enable()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    checks = Checks()
    report = {"task": TARGETED_TASK if args.targeted else TASK, "num_envs": args.num_envs, "checks": checks.results}
    env = None
    try:
        env = build_and_check(app, checks, report, args.num_envs, targeted=args.targeted)
    except Exception as exc:
        traceback.print_exc()
        checks.add("setup", False, f"exception: {type(exc).__name__}: {exc}")
    report["all_passed"] = checks.all_passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    passed = sum(r["passed"] for r in checks.results)
    print(f"SUMMARY {passed}/{len(checks.results)} PASS report={args.output}", flush=True)
    # Exit without env.close()/app.close(): both hang in this Isaac Lab build after the
    # report is written (seen twice); the driver frees GPU memory on process exit.
    os._exit(0 if checks.all_passed else 1)


if __name__ == "__main__":
    main()
