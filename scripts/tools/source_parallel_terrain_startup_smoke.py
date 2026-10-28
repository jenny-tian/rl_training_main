"""G1 startup diagnostic restricted to eight Source terrain environments.

Build the map, reset all environments and take one zero-action control step.
No policy, training, or asynchronous reset qualification is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any

from isaaclab.app import AppLauncher


def _all_finite(value: Any) -> bool:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return True


def _trace(trace_file, message: str) -> None:
    line = f"[parallel-terrain-smoke] {message}"
    print(line, file=sys.stderr, flush=True)
    trace_file.write(line + "\n")
    trace_file.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="SourceTerrain-Deeprobotics-Lite3-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=130_001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 8:
        parser.error("this G1 startup smoke is restricted to exactly eight environments")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite smoke output: {output_path}")
    trace_path = (
        args.trace_output.resolve()
        if args.trace_output is not None
        else output_path.with_suffix(output_path.suffix + ".trace.log")
    )
    if trace_path == output_path:
        parser.error("--output and --trace-output must use different paths")
    if trace_path.exists():
        parser.error(f"refusing to overwrite smoke trace: {trace_path}")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_file = trace_path.open("x", encoding="utf-8")

    simulation_app = None
    try:
        _trace(trace_file, "app-launch-start")
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        _trace(trace_file, "app-launch-complete")

        import gymnasium as gym
        import torch

        import rl_training.tasks  # noqa: F401
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_terrain_env_cfg import (
            DeeproboticsLite3SourceTerrainEnvCfg,
        )
        _trace(trace_file, "imports-complete")

        _trace(trace_file, "config-construction-start")
        env_cfg = DeeproboticsLite3SourceTerrainEnvCfg(
            source_parallel_sampler_enabled=True,
            source_parallel_sampler_num_envs=args.num_envs,
            source_parallel_sampler_seed=args.seed,
        )
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device
        _trace(trace_file, "environment-construction-start")
        construction_started = time.monotonic()
        env = gym.make(args.task, cfg=env_cfg)
        construction_seconds = time.monotonic() - construction_started
        _trace(trace_file, "environment-construction-complete")
        try:
            _trace(trace_file, "initial-reset-start")
            observation, reset_info = env.reset(seed=args.seed)
            _trace(trace_file, "initial-reset-complete")
            unwrapped = env.unwrapped
            terrain = unwrapped.scene.terrain
            allocator = unwrapped._source_parallel_terrain_allocator
            assignments = allocator.assignments
            if int(unwrapped.num_envs) != args.num_envs:
                raise AssertionError("runtime environment count differs from the explicit count")
            if tuple(terrain.terrain_origins.shape) != (
                args.num_envs,
                15,
                3,
            ):
                raise AssertionError(
                    f"unexpected terrain origin grid: {tuple(terrain.terrain_origins.shape)}"
                )
            if any(item is None for item in assignments):
                raise AssertionError("initial reset did not assign every environment")
            if [item.row for item in assignments] != list(range(args.num_envs)):
                raise AssertionError("each environment must retain its reserved terrain row")
            if not torch.isfinite(terrain.terrain_origins).all():
                raise AssertionError("terrain origins contain non-finite coordinates")

            expected_origins = terrain.terrain_origins[
                terrain.terrain_levels, terrain.terrain_types
            ]
            if not torch.equal(terrain.env_origins, expected_origins):
                raise AssertionError("initial environment origins do not match assigned tiles")
            if torch.unique(terrain.terrain_levels).numel() != args.num_envs:
                raise AssertionError("environment rows are not unique at startup")
            root = unwrapped.scene["robot"].data
            robot = unwrapped.scene["robot"]
            guard_cfg = unwrapped.termination_manager.get_term_cfg("source_terrain_tile_bounds")
            body_ids = guard_cfg.params["asset_cfg"].body_ids
            all_ids = list(range(robot.num_bodies))
            resolved_ids = all_ids[body_ids] if isinstance(body_ids, slice) else list(body_ids)
            if sorted(resolved_ids) != all_ids:
                raise AssertionError("tile guard does not cover every runtime rigid body")
            if not _all_finite((root.body_pos_w, root.body_lin_vel_w)):
                raise AssertionError("runtime body positions or velocities are non-finite")
            _trace(trace_file, f"tile-guard-body-mapping-complete: {len(resolved_ids)} bodies")
            expected_root_positions = root.default_root_state[:, :3] + terrain.env_origins
            if not torch.allclose(root.root_pos_w, expected_root_positions, atol=1e-5, rtol=0.0):
                raise AssertionError("initial robot roots do not match their assigned terrain origins")
            initial_origins = terrain.env_origins.clone()
            initial_roots = root.root_pos_w.clone()
            if torch.unique(initial_roots, dim=0).shape[0] != args.num_envs:
                raise AssertionError("duplicate robot spawn positions")
            generator = env_cfg.scene.terrain.terrain_generator
            if tuple(generator.size) != (12.0, 4.0):
                raise AssertionError("unexpected tile geometry")
            if len(generator.sub_terrains) != 15 or any(
                abs(cfg.proportion - 1.0 / 15.0) > 1e-12
                for cfg in generator.sub_terrains.values()
            ):
                raise AssertionError("expected fifteen equal terrain columns")
            _trace(trace_file, "startup-origin-assertions-complete")

            action_dim = int(unwrapped.action_manager.total_action_dim)
            zero_action = torch.zeros(
                (args.num_envs, action_dim), device=unwrapped.device, dtype=torch.float32
            )
            _trace(trace_file, "first-control-step-start")
            next_observation, reward, terminated, truncated, step_info = env.step(zero_action)
            _trace(trace_file, "first-control-step-complete")
            termination_flags = {
                name: unwrapped.termination_manager.get_term(name).cpu().tolist()
                for name in unwrapped.termination_manager.active_terms
            }
            _trace(trace_file, "termination-flags: " + json.dumps(termination_flags))
            if bool(terminated.any().item()) or bool(truncated.any().item()):
                raise AssertionError("an environment terminated during the first zero-action step")
            if not _all_finite((observation, next_observation, reward, terminated, truncated)):
                raise AssertionError("non-finite value appeared during the first physics step")

            if not _all_finite((root.body_pos_w, root.body_lin_vel_w)):
                raise AssertionError("non-finite body state after the first physics step")
            sensor_shapes = {}
            for sensor_name in ("height_scanner", "height_scanner_base"):
                ray_hits = unwrapped.scene.sensors[sensor_name].data.ray_hits_w
                sensor_shapes[sensor_name] = list(ray_hits.shape)
                if not torch.isfinite(ray_hits).all():
                    raise AssertionError(f"{sensor_name} produced non-finite ray hits")
            _trace(trace_file, "height-scanner-assertions-complete")

            from pxr import Usd, UsdGeom

            terrain_mesh_triangles = {}
            for prim_path in terrain.terrain_prim_paths:
                for prim in Usd.PrimRange(unwrapped.sim.stage.GetPrimAtPath(prim_path)):
                    if prim.IsA(UsdGeom.Mesh):
                        counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
                        terrain_mesh_triangles[str(prim.GetPath())] = sum(
                            max(0, int(count) - 2) for count in counts
                        )

            report = {
                "schema_version": "source_parallel_terrain_startup_smoke_v1",
                "status": "diagnostic_smoke_passed",
                "task": args.task,
                "seed": args.seed,
                "device": str(unwrapped.device),
                "num_envs": int(unwrapped.num_envs),
                "terrain_origin_shape": list(terrain.terrain_origins.shape),
                "initial_assignments": [
                    {
                        "env_id": item.env_id,
                        "row": item.row,
                        "column": item.column,
                        "subterrain": item.subterrain,
                        "assignment_index": item.assignment_index,
                    }
                    for item in assignments
                ],
                "action_dim": action_dim,
                "initial_origins_w": initial_origins.cpu().tolist(),
                "initial_robot_roots_w": initial_roots.cpu().tolist(),
                "terrain_origin_grid_w": terrain.terrain_origins.cpu().tolist(),
                "terrain_column_names": list(generator.sub_terrains),
                "terrain_column_proportions": [cfg.proportion for cfg in generator.sub_terrains.values()],
                "sensor_ray_hit_shapes": sensor_shapes,
                "environment_construction_seconds": construction_seconds,
                "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "terrain_mesh_triangles": terrain_mesh_triangles,
                "terrain_generator_seed": env_cfg.scene.terrain.terrain_generator.seed,
                "terrain_tile_size": list(env_cfg.scene.terrain.terrain_generator.size),
                "tile_guard_runtime_body_ids": resolved_ids,
                "tile_guard_runtime_body_names": list(robot.body_names),
                "tile_guard_covers_all_runtime_bodies": True,
                "termination_flags_after_first_step": termination_flags,
                "zero_action_steps": 1,
                "reset_info_keys": sorted(reset_info),
                "step_info_keys": sorted(step_info),
                "startup_origins_match_rows_and_columns": True,
                "robot_roots_match_assigned_origins": True,
                "height_scanners_finite": True,
                "asynchronous_reset_tested": False,
                "policy_loaded": False,
                "ppo_updates": 0,
                "qualification_claim": False,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _trace(trace_file, "report-write-complete")
            print(json.dumps({"output": str(output_path), "status": report["status"]}), flush=True)
            return 0
        finally:
            _trace(trace_file, "environment-close-start")
            env.close()
            _trace(trace_file, "environment-close-complete")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        _trace(trace_file, "fatal-exception-recorded")
        raise
    finally:
        try:
            if simulation_app is not None:
                _trace(trace_file, "simulation-app-close-start")
                try:
                    simulation_app.close()
                except BaseException:
                    traceback.print_exc(file=sys.stderr)
                    _trace(trace_file, "simulation-app-close-exception")
                    raise
                _trace(trace_file, "simulation-app-close-complete")
        finally:
            trace_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
