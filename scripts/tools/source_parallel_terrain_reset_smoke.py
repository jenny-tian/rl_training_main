"""Smoke-test the opt-in parallel Source terrain importer and asynchronous reset.

This diagnostic builds exactly two environments, takes one zero-action control
step, then directly resets one environment. It loads no policy and performs no
training or capability qualification.
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
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=130_001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 2:
        parser.error("the importer/reset smoke is restricted to exactly two environments")
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

            for sensor_name in ("height_scanner", "height_scanner_base"):
                ray_hits = unwrapped.scene.sensors[sensor_name].data.ray_hits_w
                if not torch.isfinite(ray_hits).all():
                    raise AssertionError(f"{sensor_name} produced non-finite ray hits")
            _trace(trace_file, "height-scanner-assertions-complete")

            before_assignment = allocator.assignments[0]
            before_assignments = allocator.assignments
            untouched_assignment = before_assignments[1]
            untouched_origin = terrain.env_origins[1].clone()
            untouched_root = root.root_pos_w[1].clone()
            _trace(trace_file, "single-environment-reset-start")
            unwrapped._reset_idx(torch.tensor([0], device=unwrapped.device, dtype=torch.long))
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            unwrapped.scene.update(0.0)
            _trace(trace_file, "single-environment-reset-complete")

            after_assignments = unwrapped._source_parallel_terrain_allocator.assignments
            after_assignment = after_assignments[0]
            if after_assignment.env_id != 0 or after_assignment.row != 0:
                raise AssertionError("asynchronous reset changed environment 0's owner row")
            expected_assignment_index = max(
                item.assignment_index for item in before_assignments
            ) + 1
            if after_assignment.assignment_index != expected_assignment_index:
                raise AssertionError("asynchronous reset did not consume exactly one assignment")
            if after_assignments[1] != untouched_assignment:
                raise AssertionError("asynchronous reset changed environment 1's assignment")
            if not torch.equal(terrain.env_origins[1], untouched_origin):
                raise AssertionError("asynchronous reset changed environment 1's terrain origin")
            if not torch.equal(root.root_pos_w[1], untouched_root):
                raise AssertionError("asynchronous reset moved environment 1's robot")
            if not torch.equal(
                terrain.env_origins[0],
                terrain.terrain_origins[terrain.terrain_levels[0], terrain.terrain_types[0]],
            ):
                raise AssertionError("reset environment origin does not match its new tile")
            expected_root = root.default_root_state[0, :3] + terrain.env_origins[0]
            if not torch.allclose(root.root_pos_w[0], expected_root, atol=1e-5, rtol=0.0):
                raise AssertionError("reset robot root does not use the newly assigned origin")
            _trace(trace_file, "async-reset-assertions-complete")

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
                "schema_version": "source_parallel_terrain_reset_smoke_v1",
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
                "async_reset_assignment": {
                    "env_id": after_assignment.env_id,
                    "row": after_assignment.row,
                    "column": after_assignment.column,
                    "subterrain": after_assignment.subterrain,
                    "assignment_index": after_assignment.assignment_index,
                },
                "action_dim": action_dim,
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
                "asynchronous_reset_preserved_other_environment": True,
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
