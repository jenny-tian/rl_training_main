"""Build all six Lite3 Source Terrain families and verify one PhysX step.

This is an integration smoke test only.  It does not load a policy, score an
episode, run a capability scan, or update a model.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from isaaclab.app import AppLauncher


def _tree_is_finite(value: Any) -> bool:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_tree_is_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tree_is_finite(item) for item in value)
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return bool(np.isfinite(value).all())
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="SourceTerrain-Deeprobotics-Lite3-v0")
    parser.add_argument("--seed", type=int, default=10_101)
    parser.add_argument("--num-envs", type=int, default=6)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 6:
        parser.error("the bridge smoke uses exactly six envs so each Source family is populated")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[source-terrain-smoke] Isaac Sim started", flush=True)
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from pxr import UsdGeom, UsdPhysics, UsdShade
        from isaaclab.sensors.ray_caster import MultiMeshRayCaster
        from isaaclab.sim.utils.stage import get_current_stage
        from isaaclab_tasks.utils import parse_env_cfg

        import rl_training.tasks  # noqa: F401
        from rl_training.tasks.manager_based.locomotion.velocity.terrains import (
            SourceTerrainTrackCfg,
            source_terrain_track,
        )
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_terrain_env_cfg import (
            SOURCE_TERRAIN_PROPORTIONS,
            _source_sub_terrains,
        )

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        print("[source-terrain-smoke] task configuration loaded", flush=True)
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = args.num_envs
        generator_cfg = env_cfg.scene.terrain.terrain_generator
        production_mix = _source_sub_terrains("L1")
        source_family_probabilities = {
            family: sum(cfg.proportion for cfg in production_mix.values() if cfg.family == family)
            for family in SOURCE_TERRAIN_PROPORTIONS
        }
        for family, expected in SOURCE_TERRAIN_PROPORTIONS.items():
            if abs(source_family_probabilities[family] - expected) > 1e-12:
                raise AssertionError(f"Source Terrain Pilot probability mismatch for {family}")
        geometry_variants_checked = 0
        for level in ("L1", "L2", "L3"):
            for cfg in _source_sub_terrains(level).values():
                cfg.size = (10.0, 4.0)
                meshes, _ = source_terrain_track(0.0, cfg)
                if not meshes or any(
                    not np.isfinite(mesh.vertices).all() or not mesh.is_watertight for mesh in meshes
                ):
                    raise AssertionError(f"invalid {level} Source Terrain geometry for {cfg.family}")
                geometry_variants_checked += 1
        print(
            f"[source-terrain-smoke] checked {geometry_variants_checked} L1-L3 geometry variants",
            flush=True,
        )
        generator_cfg.seed = args.seed
        generator_cfg.num_rows = 1
        generator_cfg.num_cols = 6
        generator_cfg.curriculum = True
        generator_cfg.difficulty_range = (0.0, 0.0)
        generator_cfg.use_cache = False
        generator_cfg.sub_terrains = {
            "flat": SourceTerrainTrackCfg(proportion=1 / 6, family="flat"),
            "multi_scale_rough": SourceTerrainTrackCfg(
                proportion=1 / 6, family="multi_scale_rough", rough_peak_valley_m=0.02
            ),
            "slope": SourceTerrainTrackCfg(
                proportion=1 / 6, family="slope", slope_angle_deg=5.0, direction="uphill"
            ),
            "multi_step_stairs": SourceTerrainTrackCfg(
                proportion=1 / 6,
                family="multi_step_stairs",
                direction="ascending",
                step_count=3,
                step_height_m=0.02,
                tread_depth_m=0.5,
            ),
            "slope_plus_rough": SourceTerrainTrackCfg(
                proportion=1 / 6,
                family="slope_plus_rough",
                slope_angle_deg=5.0,
                direction="uphill",
                rough_peak_valley_m=0.01,
            ),
            "friction_variation": SourceTerrainTrackCfg(
                proportion=1 / 6,
                family="friction_variation",
                friction_range=(0.7, 1.0),
                friction_layout="left_right_bands",
                friction_region_count=4,
            ),
        }

        print("[source-terrain-smoke] creating scene", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        print("[source-terrain-smoke] scene created", flush=True)
        try:
            observation, reset_info = env.reset(seed=args.seed)
            print("[source-terrain-smoke] reset complete", flush=True)
            unwrapped = env.unwrapped
            print("[source-terrain-smoke] unwrapped env ready", flush=True)
            terrain = unwrapped.scene.terrain
            print(f"[source-terrain-smoke] terrain importer={type(terrain).__name__}", flush=True)
            collider_paths = list(terrain.terrain_prim_paths)
            print(f"[source-terrain-smoke] colliders={len(collider_paths)}", flush=True)
            stage = get_current_stage()
            print("[source-terrain-smoke] USD stage ready", flush=True)
            friction_paths = [path for path in collider_paths if "/friction_" in path]
            if len(friction_paths) != 4:
                raise AssertionError(f"expected four friction material colliders, got {friction_paths}")

            terrain_root = f"{env_cfg.scene.terrain.prim_path}/terrain"
            terrain_mesh_paths = sorted(
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith(f"{terrain_root}/")
            )
            expected_mesh_paths = {f"{path}/mesh" for path in collider_paths}
            if set(terrain_mesh_paths) != expected_mesh_paths:
                raise AssertionError(
                    "terrain mesh tree differs from importer colliders: "
                    f"meshes={terrain_mesh_paths}, colliders={sorted(expected_mesh_paths)}"
                )

            bound_materials: list[dict[str, Any]] = []
            for collider_path in friction_paths:
                mesh_prim = stage.GetPrimAtPath(f"{collider_path}/mesh")
                if not mesh_prim.IsValid() or not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise AssertionError(f"friction collider missing PhysX collision API: {collider_path}")
                binding = UsdShade.MaterialBindingAPI(mesh_prim).GetDirectBinding("physics")
                material_path = binding.GetMaterialPath()
                print(f"[source-terrain-smoke] bound {collider_path} -> {material_path}", flush=True)
                if not material_path or not stage.GetPrimAtPath(material_path).IsValid():
                    raise AssertionError(f"friction collider has no direct physics material: {collider_path}")
                material_prim = stage.GetPrimAtPath(material_path)
                static_friction = float(material_prim.GetAttribute("physics:staticFriction").Get())
                dynamic_friction = float(material_prim.GetAttribute("physics:dynamicFriction").Get())
                if abs(static_friction - dynamic_friction) > 1e-9:
                    raise AssertionError(f"static/dynamic friction differ on {collider_path}")
                bound_materials.append(
                    {
                        "collider": collider_path,
                        "material": str(material_path),
                        "static_friction_candidate": static_friction,
                        "dynamic_friction_candidate": dynamic_friction,
                    }
                )
            material_values = sorted(item["static_friction_candidate"] for item in bound_materials)
            if any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(material_values, [0.7, 0.8, 0.9, 1.0], strict=True)
            ):
                raise AssertionError(f"unexpected friction material palette: {material_values}")
            print(f"[source-terrain-smoke] material palette={material_values}", flush=True)

            action_dim = int(unwrapped.action_manager.total_action_dim)
            action = torch.zeros(
                (int(unwrapped.num_envs), action_dim),
                device=unwrapped.device,
                dtype=torch.float32,
            )
            print("[source-terrain-smoke] stepping simulator", flush=True)
            next_observation, reward, terminated, truncated, step_info = env.step(action)
            print("[source-terrain-smoke] one simulation step complete", flush=True)
            if not all(
                _tree_is_finite(value)
                for value in (observation, next_observation, reward, terminated, truncated)
            ):
                raise AssertionError("non-finite value found during Source Terrain reset/step")

            height_scanner_mesh_count = None
            for sensor_name in ("height_scanner", "height_scanner_base"):
                scanner = unwrapped.scene.sensors[sensor_name]
                if not isinstance(scanner, MultiMeshRayCaster):
                    raise AssertionError(f"{sensor_name} is not a MultiMeshRayCaster")
                targets = scanner._raycast_targets_cfg
                if len(targets) != 1 or targets[0].prim_expr != "/World/ground":
                    raise AssertionError(
                        f"{sensor_name} does not target the complete Source terrain root: "
                        f"{[target.prim_expr for target in targets]}"
                    )
                if not targets[0].merge_prim_meshes or targets[0].track_mesh_transforms:
                    raise AssertionError(f"{sensor_name} has unexpected static terrain mesh options")
                scanner_mesh_count = scanner._num_meshes_per_env.get("/World/ground")
                if scanner_mesh_count != 1:
                    raise AssertionError(
                        f"{sensor_name} did not initialize its merged terrain mesh: {scanner_mesh_count}"
                    )
                ray_hits = scanner.data.ray_hits_w
                if not torch.isfinite(ray_hits).all():
                    raise AssertionError(f"{sensor_name} produced a non-finite terrain height scan")
                height_scanner_mesh_count = len(terrain_mesh_paths)
            print(
                f"[source-terrain-smoke] both height scanners initialized over "
                f"{height_scanner_mesh_count} ordinary/friction terrain meshes",
                flush=True,
            )

            print(
                json.dumps(
                    {
                        "task": args.task,
                        "seed": args.seed,
                        "num_envs": int(unwrapped.num_envs),
                        "device": str(unwrapped.device),
                        "families": sorted(generator_cfg.sub_terrains),
                        "production_mix_variant_count": len(production_mix),
                        "production_mix_family_probabilities": source_family_probabilities,
                        "l1_l3_geometry_variants_checked": geometry_variants_checked,
                        "terrain_origins_shape": list(terrain.terrain_origins.shape),
                        "terrain_collider_count": len(collider_paths),
                        "terrain_mesh_paths_for_height_scan": terrain_mesh_paths,
                        "height_scanner_class": "MultiMeshRayCaster",
                        "height_scanner_target_mesh_count": height_scanner_mesh_count,
                        "friction_material_colliders": bound_materials,
                        "action_dim": action_dim,
                        "reward": reward.detach().cpu().tolist(),
                        "terminated": terminated.detach().cpu().tolist(),
                        "truncated": truncated.detach().cpu().tolist(),
                        "reset_info_keys": sorted(reset_info),
                        "step_info_keys": sorted(step_info),
                        "finite_reset_and_step": True,
                        "model_loaded": False,
                        "episode_evaluated": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        finally:
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
