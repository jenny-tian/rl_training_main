"""Audit actual USD colliders at reset; never execute a control transition."""
import argparse
import itertools
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    failed = False
    try:
        import gymnasium as gym
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.math import matrix_from_quat
        import rl_training.tasks  # noqa: F401
        from rl_training.online_adaptation.runner_contract import configure_initial_state
        from rl_training.online_adaptation.spawn_geometry import certify_spawn_range
        from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic

        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        cfg = parse_env_cfg("RockyRoad-Deeprobotics-Lite3-v0", device=args.device, num_envs=1)
        configure_initial_state(cfg, "fixed")
        cfg.seed = 7
        cfg.scene.num_envs = 1
        env = gym.make("RockyRoad-Deeprobotics-Lite3-v0", cfg=cfg).unwrapped
        try:
            robot = env.scene["robot"]
            # Read collider-to-body transforms from the composed asset, then use
            # actual PhysX body transforms. Fabric does not update USD world poses.
            asset_path = Path(cfg.scene.robot.spawn.usd_path).resolve()
            stage = Usd.Stage.Open(str(asset_path))
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            descriptors = []
            for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                    continue
                body = prim.GetParent()
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                if not body or body.GetName() not in robot.body_names:
                    raise ValueError(f"collision has no matching simulated body: {prim.GetPath()}")
                matrix, reset_stack = cache.ComputeRelativeTransform(prim, body)
                if reset_stack:
                    raise ValueError("reset xform stack prevents rigid-local proof")
                record = {"path": str(prim.GetPath()), "body": body.GetName(), "type": prim.GetTypeName(),
                          "body_from_shape": np.asarray(matrix, dtype=float).T.tolist()}
                for field in ("radius", "height", "size", "axis"):
                    attr = prim.GetAttribute(field)
                    if attr:
                        record[field] = attr.Get()
                descriptors.append(record)
            if not descriptors:
                raise ValueError("no deployed colliders discovered")
            source_hashes = {str(p.relative_to(asset_path.parent)): sha256_file(p)
                             for p in sorted(asset_path.parent.rglob("*.usd"))}
            report = {"status": "running", "control_transitions": 0, "asset_sha256": source_hashes,
                      "colliders": descriptors, "cases": []}
            write_json_atomic(output / "audit.json", report)
            cases = [(0., 0., 0.), *itertools.product((-.02, .02), (-.02, .02), (-math.radians(2), math.radians(2)))]
            for index, (x, y, yaw) in enumerate(cases):
                term = env.event_manager.get_term_cfg("randomize_reset_base")
                term.params["pose_range"].update(x=(x, x), y=(y, y), yaw=(yaw, yaw))
                env.event_manager.set_term_cfg("randomize_reset_base", term)
                env.reset(seed=7)
                shapes = []
                rotations = matrix_from_quat(robot.data.body_quat_w[0]).cpu().numpy()
                positions = robot.data.body_pos_w[0].cpu().numpy()
                for descriptor in descriptors:
                    body_id = robot.body_names.index(descriptor["body"])
                    world_from_body = np.eye(4)
                    world_from_body[:3, :3] = rotations[body_id]
                    world_from_body[:3, 3] = positions[body_id]
                    shapes.append({**descriptor, "world_from_shape": (world_from_body @ np.array(descriptor["body_from_shape"])).tolist()})
                root = robot.data.root_pos_w[0].cpu().numpy()
                terrain = cfg.scene.terrain.terrain_generator
                sub = terrain.sub_terrains["rocky_road"]
                origin = env.scene.env_origins[0].cpu().numpy()
                platform = [[origin[0] - sub.flat_length / 2, origin[0] + sub.flat_length / 2],
                            [origin[1] - terrain.size[1] / 2, origin[1] + terrain.size[1] / 2]]
                # Only the centered case certifies the full perturbation range.
                proof = certify_spawn_range(shapes, root_position=root, platform_bounds=platform,
                                            floor_z=float(origin[2]), xy_half_range=.02 if index == 0 else 0.)
                report["cases"].append({"offset": [x, y, yaw], "root_position": root.tolist(),
                                        "joint_positions": robot.data.joint_pos[0].cpu().tolist(),
                                        "shapes": shapes, "proof": proof})
                write_json_atomic(output / "audit.json", report)
            report["status"] = "passed" if all(case["proof"]["valid"] for case in report["cases"]) else "failed"
            report["platform_bounds"] = platform
            write_json_atomic(output / "audit.json", report)
            if report["status"] != "passed":
                raise ValueError("spawn geometry failed; keep every case and fix the protocol")
        finally:
            env.close()
    except BaseException:
        failed = True
        import traceback
        error = traceback.format_exc()
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "failure.txt").write_text(error)
        print(error, flush=True)
        raise
    finally:
        try:
            launcher.app.close()
        finally:
            if failed:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
