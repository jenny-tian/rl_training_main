"""Reference Lite3 with six terrain families and explicitly bounded updates.

Reward overrides are opt-in with --reward-variant and audited separately.
Default initialization is from scratch. --resume-final loads the pinned final
checkpoint for reward experiments; reference rewards cannot be continued.

smoke: 120 environments / 100 steps, no optimization.
validate: 4096 environments / 3 PPO updates, then stop.
train: an explicitly selected budget, then stop for the next phase report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import ROOT, REFERENCE, load_reference, source_manifest, resolve_lite3_asset


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def plain(cfg):
    return json.loads(json.dumps(cfg.to_dict(), default=str))


def configuration_audit(reference, candidate, agent):
    before, after = plain(reference), plain(candidate)
    original_terrain = before["scene"].pop("terrain")
    new_terrain = after["scene"].pop("terrain")
    if before != after:
        raise ValueError("a non-terrain environment setting changed")
    for key in original_terrain:
        if key not in ("class_type", "terrain_generator") and original_terrain[key] != new_terrain[key]:
            raise ValueError(f"reference terrain importer setting changed: {key}")
    original_generator = dict(original_terrain["terrain_generator"])
    new_generator = dict(new_terrain["terrain_generator"])
    original_generator.pop("sub_terrains")
    new_generator.pop("sub_terrains")
    if original_generator != new_generator:
        raise ValueError("native terrain layout or curriculum configuration changed")
    if agent.resume or agent.algorithm.learning_rate != .001 or agent.max_iterations != 10000:
        raise ValueError("reference PPO defaults changed")
    return dict(status="passed", non_terrain_environment_exact=True, native_layout_exact=True,
                reference_environment=plain(reference), candidate_environment=plain(candidate),
                reference_agent=plain(agent), changed=["scene.terrain.class_type", "scene.terrain.terrain_generator.sub_terrains"],
                added_importer_options=dict(parallel_reset_sampler_enabled=False, parallel_reset_sampler_num_envs=None,
                                            shared_bank_rows=None, source_curriculum_allocation=None))


def audit_simulation(base, cfg, output, make_plot):
    import numpy as np
    import torch
    import trimesh
    from isaaclab.sensors import RayCaster
    from isaaclab.terrains import TerrainImporter
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics, UsdShade
    from reference_six_terrain_cfg import FAMILIES, column_names

    terrain = base.scene.terrain
    if type(terrain).update_env_origins is not TerrainImporter.update_env_origins:
        raise ValueError("terrain progression is not the native implementation")
    if terrain._curriculum_allocation is not None or terrain._parallel_reset_sampler_enabled:
        raise ValueError("an old terrain allocation mode is enabled")
    names = column_names(cfg)
    counts = torch.bincount(terrain.terrain_types, minlength=20).cpu().tolist()
    expected = torch.div(torch.arange(base.num_envs, device=base.device), base.num_envs / 20,
                         rounding_mode="floor").long()
    if not torch.equal(terrain.terrain_types, expected):
        raise ValueError("environment allocation is not the reference allocation")
    dims = {k: list(v) for k, v in base.observation_manager.group_obs_dim.items()}
    if dims != {"policy": [45], "critic": [393]}:
        raise ValueError(f"unexpected observations: {dims}")
    family_counts = {}
    for name, count in zip(names, counts):
        family_counts[FAMILIES[name]] = family_counts.get(FAMILIES[name], 0) + count

    # Audit the authored collision geometry and material bindings, separately
    # from the full mesh supplied to the reference height scanners.
    meshes, materials = [], []
    for path in terrain.terrain_prim_paths:
        prim = base.scene.stage.GetPrimAtPath(path + "/mesh")
        usd_mesh = UsdGeom.Mesh(prim)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(f"missing collision API: {path}")
        points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        meshes.append(trimesh.Trimesh(vertices=points, faces=faces, process=False))
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        materials.append(dict(path=path, material=str(material.GetPath()),
                              static_friction=float(physics.GetStaticFrictionAttr().Get()),
                              dynamic_friction=float(physics.GetDynamicFrictionAttr().Get()), faces=len(faces)))
    friction = sorted({round(v["static_friction"], 6) for v in materials if "/friction_" in v["path"]})
    if friction != [.5, .666667, .833333, 1.]:
        raise ValueError(f"missing friction materials: {friction}")
    collider_mesh = trimesh.util.concatenate(meshes)
    if len(collider_mesh.faces) != terrain.reference_ray_mesh_triangles:
        raise ValueError("colliders and height scanner have different triangle counts")
    collision_warp = convert_to_warp_mesh(collider_mesh.vertices, collider_mesh.faces, device=base.device)
    origins = terrain.terrain_origins.reshape(-1, 3)
    offsets = torch.tensor([[x, y, 20.] for x in (-3.17, -1.13, 0.03, 1.19, 3.11)
                            for y in (-3.17, -1.13, 0.03, 1.19, 3.11)], device=base.device)
    starts = (origins[:, None] + offsets).reshape(-1, 3)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1
    full_mesh = RayCaster.meshes[cfg.scene.terrain.prim_path]
    hits = raycast_mesh(starts, directions, full_mesh)[0]
    collider_hits = raycast_mesh(starts, directions, collision_warp)[0]
    if not torch.isfinite(hits).all() or not torch.isfinite(collider_hits).all():
        raise ValueError("a terrain tile is missing from the height scanner or colliders")
    error = (hits - collider_hits).abs().max().item()
    if error > 1e-4:
        raise ValueError(f"scanner/collider height mismatch: {error}")
    scanner_checks = {}
    for key in ("height_scanner", "height_scanner_base"):
        scanner = base.scene[key]
        data = scanner.data.ray_hits_w
        if not torch.isfinite(data).all():
            raise ValueError(f"nonfinite initial rays: {key}")
        scanner_checks[key] = dict(shape=list(data.shape), finite=True)
    report = dict(status="passed", observation_dimensions=dims, column_names=names,
                  environment_column_counts=counts, family_counts=family_counts,
                  initial_terrain_level_histogram=torch.bincount(terrain.terrain_levels, minlength=10).cpu().tolist(),
                  native_allocation=True, native_curriculum=True, collision_mesh_count=len(meshes),
                  collision_triangles=len(collider_mesh.faces), ray_probes=len(starts),
                  ray_probe_tiles=200, scanner_collider_max_error_m=error,
                  scanners=scanner_checks, materials=materials, friction_coefficients=friction,
                  asset_root_path=str(cfg.scene.robot.spawn.usd_path))
    save(output / "simulation_audit.json", report)
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        selected = ["flat", "rough", "slope_up", "stairs_up", "stairs_down", "friction_lateral", "slope_rough_up"]
        grid = torch.linspace(-3.9, 3.9, 80, device=base.device)
        xx, yy = torch.meshgrid(grid, grid, indexing="ij")
        fig = plt.figure(figsize=(18, 9))
        for index, name in enumerate(selected):
            col = names.index(name)
            origin = terrain.terrain_origins[-1, col]
            points = origin + torch.stack((xx, yy, torch.full_like(xx, 20.)), -1)
            direction = torch.zeros_like(points)
            direction[..., 2] = -1
            zz = raycast_mesh(points.reshape(-1, 3), direction.reshape(-1, 3), full_mesh)[0][:, 2].reshape(xx.shape)
            ax = fig.add_subplot(2, 4, index + 1, projection="3d")
            ax.plot_surface(xx.cpu().numpy(), yy.cpu().numpy(), (zz - origin[2]).cpu().numpy(), cmap="terrain",
                            rcount=80, ccount=80, linewidth=0)
            ax.set_title(name)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("height (m)")
        fig.suptitle("Actual generated terrain: last curriculum row, ray-sampled geometry (not trained behavior)")
        fig.tight_layout()
        fig.savefig(output / "terrain_geometry.png", dpi=150)
        plt.close(fig)
    return report


def worker(args):
    from isaaclab.app import AppLauncher
    launcher = dict(headless=True, device="cuda:0")
    if args.terrain_profile == 'low_stairs':
        launcher['kit_args'] = f'--/renderer/multiGpu/enabled=False --/renderer/activeGpu={args.gpu}'
    app = AppLauncher(**launcher).app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    from reference_reward_resume import FINAL_CHECKPOINT, restore_final, equal_state
    reference, agent = load_reference()
    from reference_six_terrain_cfg import replace_terrain
    cfg = replace_terrain(reference)
    save(args.output / "configuration_audit.json", configuration_audit(reference, cfg, agent))
    from reference_reward_variants import apply_reward_variant
    if args.terrain_profile == 'low_stairs':
        from reference_low_stairs_cfg import apply_terrain, apply_rewards
        cfg, terrain_audit = apply_terrain(cfg)
        save(args.output / 'low_stairs_terrain_audit.json', terrain_audit)
        cfg, reward_audit = apply_rewards(cfg)
        reward_audit['low_stairs_sources'] = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('reference_low_stairs_cfg.py','reference_low_stairs_rewards.py')}
    else:
        cfg, reward_audit = apply_reward_variant(cfg, args.reward_variant)
    reward_audit['source_sha256'] = hashlib.sha256(
        Path(__file__).with_name('reference_reward_variants.py').read_bytes()).hexdigest()
    save(args.output / "reward_override_audit.json", reward_audit)
    asset_resolution = resolve_lite3_asset(cfg)
    save(args.output / "asset_resolution.json", asset_resolution)
    cfg.seed = agent.seed = args.seed
    cfg.scene.num_envs = 120 if args.mode == "smoke" else 4096
    cfg.sim.device = agent.device = "cuda:0"
    agent.max_iterations = args.updates
    agent.logger = "tensorboard"
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
    save(args.output / "runtime_overrides.json", dict(seed=args.seed, num_envs=cfg.scene.num_envs,
         device=agent.device, max_iterations=args.updates, mode=args.mode,
         checkpoint=str(FINAL_CHECKPOINT) if args.resume_final else None,
         resume=args.resume_final, reference_budget=10000, automatic_extension=False,
         reward_variant=args.reward_variant, terrain_profile=args.terrain_profile))
    dump_yaml(str(args.output / "params/env.yaml"), cfg)
    dump_yaml(str(args.output / "params/agent.yaml"), agent)
    base = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
    audit_simulation(base, cfg, args.output, make_plot=args.mode == "smoke")
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(args.output / "training"), device="cuda:0")
    runner.add_git_repo_to_log(__file__)
    runner.add_git_repo_to_log(str(REFERENCE / "scripts/reinforcement_learning/rsl_rl/train.py"))
    initial = copy.deepcopy(runner.alg.save())
    if initial["optimizer_state_dict"]["state"] or runner.current_learning_iteration != 0:
        raise ValueError("optimizer or iteration did not start from scratch")
    if initial["actor_state_dict"]["mlp.0.weight"].shape != (512, 45):
        raise ValueError("actor structure differs from reference")
    if initial["critic_state_dict"]["mlp.0.weight"].shape != (512, 393):
        raise ValueError("critic structure differs from reference")
    restore = restore_final(runner) if args.resume_final else None
    if restore:
        save(args.output / "checkpoint_restore_audit.json", restore)
        initial = copy.deepcopy(runner.alg.save())
    initial_update_count = 10000 if restore else 0
    initial_optimizer_steps = 200000 if restore else 0
    report = dict(status="running", mode=args.mode, seed=args.seed, from_scratch=not args.resume_final,
                  checkpoint_loaded=str(FINAL_CHECKPOINT) if restore else None,
                  reward_variant=args.reward_variant,
                  terrain_profile=args.terrain_profile,
                  initial_optimizer_state_entries=len(initial["optimizer_state_dict"]["state"]),
                  initial_iteration=runner.current_learning_iteration, initial_learning_rate=runner.alg.learning_rate,
                  initial_model_hashes={k: hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in initial[k].values())).hexdigest()
                                       for k in ("actor_state_dict", "critic_state_dict")},
                  completed_updates=0, environment_steps=0, resets=0, terminated=0, truncated=0,
                  updates=[], automatic_extension=False)
    save(args.output / "report.json", report)
    original_step, original_log = env.step, runner.logger.log
    step_budget = 100 if args.mode == "smoke" else args.updates * agent.num_steps_per_env

    def checked_step(actions):
        if report["environment_steps"] >= step_budget:
            raise RuntimeError("interaction budget exceeded")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("training actions are not finite")
        observations, rewards, dones, extras = original_step(actions)
        if not torch.isfinite(rewards).all() or not all(torch.isfinite(v).all() for v in observations.values()):
            raise FloatingPointError("training observations or rewards are not finite")
        # Clipped height observations could hide missed rays; inspect raw rays.
        for key in ("height_scanner", "height_scanner_base"):
            if not torch.isfinite(base.scene[key].data.ray_hits_w).all():
                raise FloatingPointError(f"height rays are not finite: {key}")
        report["environment_steps"] += 1
        report["resets"] += int(dones.sum())
        report["terminated"] += int(base.reset_terminated.sum())
        report["truncated"] += int(base.reset_time_outs.sum())
        return observations, rewards, dones, extras

    def checked_log(**kwargs):
        losses = {k: float(v) for k, v in kwargs["loss_dict"].items()}
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError("PPO loss is not finite")
        report["completed_updates"] += 1
        report["updates"].append(dict(update=report["completed_updates"], losses=losses,
            learning_rate=runner.alg.learning_rate, collect_s=kwargs["collect_time"], learn_s=kwargs["learn_time"],
            mean_terrain_level=float(base.scene.terrain.terrain_levels.float().mean())))
        original_log(**kwargs)
        save(args.output / "report.json", report)
        print(f"REFERENCE SIX {args.mode}: {report['completed_updates']}/{args.updates}", flush=True)

    env.step, runner.logger.log = checked_step, checked_log
    if args.mode == "smoke":
        with torch.inference_mode():
            for _ in range(step_budget):
                env.step(torch.zeros(base.num_envs, 12, device=base.device))
        if not equal_state(initial,runner.alg.save()):
            raise ValueError("smoke changed model or optimizer state")
    else:
        runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
        current = runner.alg.save()
        for key in ("actor_state_dict", "critic_state_dict"):
            if not all(torch.isfinite(v).all() for v in current[key].values()):
                raise FloatingPointError("model parameters are not finite")
            if not any(not torch.equal(initial[key][k], v) for k, v in current[key].items()):
                raise ValueError(f"model did not update: {key}")
        if {int(v["step"]) for v in current["optimizer_state_dict"]["state"].values()} != {initial_optimizer_steps + args.updates * 20}:
            raise ValueError("unexpected Adam step count")
        if report["completed_updates"] != args.updates:
            raise ValueError("incomplete PPO budget")
        total_updates = initial_update_count + args.updates
        checkpoint = args.output / f"model_updates_{total_updates:05d}.pt"
        runner.save(str(checkpoint), infos=dict(from_scratch=not args.resume_final, completed_updates=total_updates,
                                              phase_updates=args.updates, reward_variant=args.reward_variant,
                                              terrain_profile=args.terrain_profile,
                                              validation_only=args.mode != "train"))
        report["output_checkpoint"] = str(checkpoint)
        report["output_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if report["environment_steps"] != step_budget:
        raise ValueError("incomplete environment step budget")
    report.update(status="passed", interactions=step_budget * base.num_envs,
                  cumulative_updates=initial_update_count + report['completed_updates'],
                  final_learning_rate=runner.alg.learning_rate, sources=source_manifest(),
                  final_terrain_level_histogram=torch.bincount(base.scene.terrain.terrain_levels, minlength=10).cpu().tolist())
    save(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "updates")}), flush=True)
    # Explicit parent process bounds teardown as for the preceding simulation jobs.
    sys.stdout.flush()
    os._exit(0)


def main():
    from reference_reward_variants import VARIANTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "validate", "train"), required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-variant", choices=(*VARIANTS, 'low_stairs'), default="reference")
    parser.add_argument("--terrain-profile", choices=('reference','low_stairs'), default='reference')
    parser.add_argument("--resume-final", action="store_true",
                        help="Restore reviewed final10000, including optimizer; at most 500 additional updates")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if (args.terrain_profile == 'low_stairs') != (args.reward_variant == 'low_stairs'):
        parser.error('low_stairs requires matching terrain and reward profiles')
    if args.terrain_profile == 'low_stairs' and args.resume_final:
        parser.error('low_stairs must start from scratch; old checkpoints saw 5-23 cm stairs')
    if args.terrain_profile == 'low_stairs' and args.mode == 'train' and args.updates is None:
        parser.error('low_stairs training requires an explicit update budget')
    if args.resume_final and (args.reward_variant == 'reference' or args.mode == 'validate'):
        parser.error("final checkpoint continuation requires modified rewards and smoke/train mode")
    if args.resume_final and args.mode == 'train' and (args.updates is None or not 1 <= args.updates <= 500):
        parser.error("final checkpoint continuation requires an explicit budget of 1..500 updates")
    if args.updates is None:
        args.updates = {"smoke": 0, "validate": 3, "train": 10000}[args.mode]
    if ((args.mode == "smoke" and args.updates != 0) or (args.mode == "validate" and args.updates != 3)
            or (args.mode == "train" and not 1 <= args.updates <= 10000)):
        parser.error("mode and update budget disagree")
    args.output = args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        timeout = 3600 if args.mode != 'train' else (172800 if args.terrain_profile == 'low_stairs' else 86400)
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"], gpu=args.gpu,
                   output=args.output, timeout=timeout)
    try:
        worker(args)
    except Exception:
        traceback.print_exc()
        save(args.output / "error.json", dict(error=traceback.format_exc()))
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
