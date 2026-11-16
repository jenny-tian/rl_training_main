"""Approved A/B allocation trial: each branch has 64 integration steps and 100 updates."""
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

from reference_stair_allocation import PARENT, DIGEST, OUTPUT, parent_state, migrate, counts
from reference_stairs_eval import save, sha
from reference_lite3_loader import load_reference, resolve_lite3_asset, source_manifest
from reference_low_stairs_resume import STATE_KEY, make_resumed_runner, terrain_contract, capture_curriculum, native_rewards, restore_algorithm
from reference_low_stairs_guarded_ppo import Preserver, attach, attach_saving, KEY
from reference_low_stairs_guarded_bank import BANK
from reference_low_stairs_clearance_train import check_algorithm
from reference_reward_resume import equal_state


def directory(variant, phase):
    # The first A startup stopped before reset/rollout. Preserve it as evidence.
    return OUTPUT / variant / ('integration_v2' if phase == 0 else 'train_001_100')


def simulation_audit(base, cfg, expected_types):
    """Read sensors only after the single restored initial reset has completed."""
    if not getattr(base, '_resume_initial_reset_done', False):
        raise ValueError('sensor audit requires the completed restored initial reset')
    import numpy as np
    import torch
    import trimesh
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics, UsdShade
    terrain = base.scene.terrain
    assert equal_state(terrain.terrain_types, expected_types)
    assert terrain._curriculum_allocation is None and not terrain._parallel_reset_sampler_enabled
    dims = {k: list(v) for k, v in base.observation_manager.group_obs_dim.items()}
    assert dims == dict(policy=[45], critic=[393])
    meshes, materials = [], []
    for path in terrain.terrain_prim_paths:
        prim = base.scene.stage.GetPrimAtPath(path + '/mesh')
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        materials.append(dict(path=path, material=str(material.GetPath()),
            static_friction=float(physics.GetStaticFrictionAttr().Get()),
            dynamic_friction=float(physics.GetDynamicFrictionAttr().Get()), faces=len(faces)))
    friction = sorted({round(v['static_friction'], 6) for v in materials if '/friction_' in v['path']})
    assert friction == [.5, .666667, .833333, 1.]
    collider = trimesh.util.concatenate(meshes)
    assert len(collider.faces) == terrain.reference_ray_mesh_triangles
    collision_warp = convert_to_warp_mesh(collider.vertices, collider.faces, device=base.device)
    offsets = torch.tensor([[x,y,20.] for x in (-3.17,-1.13,.03,1.19,3.11)
                            for y in (-3.17,-1.13,.03,1.19,3.11)], device=base.device)
    starts = (terrain.terrain_origins.reshape(-1,3)[:,None] + offsets).reshape(-1,3)
    directions = torch.zeros_like(starts); directions[:,2] = -1
    scan_hits = raycast_mesh(starts, directions, RayCaster.meshes[cfg.scene.terrain.prim_path])[0]
    collision_hits = raycast_mesh(starts, directions, collision_warp)[0]
    assert torch.isfinite(scan_hits).all() and torch.isfinite(collision_hits).all()
    error = float((scan_hits-collision_hits).abs().max())
    assert error <= 1e-4
    scanners = {}
    for name in ('height_scanner', 'height_scanner_base'):
        data = base.scene[name].data.ray_hits_w
        assert torch.isfinite(data).all(), name
        scanners[name] = list(data.shape)
    return dict(status='passed', observation_dimensions=dims, family_counts=counts(expected_types.cpu()),
        ray_probes=len(starts), scanner_collider_max_error_m=error, scanners=scanners,
        materials=materials, restored_initial_reset_done=True,
        collider_sha256=hashlib.sha256(collider.vertices.tobytes()+collider.faces.tobytes()).hexdigest())


def sources():
    from reference_low_stairs_zero_preservation_train import sources as historical
    files = [Path(__file__), Path(__file__).with_name('reference_stair_allocation.py'),
             Path(__file__).with_name('reference_six_terrain_cfg.py')]
    return historical() + [dict(path=str(p), sha256=sha(p)) for p in files]


def preflight(variant, phase):
    from reference_low_stairs_zero_preservation_evaluate import checked_training
    checkpoint, digest = checked_training()
    assert checkpoint == PARENT and digest == DIGEST
    if phase:
        for branch in ('A', 'B'):
            path = directory(branch, 0)
            report = json.loads((path / 'report.json').read_text())
            execution = json.loads((path / 'execution.json').read_text())
            assert report['status'] == 'passed' and execution['status'] == 'completed'
            assert report['optimizer_updates'] == 0 and report['environment_steps'] == 64
            assert report['sources'] == sources()
        a, b = (json.loads((directory(v, 0) / 'report.json').read_text()) for v in ('A', 'B'))
        assert a['mesh_sha256'] == b['mesh_sha256'] and a['initial_level_histogram'] == b['initial_level_histogram']
        audits = [json.loads((directory(v, 0) / 'simulation_audit.json').read_text()) for v in ('A', 'B')]
        assert audits[0]['collider_sha256'] == audits[1]['collider_sha256']
        assert audits[0]['materials'] == audits[1]['materials']


def worker(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from isaacsim.core.cloner import GridCloner
    from pxr import UsdGeom
    from reference_reward_variants import plain

    preflight(args.variant, args.phase)
    reference, agent = load_reference()
    from reference_low_stairs_guarded_cfg import configure
    from reference_low_stairs_validate import geometry_audit
    from reference_low_stairs_rewards import sample
    cfg, audit = configure(reference)
    cfg.seed = agent.seed = 42
    cfg.scene.num_envs = 4096
    cfg.sim.device = agent.device = 'cuda:0'
    agent.max_iterations = 100; agent.logger = 'tensorboard'
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    resolve_lite3_asset(cfg)
    old_contract = terrain_contract(cfg)
    configuration = dict(environment=plain(cfg), agent=plain(agent), contract=old_contract)
    assert configuration == json.loads((PARENT.parent / 'configuration.json').read_text())
    assert cfg.scene.filter_collisions and cfg.scene.replicate_physics and cfg.sim.device == 'cuda:0'
    save(args.output / 'configuration.json', configuration)
    save(args.output / 'configuration_audit.json', audit)
    save(args.output / 'implementation_sources.json', sources())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
    replication = []
    native_replication = GridCloner.replicate_physics
    def record_replication(self, *a, **kw):
        replication.append(dict(enable_env_ids=kw.get('enable_env_ids'), count=len(kw.get('prim_paths', []))))
        return native_replication(self, *a, **kw)
    GridCloner.replicate_physics = record_replication
    try:
        base = ManagerBasedRLEnv(cfg=cfg)
    finally:
        GridCloner.replicate_physics = native_replication
    assert any(r == dict(enable_env_ids=True, count=4096) for r in replication), replication
    terrain = base.scene.terrain
    assert type(terrain).update_env_origins is TerrainImporter.update_env_origins
    mesh = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath(terrain.reference_scan_path))
    mesh_digest = hashlib.sha256(np.asarray(mesh.GetPointsAttr().Get()).tobytes()
        + np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).tobytes()).hexdigest()
    parent = parent_state()
    saved, curriculum, contract, migration = migrate(base, parent, old_contract, args.variant)
    save(args.output / 'allocation_migration.json', migration)
    runner, env, restored = make_resumed_runner(base, agent.to_dict(), saved, curriculum, contract,
                                               log_dir=str(args.output / 'training'))
    preserver = Preserver(BANK, base.device, coefficient=0.)
    preserver.restore(parent['infos'][KEY])
    attach(runner.alg, preserver); attach_saving(runner, preserver)
    assert preserver.calls == 2000 and runner.current_learning_iteration == 10100
    assert equal_state(capture_curriculum(base, contract), curriculum)
    save(args.output / 'simulation_audit.json', simulation_audit(base, cfg, curriculum['terrain_types']))
    save(args.output / 'restore.json', dict(status='passed', **restored, parent=str(PARENT), parent_sha256=DIGEST))
    save(args.output / 'geometry.json', geometry_audit(base, cfg))
    report = dict(status='running', variant=args.variant, phase=args.phase, parent_sha256=DIGEST,
        physical_gpu=1, num_envs=4096, max_updates=100, optimizer_updates=0, environment_steps=0,
        updates=[], checkpoints=[], sources=sources(), mesh_sha256=mesh_digest,
        collision_filtering=dict(runtime_replications=replication, gpu_env_ids=True),
        initial_level_histogram=torch.bincount(terrain.terrain_levels, minlength=10).cpu().tolist(),
        initial_counts=counts(terrain.terrain_types.cpu()), initial_learning_rate=runner.alg.learning_rate,
        auxiliary_coefficient=0., terminated=0, truncated=0, automatic_extension=False)
    coverage = torch.zeros(200, dtype=torch.long, device=base.device)
    clearance = torch.zeros_like(coverage)
    moving = torch.zeros_like(coverage)
    native_reward = base.reward_manager.compute
    def reward_capture(dt):
        result = native_reward(dt)
        slot = terrain.terrain_types * 10 + terrain.terrain_levels
        coverage.add_(torch.bincount(slot, minlength=200))
        x = sample(base)
        expected = (terrain.terrain_types >= 12) & (terrain.terrain_types < 16) & (x['speed'] > .1)
        assert torch.equal(x['enabled'], expected)
        moving.add_(torch.bincount(slot[x['enabled']], minlength=200))
        term = base.reward_manager.get_term_cfg('stair_clearance').func
        clearance.add_(torch.bincount(slot[term.state.cost.mean(-1) > 0], minlength=200))
        return result
    base.reward_manager.compute = reward_capture
    native_step = env.step
    def step(action):
        budget = 64 if args.phase == 0 else 2400
        if report['environment_steps'] >= budget:
            raise RuntimeError('approved interaction budget exceeded')
        result = native_step(action)
        assert torch.isfinite(action).all() and torch.isfinite(result[1]).all()
        assert all(torch.isfinite(v).all() for v in result[0].values())
        for name in ('height_scanner', 'height_scanner_base'):
            assert torch.isfinite(base.scene[name].data.ray_hits_w).all()
        term = base.reward_manager.get_term_cfg('stair_clearance').func
        assert not term.state.active[result[2].bool()].any()
        assert equal_state(terrain.terrain_types, curriculum['terrain_types'])
        report['environment_steps'] += 1
        report['terminated'] += int(base.reset_terminated.sum())
        report['truncated'] += int(base.reset_time_outs.sum())
        return result
    env.step = step

    def checkpoint(n):
        check_algorithm(runner.alg.save(), 100+n)
        assert preserver.calls == (100+n)*20
        path = args.output / f'model_updates_{10100+n:05d}.pt'
        iteration = runner.current_learning_iteration
        try:
            runner.current_learning_iteration = 10099+n
            runner.save(str(path), infos=dict(phase_updates=n, completed_updates=10100+n,
                trial_budget=100, parent_sha256=DIGEST, variant=args.variant,
                validation_only=n == 0, stair_allocation_v1=migration))
        finally:
            runner.current_learning_iteration = iteration
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        assert equal_state(loaded['infos'][STATE_KEY], capture_curriculum(base, contract))
        assert loaded['infos'][KEY] == preserver.state()
        if n == 0:
            assert all(equal_state(parent[k], loaded[k]) for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'))
            assert equal_state(loaded['infos'][STATE_KEY], curriculum)
        report['checkpoints'].append(dict(phase_updates=n, path=str(path), sha256=sha(path)))
        return loaded

    if args.phase == 0:
        runner.logger.init_logging_writer()
        loaded = checkpoint(0)
        restored_again = restore_algorithm(runner, loaded)
        duplicate = Preserver(BANK, base.device, coefficient=0.)
        duplicate.restore(loaded['infos'][KEY])
        assert equal_state(preserver.sample()[1:], duplicate.sample()[1:])
        assert equal_state(preserver.sample()[0]['policy'], duplicate.sample()[0]['policy'])
        with torch.inference_mode():
            obs = env.get_observations()
            for _ in range(64):
                obs, *_ = env.step(runner.alg.actor(obs))
        assert all(equal_state(parent[k], runner.alg.save()[k]) for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'))
        assert preserver.calls == 2000
        report['restored_again'] = restored_again
        runner.logger.stop_logging_writer()
    else:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        native_init = runner.logger.init_logging_writer
        def initialize():
            native_init(); checkpoint(0)
        runner.logger.init_logging_writer = initialize
        native_log = runner.logger.log
        def record(**kw):
            n = len(report['updates'])+1
            assert n <= 100 and kw['it'] == 10099+n
            losses = {k: float(v) for k, v in kw['loss_dict'].items()}
            assert all(math.isfinite(v) for v in losses.values()) and losses['capability_preservation'] == 0.
            report['updates'].append(dict(update=n, losses=losses, learning_rate=runner.alg.learning_rate,
                collect_s=kw['collect_time'], learn_s=kw['learn_time'],
                mean_terrain_level=float(terrain.terrain_levels.float().mean()), gait_level=float(native_rewards().gait_level)))
            report['optimizer_updates'] = n
            native_log(**kw)
            if n in (50, 100): checkpoint(n)
            if n % 20 == 0: save(args.output / 'report.json', report)
        runner.logger.log = record
        save(args.output / 'report.json', report)
        runner.learn(num_learning_iterations=100, init_at_random_ep_len=True)
        assert len(report['updates']) == 100 and report['environment_steps'] == 2400
        check_algorithm(runner.alg.save(), 200)
        assert preserver.calls == 4000 and runner.current_learning_iteration == 10199
        report.update(output_checkpoint=report['checkpoints'][-1]['path'], output_sha256=report['checkpoints'][-1]['sha256'])
    assert sha(PARENT) == DIGEST
    report.update(status='passed', interactions=report['environment_steps']*4096,
        final_learning_rate=runner.alg.learning_rate, final_counts=counts(terrain.terrain_types.cpu()),
        final_level_histogram=torch.bincount(terrain.terrain_levels, minlength=10).cpu().tolist(),
        coverage_by_column_level=coverage.reshape(20,10).cpu().tolist(),
        translation_by_column_level=moving.reshape(20,10).cpu().tolist(),
        clearance_by_column_level=clearance.reshape(20,10).cpu().tolist(), reference_sources=source_manifest())
    save(args.output / 'report.json', report)
    print(f'ALLOCATION {args.variant} PHASE {args.phase} PASSED', flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('A', 'B'), required=True)
    p.add_argument('--phase', type=int, choices=(0, 1), required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args(); args.output = directory(args.variant, args.phase)
    if not args.child:
        preflight(args.variant, args.phase)
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=7200, minimum_free_mib=12288)
    try:
        worker(args)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
