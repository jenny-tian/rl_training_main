"""Pre-registered non-stair retention check: 160 first episodes per profile/model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from reference_stair_allocation import OUTPUT, PARENT, DIGEST
from reference_stairs_eval import ROOT, save, sha, load_actor, MODELS
from reference_eval_checkpoint import register_checkpoint

SEED = 20261116
TERRAINS = {'flat':0, 'rough':4, 'slope_up':8, 'slope_down':10,
            'friction_lateral':16, 'friction_longitudinal':17, 'slope_rough_up':18, 'slope_rough_down':19}


def protocol():
    return [dict(env_id=i*20+j*10+r, terrain=name, column=col, row=level, repeat=r)
            for i,(name,col) in enumerate(TERRAINS.items()) for j,level in enumerate((1,8)) for r in range(10)]


def command_speed(t):
    return .4*min(max((t-.5)/.5, 0.), 1.)


def outcome(*, terminated, lateral, forward, support, dwell, expired):
    if terminated: return 'orientation_failure'
    if abs(lateral) > 1.: return 'lateral_exit'
    if forward >= 3. and support >= 2 and dwell >= .25: return 'success'
    if expired: return 'timeout'
    return None


def collect(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_from_euler_xyz
    from reference_lite3_loader import load_reference, resolve_lite3_asset, source_manifest
    from reference_reward_variants import plain
    reference, _ = load_reference()
    from reference_six_terrain_cfg import replace_terrain
    from reference_low_stairs_cfg import apply_terrain
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as rewards_module
    cfg, _ = apply_terrain(replace_terrain(reference))
    cfg.seed = SEED; cfg.scene.num_envs = 160; cfg.sim.device = 'cuda:0'
    cfg.scene.terrain.terrain_generator.seed = 42
    cfg.scene.terrain.terrain_generator.use_cache = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.curriculum.terrain_levels = cfg.curriculum.gait_level = None
    cfg.events.randomize_reset_base.params['pose_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.events.randomize_reset_base.params['velocity_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    if args.profile == 'nominal':
        cfg.observations.policy.enable_corruption = False
        for name in ('randomize_rigid_body_mass','randomize_rigid_body_inertia','randomize_com_positions','randomize_actuator_gains'):
            setattr(cfg.events, name, None)
        cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(.9,.9),
            dynamic_friction_range=(.9,.9), restitution_range=(0.,0.), make_consistent=True)
    resolve_lite3_asset(cfg)
    config = plain(cfg)
    base = ManagerBasedRLEnv(cfg=cfg)
    base.reset()
    robot, terrain = base.scene['robot'], base.scene.terrain
    specs = protocol(); dt = base.step_dt; n = len(specs)
    levels = torch.tensor([s['row'] for s in specs], device=base.device)
    types = torch.tensor([s['column'] for s in specs], device=base.device)
    terrain.terrain_levels.copy_(levels); terrain.terrain_types.copy_(types)
    terrain.env_origins.copy_(terrain.terrain_origins[levels, types])
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform([-.05,-.10,-.03], [.05,.10,.03], size=(10,3))
    offsets = torch.tensor(np.asarray([jitter[s['repeat']] for s in specs]), device=base.device, dtype=torch.float32)
    roots = robot.data.default_root_state.clone(); roots[:,:3] += terrain.env_origins
    roots[:,:2] += offsets[:,:2]; roots[:,7:] = 0
    roots[:,3:7] = quat_from_euler_xyz(offsets[:,2]*0, offsets[:,2]*0, offsets[:,2])
    robot.write_root_state_to_sim(roots)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    base.sim.forward(); base.scene.update(dt)
    command = base.command_manager.get_term('base_velocity')
    tick = -1
    def set_command(t):
        command.is_heading_env[:] = command.is_standing_env[:] = False
        command.vel_command_b[:] = 0
        if t >= 0: command.vel_command_b[:,0] = command_speed(t*dt)
    native_compute = base.command_manager.compute
    def compute(dt):
        native_compute(dt); set_command(tick+1 if tick >= 0 else -1)
    base.command_manager.compute = compute; set_command(-1)
    rewards_module.update_gait_level_from_terrain_mean(torch.tensor(5.))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    actor = load_actor(args.model).to(base.device)
    actor_digest = lambda: hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    before = actor_digest()
    with torch.inference_mode():
        for _ in range(50):
            obs, _, terminated, truncated, _ = base.step(torch.zeros(n,12,device=base.device))
            if (terminated|truncated).any(): raise RuntimeError('retention preparation failed')
    base.episode_length_buf[:] = 0; base.action_manager.reset(); tick = 0; set_command(0)
    obs = base.observation_manager.compute()
    initial = dict(root=robot.data.root_state_w.cpu().numpy(), joint_pos=robot.data.joint_pos.cpu().numpy(),
        joint_vel=robot.data.joint_vel.cpu().numpy(), observation=obs['policy'].cpu().numpy(),
        origins=terrain.env_origins.cpu().numpy(), materials=robot.root_physx_view.get_material_properties().cpu().numpy(),
        phase_reference=base.phase_foot_ref_body.cpu().numpy())
    np.savez_compressed(args.output/'initial.npz', **initial)
    sensor = base.scene['contact_forces']
    feet_ids, _ = sensor.find_bodies(['FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT'], preserve_order=True)
    save(args.output/'config.json', dict(cfg=config, specs=specs, model=args.model,
        checkpoint=str(MODELS[args.model]), model_sha256=args.checkpoint_sha256, profile=args.profile,
        source_sha256=sha(__file__), sources=source_manifest(), reward_scoring='reference_frozen_common',
        step_dt=dt, preparation_steps=50, maximum_s=20., success_forward_m=3., lateral_limit_m=1., dwell_s=.25))
    active = np.ones(n,bool); records = [None]*n; dwell = np.zeros(n)
    origin_start = initial['root'][:,:3].copy()
    buffers = {k:[] for k in ('active','root','velocity','gravity','command','action','terminated','truncated','support')}
    native_reward = base.reward_manager.compute
    def capture(dt):
        result = native_reward(dt); data = robot.data
        values = dict(root=data.root_pos_w, velocity=torch.cat((data.root_lin_vel_b,data.root_ang_vel_b),-1),
            gravity=data.projected_gravity_b, command=command.command, action=base.action_manager.action,
            terminated=base.reset_terminated, truncated=base.reset_time_outs,
            support=(sensor.data.net_forces_w[:,feet_ids,2] > 1.).sum(-1))
        arrays = {k:v.detach().cpu().numpy().copy() for k,v in values.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        buffers['active'].append(active.copy())
        for k,v in arrays.items(): buffers[k].append(v)
        delta = arrays['root'] - origin_start
        for i in np.flatnonzero(active):
            clear = delta[i,0] >= 3. and arrays['support'][i] >= 2 and abs(delta[i,1]) <= 1.
            dwell[i] = dwell[i]+dt if clear else 0.
            result_name = outcome(terminated=arrays['terminated'][i], lateral=delta[i,1], forward=delta[i,0],
                support=arrays['support'][i], dwell=dwell[i], expired=arrays['truncated'][i] or tick == 999)
            if result_name:
                records[i] = dict(**specs[i], outcome=result_name, duration_s=(tick+1)*dt, displacement=delta[i].tolist())
                active[i] = False
        return result
    base.reward_manager.compute = capture
    with torch.inference_mode():
        for tick in range(1000):
            assert torch.equal(obs['policy'][:,6:9], command.command)
            obs, *_ = base.step(actor(obs['policy']).clamp(-100,100))
            if not active.any(): break
    assert all(r is not None for r in records) and actor_digest() == before and sha(MODELS[args.model]) == args.checkpoint_sha256
    np.savez_compressed(args.output/'trace.npz', **{k:np.asarray(v) for k,v in buffers.items()})
    save(args.output/'report.json', dict(status='passed', model=args.model, profile=args.profile,
        model_sha256=args.checkpoint_sha256, episodes=n, records=records, actor_unchanged=True,
        first_episodes_only=True, pre_reset_capture=True, optimizer_updates=0, step_dt=dt))


def run_matrix(checkpoints):
    from reference_core500_evaluate import same_initial
    from reference_low_stairs_a_evaluate import same_configuration
    from reference_stair_allocation_evaluate import read, record_regressions
    matrix = []; regression = []
    for profile in ('nominal', 'training_randomized'):
        baseline = OUTPUT/'evaluation/retention/zero10100'/profile
        for model, (checkpoint,digest) in [('zero10100',(PARENT,DIGEST)), *checkpoints.items()]:
            dest = OUTPUT/'evaluation/retention'/model/profile
            label = f'allocation_retention_{model.lower()}'
            if not dest.exists():
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--model',label,
                    '--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--profile',profile,
                    '--output',str(dest)], cwd=ROOT, check=True)
            report = read(dest/'report.json')
            assert read(dest/'execution.json')['status'] == 'completed' and report['status'] == 'passed'
            assert report['episodes'] == len(report['records']) == 160 and report['model_sha256'] == digest
            assert report['actor_unchanged'] and report['pre_reset_capture'] and report['first_episodes_only']
            assert same_initial(dest,baseline) and same_configuration(read(dest/'config.json'),read(baseline/'config.json'))
            matrix.append(dict(model=model, profile=profile, path=str(dest), episodes=160))
            if model == 'B':
                for comparison,old_path in [('zero10100',baseline),('A',OUTPUT/'evaluation/retention/A'/profile)]:
                    regression += [dict(comparator=comparison, profile=profile, **r)
                                   for r in record_regressions(read(old_path/'report.json'), report, 'stairs')]
    return dict(matrix=matrix, regressions=regression, passed=not regression)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--profile', choices=('nominal','training_randomized'), required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--child', action='store_true')
    args = p.parse_args(); args.output = args.output.resolve()
    register_checkpoint(args.model, args.checkpoint, args.checkpoint_sha256)
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=2400, minimum_free_mib=10240)
    try:
        collect(args); sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output/'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__': sys.exit(main())
