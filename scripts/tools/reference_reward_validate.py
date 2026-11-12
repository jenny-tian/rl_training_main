"""Calibrate height or validate reward variants in Isaac Lab; no optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import ROOT, load_reference, resolve_lite3_asset, source_manifest
from reference_stairs_eval import load_actor, MODELS, save, sha
from reference_reward_variants import VARIANTS, BASE_HEIGHT_M, apply_reward_variant, plain


def collect(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    reference, agent = load_reference()
    from reference_six_terrain_cfg import replace_terrain, column_names, FAMILIES
    from reference_six_train import configuration_audit
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as reference_rewards
    original = replace_terrain(reference)
    save(args.output/'reference_configuration_audit.json', configuration_audit(reference, original, agent))
    cfg, audit = apply_reward_variant(original, args.variant)
    save(args.output/'reward_override_audit.json', audit)
    asset = resolve_lite3_asset(cfg)
    cfg.seed = 20261112
    cfg.scene.num_envs = 64 if args.mode == 'calibrate' else 120
    cfg.sim.device = 'cuda:0'
    cfg.commands.base_velocity.debug_vis = False
    calibration = args.mode == 'calibrate'
    if calibration:
        gen = cfg.scene.terrain.terrain_generator
        gen.sub_terrains = {'flat': gen.sub_terrains['flat']}
        gen.sub_terrains['flat'].proportion = 1.
        gen.size = (40., 40.); gen.num_rows = gen.num_cols = 1; gen.use_cache = False
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.observations.policy.enable_corruption = False
        for name in ('randomize_rigid_body_mass', 'randomize_rigid_body_inertia',
                     'randomize_com_positions', 'randomize_actuator_gains'):
            setattr(cfg.events, name, None)
        cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(.9,.9),
            dynamic_friction_range=(.9,.9), restitution_range=(0.,0.), make_consistent=True)
        for name in ('pose_range', 'velocity_range'):
            cfg.events.randomize_reset_base.params[name] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.curriculum.terrain_levels = cfg.curriculum.gait_level = None
    save(args.output/'config.json', dict(cfg=plain(cfg), asset=asset, mode=args.mode, variant=args.variant,
        steps=300 if calibration else 240, optimizer_updates=0,
        reward_source_sha256=sha(Path(__file__).with_name('reference_reward_variants.py'))))
    base = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = base.reset()
    robot = base.scene['robot']; command = base.command_manager.get_term('base_velocity')
    reference_rewards.update_gait_level_from_terrain_mean(3.)
    dt = base.step_dt
    assert dt == .02 and obs['policy'].shape == (base.num_envs,45) and obs['critic'].shape == (base.num_envs,393)
    commands = ((0.,0.,0.), (0.,0.,.2), (0.,0.,-.2), (0.,0.,.5), (0.,0.,-.5), (.4,0.,0.))
    tick = 0
    def set_command(next_tick):
        cmd = commands[0] if calibration else commands[min(next_tick//40,5)]
        command.is_heading_env[:] = command.is_standing_env[:] = False
        command.vel_command_b[:] = command.vel_command_b.new_tensor(cmd)
    original_compute = base.command_manager.compute
    def compute_command(dt):
        original_compute(dt); set_command(tick+1)
    base.command_manager.compute = compute_command
    set_command(0); obs = base.observation_manager.compute()
    actor = None if calibration else load_actor('update10000').to(base.device)
    model_sha = sha(MODELS['update10000'])
    actor_digest = lambda: hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    initial_actor_digest = actor_digest() if actor is not None else None
    height_trace=[]; joint_trace=[]; check_errors=[]; results=[]
    names = list(base.reward_manager.active_terms)
    disabled = {'core_no_phase': 'phase_foot_trajectory_exp', 'core_no_gait': 'feet_gait'}.get(args.variant)
    # Isaac keeps zero-weight terms in active_terms but skips their evaluation.
    if disabled: assert base.reward_manager.get_term_cfg(disabled).weight == 0.
    original_reward = base.reward_manager.compute
    def capture(dt):
        value = original_reward(dt)
        rates = base.reward_manager._step_reward
        assert torch.allclose(value, rates.sum(-1)*dt, atol=1e-5, rtol=1e-4)
        scan = base.scene['height_scanner_base'].data.ray_hits_w[:,:,2]
        assert torch.isfinite(scan).all() and torch.isfinite(rates).all()
        if disabled: assert (rates[:,names.index(disabled)] == 0.).all()
        if args.variant == 'core_no_phase': assert not hasattr(base,'phase_foot_ref_body')
        height_trace.append((robot.data.root_pos_w[:,2]-scan.mean(-1)).cpu().numpy().copy())
        joint_trace.append(robot.data.joint_pos.cpu().numpy().copy())
        if 'stand_still' in VARIANTS[args.variant]:
            term = base.reward_manager.get_term_cfg('stand_still')
            cmd = command.command
            standing = (cmd[:,:2].norm(dim=1)<.1) & (cmd[:,2].abs()<.1)
            ids = term.params['asset_cfg'].joint_ids
            expected = (robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids]).abs().sum(-1)*standing*term.weight
            error = (rates[:,names.index('stand_still')]-expected).abs().max().item()
            assert error < 1e-6
            check_errors.append(error)
        if args.variant == 'core_airtime_nonnegative':
            for name in ('feet_air_time_lin_xy','feet_air_time_ang_z_lite3'):
                assert (rates[:,names.index(name)]>=0).all()
        return value
    base.reward_manager.compute = capture
    steps = 300 if calibration else 240
    with torch.inference_mode():
        for tick in range(steps):
            assert torch.equal(obs['policy'][:,6:9], command.command)
            actions = torch.zeros(base.num_envs,12,device=base.device) if calibration else actor(obs['policy']).clamp(-100,100)
            assert torch.isfinite(actions).all()
            obs, rew, done, trunc, _ = base.step(actions)
            assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(rew).all()
            if calibration: assert not (done|trunc).any(), 'calibration posture terminated'
            results.append(dict(step=tick+1,terminated=int(done.sum()),truncated=int(trunc.sum())))
    assert sha(MODELS['update10000']) == model_sha
    if actor is not None: assert actor_digest() == initial_actor_digest
    heights = np.asarray(height_trace)
    np.savez_compressed(args.output/'trace.npz',height_above_scan=heights,joint_pos=np.asarray(joint_trace))
    last = heights[-50:]
    if calibration:
        assert np.max(np.abs(last-BASE_HEIGHT_M)) < .025, 'height target needs recalibration'
    cols = column_names(original)
    families = sorted({FAMILIES[cols[int(i)]] for i in base.scene.terrain.terrain_types.cpu().tolist()}) if not calibration else ['flat']
    if not calibration: assert len(families)==6
    save(args.output/'report.json',dict(status='passed',mode=args.mode,variant=args.variant,steps=steps,
        environments=base.num_envs,optimizer_updates=0,checkpoint_sha256=model_sha,actor_unchanged=True,
        reward_terms=names,families=families,all_finite=True,stand_gate_max_error=max(check_errors,default=None),
        mean_last_second_height=float(last.mean()),median_last_second_height=float(np.median(last)),
        last_second_height_min=float(last.min()),last_second_height_max=float(last.max()),
        proposed_height_m=BASE_HEIGHT_M,calibration_target_verified=calibration,
        calibration_joint_names=list(robot.joint_names) if calibration else None,
        calibration_default_joint_pos=robot.data.default_joint_pos[0].cpu().tolist() if calibration else None,
        results=results,sources=source_manifest(),
        limitation='Integration/calibration only. Frozen policy; no evidence of learned behavior improvement.'))
    print(f'REWARD VALIDATION passed: {args.mode}/{args.variant}, {steps} steps, no optimizer',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('calibrate','smoke'),required=True)
    p.add_argument('--variant',choices=tuple(VARIANTS),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),default=0)
    p.add_argument('--child',action='store_true')
    args=p.parse_args();args.output=args.output.resolve()
    if args.mode=='calibrate' and args.variant!='core': p.error('calibration uses the core preset')
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=args.gpu,output=args.output,timeout=1800)
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cuda:0').app
        collect(args);sys.stdout.flush();os._exit(0)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__': sys.exit(main())
