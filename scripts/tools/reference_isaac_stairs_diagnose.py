"""Frozen final-policy diagnosis in its original Isaac Lab environment.

Native square up/down stairs, fixed heights/commands/starts. No optimizer or reward edits.
Nominal and training-randomized physics are reported separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import ROOT, load_reference, resolve_lite3_asset, source_manifest
from reference_stairs_eval import MODELS, load_actor, save, sha
from reference_eval_checkpoint import register_checkpoint

HEIGHTS = (0., .05, .10, .15)
HEIGHT_SETS = {'reference': HEIGHTS, 'low_stairs': (0., .02, .03, .05)}
SPEEDS = (.32, .4, .6)
SEED = 20261112
EXPECTED_SHA = '5cf4c1160b6f5b7b5102844526b26f3886c27562d98ea9558e3fa09da504f5a1'


def protocol(repeats, heights=HEIGHTS):
    return [dict(env_id=len(SPEEDS)*repeats*h + repeats*s + r,
                 height=height, speed=speed, repeat=r, column=h)
            for h, height in enumerate(heights) for s, speed in enumerate(SPEEDS)
            for r in range(repeats)]


def reward_tradeoff(command=.4, pitch_deg=15.):
    """Illustrative single-term rates, not total returns or a causal experiment."""
    return dict(stationary_velocity_rate=4*math.exp(-command**2/.5),
                matched_velocity_rate=4., orientation_rate=-20*math.sin(math.radians(pitch_deg))**2)


def step_cleared(direction, feet, root, front, top, tilt, support, torso_contact):
    """Preserve the upward criterion; require a supported descent for down."""
    clear = (feet[:,0].min()>front+.022 and feet[:,2].min()>top-.01
             and root[2]>top+.18 and tilt<44.)
    if direction=='down':
        clear = (clear and root[2]<top+.55 and feet[:,2].max()<top+.12
                 and support>=2 and not torso_contact)
    elif direction!='up':
        raise ValueError(direction)
    return bool(clear)


def collect(args, app):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.math import quat_from_euler_xyz
    from isaaclab.utils.warp import raycast_mesh
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference, _ = load_reference()  # imports the reference reward namespace
    from reference_six_terrain_cfg import replace_terrain
    # Resolve after load_reference, which supplies the reference namespace.
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as reward_functions
    cfg = replace_terrain(reference)
    saved = json.loads((ROOT/'logs/reference_six/training/scratch_seed42_10000/configuration_audit.json').read_text())
    for key in ('rewards', 'actions', 'observations', 'terminations'):
        current = json.loads(json.dumps(cfg.to_dict()[key], default=str))
        if current != saved['candidate_environment'][key]:
            raise ValueError(f'training configuration changed: {key}')
    cfg.seed = SEED
    heights_to_test = HEIGHT_SETS[args.height_set]
    specs = protocol(args.repeats, heights_to_test)
    cfg.scene.num_envs = len(specs)
    cfg.sim.device = 'cuda:0'
    resolve_lite3_asset(cfg)
    generator = cfg.scene.terrain.terrain_generator
    source = copy.deepcopy(generator.sub_terrains)
    generator.num_rows, generator.num_cols = 1, 4
    generator.curriculum = True  # deterministic column allocation, no progression
    generator.seed = SEED
    generator.use_cache = False
    subs = {'flat': copy.deepcopy(source['flat'])}
    for h in heights_to_test[1:]:
        sub = copy.deepcopy(source[f'stairs_{args.direction}'])
        sub.step_height_range = (h, h)
        subs[f'stairs_{round(h*100):02d}cm'] = sub
    for sub in subs.values():
        sub.proportion = .25
    generator.sub_terrains = subs
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.gait_level = None
    cfg.commands.base_velocity.debug_vis = False
    # All variants receive exactly the same upright initial pose distribution.
    reset = cfg.events.randomize_reset_base
    reset.params['pose_range'] = dict(x=(.20,.30), y=(-.10,.10), z=(0.,0.),
                                    roll=(0.,0.), pitch=(0.,0.), yaw=(-.03,.03))
    reset.params['velocity_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    if args.profile == 'nominal':
        cfg.observations.policy.enable_corruption = False
        for name in ('randomize_rigid_body_mass', 'randomize_rigid_body_inertia',
                     'randomize_com_positions', 'randomize_actuator_gains'):
            setattr(cfg.events, name, None)
        cfg.events.randomize_rigid_body_material.params.update(
            static_friction_range=(.9,.9), dynamic_friction_range=(.9,.9),
            restitution_range=(0.,0.), make_consistent=True)
        # The training actuator class and delay distribution remain enabled.
    if args.video_env is not None:
        if not 0 <= args.video_env < len(specs):
            raise ValueError('video environment outside protocol')
        cfg.viewer.resolution = (960,540)
        cfg.sim.render_interval = cfg.decimation*5
        cfg.scene.terrain.visual_material = None
    config = json.loads(json.dumps(cfg.to_dict(), default=str))
    base = ManagerBasedRLEnv(cfg=cfg, render_mode='rgb_array' if args.video_env is not None else None)
    args.environment = base
    obs, _ = base.reset()
    robot, terrain = base.scene['robot'], base.scene.terrain
    sensor = base.scene['contact_forces']
    command = base.command_manager.get_term('base_velocity')
    device, n, dt = base.device, base.num_envs, base.step_dt
    assert dt == .02 and cfg.sim.dt == .005
    expected_columns = torch.tensor([s['column'] for s in specs],device=device)
    assert torch.equal(terrain.terrain_types, expected_columns)
    model_hash = sha(MODELS[args.model])
    actor = load_actor(args.model).to(device)
    action_generator = torch.Generator(device=device).manual_seed(SEED+100)
    state = torch.load(MODELS[args.model],map_location=device,weights_only=True)['actor_state_dict']
    action_std = state['distribution.log_std_param'].exp()
    actor_hashes = {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in actor.state_dict().items()}
    foot_ids = [robot.body_names.index(f'{leg}_FOOT') for leg in ('FL','FR','HL','HR')]
    contact_ids = [sensor.body_names.index(f'{leg}_FOOT') for leg in ('FL','FR','HL','HR')]
    torso_id = sensor.body_names.index('TORSO')
    action_joints = base.action_manager.get_term('joint_pos')._joint_ids
    action_joint_names = [robot.joint_names[i] for i in action_joints]
    expected_names = [f'{leg}_{joint}_joint' for leg in ('FL','FR','HL','HR') for joint in ('HipX','HipY','Knee')]
    assert action_joint_names == expected_names
    assert obs['policy'].shape == (n,45) and obs['critic'].shape == (n,393)

    # Actual scanner geometry, along the forward centreline of each tile.
    mesh = RayCaster.meshes[cfg.scene.terrain.prim_path]
    xs = torch.linspace(0.,3.8,1901,device=device)
    tile_origins = terrain.terrain_origins[0]
    starts = tile_origins[:,None,:] + torch.stack((xs,torch.zeros_like(xs),torch.full_like(xs,20.)),dim=-1)[None]
    directions = torch.zeros_like(starts); directions[...,2] = -1
    hits = raycast_mesh(starts.reshape(-1,3),directions.reshape(-1,3),mesh)[0].reshape(4,-1,3)
    heights = hits[:,:,2]-tile_origins[:,None,2]
    assert torch.isfinite(hits).all()
    geometry = []
    for col,h in enumerate(heights_to_test):
        z = heights[col].cpu().numpy()
        sign = 1 if args.direction=='up' else -1
        edges = np.flatnonzero(sign*np.diff(z) > .5*max(h,.01))+1
        fronts = xs[edges].cpu().numpy().tolist()
        # Sampling at 2 mm resolution brackets each riser; use midpoint.
        fronts = [round(v-.001,4) for v in fronts]
        tops = z[edges].tolist()
        if h:
            assert len(fronts) >= 5 and np.allclose(np.diff([0.]+tops),sign*h,atol=.002)
        else:
            assert np.max(np.abs(z)) < 1e-4
        geometry.append(dict(column=col,height=h,fronts=fronts,tops=tops,
                             root_success_x=fronts[-1]+.30 if fronts else 3.3))
    save(args.output/'geometry.json',geometry)

    # Paired initial offsets across height and command groups.
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform([-.05,-.10,-.03],[.05,.10,.03],size=(args.repeats,3))
    offsets = torch.tensor(np.asarray([jitter[s['repeat']] for s in specs]),device=device,dtype=torch.float32)
    roots = robot.data.default_root_state.clone()
    roots[:,:3] += base.scene.env_origins
    roots[:,0] += .25 + offsets[:,0]
    roots[:,1] += offsets[:,1]
    roots[:,3:7] = quat_from_euler_xyz(offsets[:,2]*0,offsets[:,2]*0,offsets[:,2])
    roots[:,7:] = 0
    robot.write_root_state_to_sim(roots)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_vel))
    base.sim.forward(); base.scene.update(dt)
    target_speed = torch.tensor([s['speed'] for s in specs],device=device)
    rollout_tick = -1
    def set_command(tick=None):
        tick = rollout_tick if tick is None else tick
        t = max(0.,tick*dt)
        command.is_heading_env[:] = False
        command.is_standing_env[:] = False
        command.vel_command_b[:] = 0
        if tick >= 0:
            command.vel_command_b[:,0] = target_speed*min(max((t-.5)/.5,0.),1.)
    original_commands = base.command_manager.compute
    def compute_command(dt):
        original_commands(dt)
        set_command(rollout_tick+1 if rollout_tick>=0 else -1)
    base.command_manager.compute = compute_command
    set_command()
    final_progress = json.loads((ROOT/'logs/reference_six/training/scratch_seed42_10000/progress.json').read_text())
    final_mean = final_progress['latest_update']['mean_terrain_level']
    gait = reward_functions.update_gait_level_from_terrain_mean(final_mean)
    with torch.inference_mode():
        for _ in range(50):
            obs, _, terminated, truncated, _ = base.step(torch.zeros(n,12,device=device))
            if (terminated|truncated).any():
                raise RuntimeError('standing preparation terminated')
    base.episode_length_buf[:] = 0
    base.action_manager.reset()
    rollout_tick = 0
    set_command()
    obs = base.observation_manager.compute()
    initial = dict(root=robot.data.root_state_w.cpu().numpy(),joint_pos=robot.data.joint_pos.cpu().numpy(),
                   joint_vel=robot.data.joint_vel.cpu().numpy(),observation=obs['policy'].cpu().numpy(),
                   origins=base.scene.env_origins.cpu().numpy(),materials=robot.root_physx_view.get_material_properties().cpu().numpy(),
                   phase_foot_reference=base.phase_foot_ref_body.cpu().numpy())
    np.savez_compressed(args.output/'initial.npz',**initial)
    reward_names = list(base.reward_manager.active_terms)
    termination_names = list(base.termination_manager.active_terms)
    save(args.output/'config.json',dict(profile=args.profile,direction=args.direction,configuration=config,geometry=geometry,
         model=args.model,checkpoint=str(MODELS[args.model]),checkpoint_sha256=model_hash,specs=specs,
         reward_scoring='reference_frozen_common',
         action_mode=args.action_mode,action_std=action_std.cpu().tolist(),
         reward_names=reward_names,termination_names=termination_names,foot_names=['FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT'],
         joint_names=list(robot.joint_names),action_joint_names=action_joint_names,gait_level=gait,
         source_final_mean_terrain_level=final_mean,
         overrides=['240 environments by default; fixed native stairs 5/10/15 cm plus flat; one row/four columns',
                    'paired upright starts, stationary 1 s PD preparation, 0.5 s pause + 0.5 s command ramp',
                    'curriculum progression disabled; reward gait multiplier fixed to final training mean',
                    'nominal: fixed friction .9/restitution 0, no mass/inertia/COM/gain/observation randomization; actuator delay remains native',
                    'training_randomized: original mass/inertia/COM/material/gain/observation randomization and actuator delays'],
         rate_units='weighted reward per second; multiply by step_dt for the step reward',sources=source_manifest()))
    active = np.ones(n,bool)
    records = [None]*n
    full_dwell = np.zeros(n)
    best_risers = np.zeros(n,int)
    riser_dwell = np.zeros((n,max(len(g['fronts']) for g in geometry)))
    buffers = {key:[] for key in ('active','root','quat','velocity','gravity','feet','forces','action','joint_pos',
                                'joint_vel','joint_target','torque','reward_rates','command','scan_mean',
                                'terminated','truncated','term_flags')}
    original_reward = base.reward_manager.compute
    steps = args.steps or round(cfg.episode_length_s/dt)
    index = 0
    frames, frame_steps = [], []
    if args.video_env is not None:
        from pxr import UsdGeom
        for i,path in enumerate(base.scene.env_prim_paths):
            if i != args.video_env:
                UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(path)).MakeInvisible()
        UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(base.scene.env_prim_paths[args.video_env])).MakeVisible()

    def capture(dt):
        value = original_reward(dt)
        data = robot.data
        rates = base.reward_manager._step_reward
        if not torch.allclose(value,rates.sum(-1)*dt,atol=1e-5,rtol=1e-4):
            raise ValueError('reward rate accounting mismatch')
        scan = base.scene['height_scanner_base'].data.ray_hits_w[:,:,2]
        values = dict(root=data.root_pos_w-base.scene.env_origins,quat=data.root_quat_w,
            velocity=torch.cat((data.root_lin_vel_b,data.root_ang_vel_b),-1),gravity=data.projected_gravity_b,
            feet=data.body_pos_w[:,foot_ids]-base.scene.env_origins[:,None,:],
            forces=sensor.data.net_forces_w[:,contact_ids],action=base.action_manager.action,
            joint_pos=data.joint_pos,joint_vel=data.joint_vel,joint_target=data.joint_pos_target,
            torque=data.applied_torque,reward_rates=rates,command=command.command,
            scan_mean=scan.mean(-1)-base.scene.env_origins[:,2],
            terminated=base.reset_terminated,truncated=base.reset_time_outs,
            term_flags=torch.stack([base.termination_manager.get_term(k) for k in termination_names],-1))
        arrays={k:v.detach().cpu().numpy().copy() for k,v in values.items()}
        if any(not np.isfinite(v).all() for v in arrays.values()) or not torch.isfinite(scan).all():
            raise FloatingPointError('nonfinite original-environment diagnostic')
        buffers['active'].append(active.copy())
        for key,array in arrays.items(): buffers[key].append(array)
        torso_contact = sensor.data.net_forces_w[:,torso_id].norm(dim=-1).cpu().numpy()>1
        for i in np.flatnonzero(active):
            spec, geom = specs[i], geometry[specs[i]['column']]
            root,feet = arrays['root'][i],arrays['feet'][i]
            tilt = math.degrees(math.acos(np.clip(-arrays['gravity'][i,2],-1.,1.)))
            support=np.count_nonzero(arrays['forces'][i,:,2]>1.)
            for j,(front,top) in enumerate(zip(geom['fronts'],geom['tops'])):
                clear = step_cleared(args.direction,feet,root,front,top,tilt,support,torso_contact[i])
                riser_dwell[i,j] = riser_dwell[i,j]+dt if clear else 0.
                if riser_dwell[i,j]>=.1-1e-8: best_risers[i]=max(best_risers[i],j+1)
            top=geom['tops'][-1] if geom['tops'] else 0.
            crossed=(root[0]>geom['root_success_x'] and .18<root[2]-top<.55)
            if geom['fronts']: crossed &= feet[:,0].min()>geom['fronts'][-1]+.05
            if args.direction=='down' and geom['fronts']:
                crossed &= step_cleared('down',feet,root,geom['fronts'][-1],top,tilt,support,torso_contact[i])
            full_dwell[i]=full_dwell[i]+dt if crossed else 0.
            outcome=None
            if arrays['terminated'][i]: outcome='orientation_failure'
            elif abs(root[1])>1.0: outcome='lateral_exit'
            elif full_dwell[i]>=.25 and support>=2 and not torso_contact[i]: outcome='success'
            elif arrays['truncated'][i] or index==steps-1:
                recent=max(0,index-round(5/dt))
                progress=root[0]-buffers['root'][recent][i,0]
                outcome='stuck' if progress<.15 else 'timeout'
            if outcome:
                records[i]={**spec,'outcome':outcome,'duration_s':round((index+1)*dt,4),
                    'final_root':root.tolist(),'best_risers':int(best_risers[i]),
                    'termination_terms':[name for name,flag in zip(termination_names,arrays['term_flags'][i]) if flag]}
        if args.video_env is not None and active[args.video_env] and (index%5==0 or records[args.video_env]):
            from PIL import Image,ImageDraw
            position=data.root_pos_w[args.video_env].cpu().numpy()
            base.sim.set_camera_view(eye=(position+[-.8,-2.5,1.15]).tolist(),target=(position+[.55,0.,0.]).tolist())
            for _ in range(12 if not frames else 2): frame=base.render()
            if frame.shape!=(540,960,3) or frame.std()<2: raise RuntimeError('blank simulator frame')
            frame=Image.fromarray(frame); draw=ImageDraw.Draw(frame)
            spec=specs[args.video_env]
            draw.rectangle((0,0,960,50),fill=(20,25,32))
            draw.text((10,6),f"{args.model} | Isaac Lab | {args.profile} | {args.direction} {spec['height']*100:.0f} cm | {spec['speed']:.2f} m/s",fill='white')
            outcome=records[args.video_env]['outcome'] if records[args.video_env] else 'running'
            draw.text((10,28),f't={(index+1)*dt:.2f}s | {outcome} | frozen policy',fill='white')
            frames.append(np.asarray(frame)); frame_steps.append(index+1)
        active[:]=[r is None for r in records]
        return value
    base.reward_manager.compute=capture
    with torch.inference_mode():
        for index in range(steps):
            rollout_tick=index
            # The native manager computes one observation at each step end,
            # after the next tick's command is installed by compute_command.
            if not torch.allclose(obs['policy'][:,6:9],command.command):
                raise ValueError('actor input command differs from reward command')
            action=actor(obs['policy'])
            if args.action_mode=='sampled':
                action=action+torch.randn(action.shape,device=device,generator=action_generator)*action_std
            action=action.clamp(-100,100)
            if not torch.isfinite(action).all(): raise FloatingPointError('nonfinite action')
            obs,_,terminated,truncated,_=base.step(action)
            if not np.array_equal((terminated|truncated).cpu().numpy(),buffers['terminated'][-1]|buffers['truncated'][-1]):
                raise ValueError('pre-reset flags differ from returned flags')
            if (index+1)%250==0: print(f"ISAAC STAIRS {args.profile}: {index+1}/{steps}, alive={active.sum()}",flush=True)
            if not active.any(): break
    arrays={k:np.asarray(v) for k,v in buffers.items()}
    np.savez_compressed(args.output/'trace.npz',**arrays)
    assert all(r is not None for r in records)
    after={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in actor.state_dict().items()}
    assert actor_hashes==after and sha(MODELS[args.model])==model_hash
    groups=[]
    for h in heights_to_test:
        for speed in SPEEDS:
            rows=[r for r in records if r['height']==h and r['speed']==speed]
            groups.append(dict(height=h,speed=speed,episodes=len(rows),
                outcomes={outcome:sum(r['outcome']==outcome for r in rows) for outcome in sorted({r['outcome'] for r in rows})},
                first_riser=sum(r['best_risers']>=1 for r in rows),best_risers=max(r['best_risers'] for r in rows)))
    report=dict(status='passed',profile=args.profile,direction=args.direction,action_mode=args.action_mode,episodes=n,steps=index+1,step_dt=dt,probe=args.steps is not None,
                height_set=args.height_set,
                model=args.model,actor_unchanged=True,checkpoint_sha256=model_hash,actor_hashes=after,
                first_episodes_only=True,pre_reset_capture=True,groups=groups,records=records,
                gait_level=gait,reward_names=reward_names,termination_names=termination_names,
                video_env=args.video_env,video_frame_steps=frame_steps,
                limitations=['Native stairs are square rings; count/entry distance differ from the six-riser MuJoCo scene.',
                             'Fixed starts/commands/heights, no training. Nominal still retains native random actuator delays.',
                             'Reward traces describe this frozen policy, not causal effects of modifying rewards.'])
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(args.output/'rollout.mp4',frames,fps=10,codec='libx264',quality=7,macro_block_size=2)
        imageio.imwrite(args.output/'snapshot.png',frames[min(len(frames)-1,80)])
    save(args.output/'report.json',report)
    print(json.dumps(groups),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='update10000')
    parser.add_argument('--checkpoint',type=Path); parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--profile',choices=('nominal','training_randomized'),required=True)
    parser.add_argument('--direction',choices=('up','down'),default='up')
    parser.add_argument('--height-set',choices=tuple(HEIGHT_SETS),default='reference')
    parser.add_argument('--action-mode',choices=('mean','sampled'),default='mean')
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=(0,1),default=0)
    parser.add_argument('--video-env',type=int)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args(); args.output=args.output.resolve()
    register_checkpoint(args.model,args.checkpoint,args.checkpoint_sha256)
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=args.gpu,output=args.output,timeout=2400)
    try:
        from isaaclab.app import AppLauncher
        # Vulkan uses the physical GPU index; CUDA_VISIBLE_DEVICES remaps physics to cuda:0.
        graphics_gpu = args.gpu if args.video_env is not None or args.height_set == 'low_stairs' else 0
        app=AppLauncher(headless=True,enable_cameras=args.video_env is not None,device='cuda:0',
            kit_args=f'--/renderer/multiGpu/enabled=False --/renderer/activeGpu={graphics_gpu}').app
        collect(args,app)
        # The existing supervised headless training worker uses the same bounded
        # exit, avoiding Isaac teardown hangs after all artifacts are flushed.
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc()
        save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__=='__main__':
    sys.exit(main())
