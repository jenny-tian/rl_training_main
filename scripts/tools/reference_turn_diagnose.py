"""Paired frozen checkpoint turn evaluation in original Isaac Lab physics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from reference_lite3_loader import ROOT, load_reference, resolve_lite3_asset, source_manifest
from reference_stairs_eval import MODELS, TRAIN, load_actor, sha, save

CASES = ('standing','forward','low_left','low_right','turn_left','turn_right','fast_left','fast_right',
         'stop_left','stop_right','reverse_left','reverse_right','heading_left','heading_right','arc_left','arc_right')
SEED = 20261114


def schedule(case, t):
    if case not in CASES: raise ValueError(case)
    if t<2 or case=='standing': return (0.,0.,0.,None)
    sign=-1 if case.endswith('right') else 1
    if case=='forward': return (.4 if t<14 else 0.,0.,0.,None)
    if case.startswith('heading'): return (0.,0.,0.,sign*math.pi/2)
    speed=.2 if case.startswith('low') else .8 if case.startswith('fast') else .5
    if case.startswith('stop') and t>=8: speed=0.
    if case.startswith('reverse') and t>=8: sign=-sign
    vx=.3 if case.startswith('arc') else 0.
    return (vx,0.,sign*speed,None)


def collect(args, app):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_from_euler_xyz,wrap_to_pi
    reference,_=load_reference()
    from reference_six_terrain_cfg import replace_terrain
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as rewards_module
    cfg=replace_terrain(reference)
    specs=[dict(env_id=i*args.repeats+r,case=name,repeat=r) for i,name in enumerate(CASES) for r in range(args.repeats)]
    n=len(specs); cfg.scene.num_envs=n; cfg.seed=SEED; cfg.sim.device='cuda:0'
    resolve_lite3_asset(cfg)
    # Native terrain scanner and friction bindings retained on a large flat tile.
    gen=cfg.scene.terrain.terrain_generator
    gen.sub_terrains={'flat':gen.sub_terrains['flat']}; gen.sub_terrains['flat'].proportion=1.
    gen.size=(40.,40.); gen.num_rows=gen.num_cols=1; gen.seed=SEED; gen.use_cache=False
    cfg.scene.terrain.max_init_terrain_level=0
    cfg.curriculum.terrain_levels=cfg.curriculum.gait_level=None
    cfg.commands.base_velocity.debug_vis=False
    cfg.events.randomize_reset_base.params['pose_range']={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.events.randomize_reset_base.params['velocity_range']={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    if args.profile=='nominal':
        cfg.observations.policy.enable_corruption=False
        for name in ('randomize_rigid_body_mass','randomize_rigid_body_inertia','randomize_com_positions','randomize_actuator_gains'):
            setattr(cfg.events,name,None)
        cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(.9,.9),dynamic_friction_range=(.9,.9),
                                                              restitution_range=(0.,0.),make_consistent=True)
    if args.video_env is not None:
        cfg.viewer.resolution=(960,540); cfg.sim.render_interval=cfg.decimation*5
        cfg.scene.terrain.visual_material=None
    cfg_dict=json.loads(json.dumps(cfg.to_dict(),default=str))
    base=ManagerBasedRLEnv(cfg=cfg,render_mode='rgb_array' if args.video_env is not None else None)
    obs,_=base.reset()
    robot=base.scene['robot']; command=base.command_manager.get_term('base_velocity'); dt=base.step_dt
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.model not in MODELS: MODELS[args.model]=TRAIN/'training'/f'model_{int(args.model[6:])-1}.pt'
    model_hash=sha(MODELS[args.model]); actor=load_actor(args.model).to(base.device)
    actor_hash=lambda:{k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in actor.state_dict().items()}
    before=actor_hash()
    jitter=np.random.default_rng(SEED).uniform([-.05,-.05,-.03],[.05,.05,.03],size=(args.repeats,3))
    offsets=torch.tensor(np.asarray([jitter[s['repeat']] for s in specs]),dtype=torch.float32,device=base.device)
    roots=robot.data.default_root_state.clone(); roots[:,:3]+=base.scene.env_origins
    roots[:,:2]+=offsets[:,:2]; roots[:,3:7]=quat_from_euler_xyz(offsets[:,2]*0,offsets[:,2]*0,offsets[:,2]); roots[:,7:]=0
    robot.write_root_state_to_sim(roots); robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_vel))
    base.sim.forward(); base.scene.update(dt)
    initial_heading=robot.data.heading_w.clone(); tick=-1
    def set_command(next_tick):
        command.is_heading_env[:]=False; command.is_standing_env[:]=False; command.vel_command_b[:]=0
        if next_tick<0: return
        for i,name in enumerate(CASES):
            part=slice(i*args.repeats,(i+1)*args.repeats)
            vx,vy,wz,heading=schedule(name,round(next_tick*dt,8))
            command.vel_command_b[part]=command.vel_command_b.new_tensor((vx,vy,wz))
            if heading is not None:
                command.heading_target[part]=initial_heading[part]+heading
                command.is_heading_env[part]=True
        command._update_command()
    original_command=base.command_manager.compute
    def command_compute(dt):
        original_command(dt); set_command(tick+1 if tick>=0 else -1)
    base.command_manager.compute=command_compute; set_command(-1)
    final=json.loads((TRAIN/'progress.json').read_text())['latest_update']['mean_terrain_level']
    rewards_module.update_gait_level_from_terrain_mean(final)
    with torch.inference_mode():
        for _ in range(50):
            obs,_,done,trunc,_=base.step(torch.zeros(n,12,device=base.device))
            if (done|trunc).any(): raise RuntimeError('preparation failed')
    base.episode_length_buf[:]=0; base.action_manager.reset(); tick=0; set_command(0)
    obs=base.observation_manager.compute()
    initial=dict(root=robot.data.root_state_w.cpu().numpy(),joint_pos=robot.data.joint_pos.cpu().numpy(),
        joint_vel=robot.data.joint_vel.cpu().numpy(),observation=obs['policy'].cpu().numpy(),
        materials=robot.root_physx_view.get_material_properties().cpu().numpy(),phase_reference=base.phase_foot_ref_body.cpu().numpy())
    np.savez_compressed(args.output/'initial.npz',**initial)
    reward_names=list(base.reward_manager.active_terms); term_names=list(base.termination_manager.active_terms)
    save(args.output/'config.json',dict(model=args.model,profile=args.profile,model_sha256=model_hash,
         cfg=cfg_dict,specs=specs,step_dt=dt,reward_names=reward_names,termination_names=term_names,
         protocol_source_sha256=sha(__file__),sources=source_manifest(),
         overrides='flat 40 m tile; fixed paired starts and commands; final gait multiplier; original nominal or randomized physics; deterministic policy'))
    buffers={k:[] for k in ('active','command','velocity','root','heading','heading_error','gravity','action','reward_rates','terminated','truncated')}
    active=np.ones(n,bool); records=[None]*n; index=0
    frames=[]; frame_steps=[]
    if args.video_env is not None:
        from pxr import UsdGeom
        for i,path in enumerate(base.scene.env_prim_paths):
            if i!=args.video_env: UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(path)).MakeInvisible()
        UsdGeom.Imageable(base.scene.stage.GetPrimAtPath(base.scene.env_prim_paths[args.video_env])).MakeVisible()
    original_reward=base.reward_manager.compute
    steps=args.steps or round(cfg.episode_length_s/dt)
    def capture(dt):
        reward=original_reward(dt); data=robot.data; rates=base.reward_manager._step_reward
        assert torch.allclose(reward,rates.sum(-1)*dt,atol=1e-5,rtol=1e-4)
        values=dict(command=command.command,velocity=torch.cat((data.root_lin_vel_b,data.root_ang_vel_b),-1),
            root=data.root_pos_w-base.scene.env_origins,heading=data.heading_w,
            heading_error=wrap_to_pi(command.heading_target-data.heading_w),gravity=data.projected_gravity_b,
            action=base.action_manager.action,reward_rates=rates,terminated=base.reset_terminated,truncated=base.reset_time_outs)
        arrays={k:v.detach().cpu().numpy().copy() for k,v in values.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        buffers['active'].append(active.copy())
        for key,v in arrays.items(): buffers[key].append(v)
        for i in np.flatnonzero(active):
            outcome=None
            if arrays['terminated'][i]: outcome='orientation_failure'
            elif np.max(np.abs(arrays['root'][i,:2]))>15: outcome='boundary_exit'
            elif arrays['truncated'][i] or index==steps-1: outcome='survived'
            if outcome: records[i]={**specs[i], 'outcome':outcome,'duration_s':(index+1)*dt}
        if args.video_env is not None and active[args.video_env] and (index%5==0 or records[args.video_env]):
            from PIL import Image,ImageDraw
            position=data.root_pos_w[args.video_env].cpu().numpy()
            base.sim.set_camera_view(eye=[2.,-2.5,1.6],target=[0.,0.,.2])
            for _ in range(12 if not frames else 2): frame=base.render()
            assert frame.shape==(540,960,3) and frame.std()>2
            frame=Image.fromarray(frame); draw=ImageDraw.Draw(frame); i=args.video_env
            draw.rectangle((0,0,960,54),fill=(20,25,32))
            draw.text((10,6),f"{args.model} | {args.profile} | {specs[i]['case']} | start {specs[i]['repeat']} | Isaac Lab",fill='white')
            draw.text((10,29),f"t={(index+1)*dt:.2f}s | yaw command={arrays['command'][i,2]:.2f} actual={arrays['velocity'][i,5]:.2f} rad/s",fill='white')
            frames.append(np.asarray(frame)); frame_steps.append(index+1)
        active[:]=[r is None for r in records]
        return reward
    base.reward_manager.compute=capture
    with torch.inference_mode():
        for index in range(steps):
            tick=index
            assert torch.allclose(obs['policy'][:,6:9],command.command)
            action=actor(obs['policy']).clamp(-100,100); assert torch.isfinite(action).all()
            obs,_,_,_,_=base.step(action)
            if (index+1)%250==0: print(f'TURN {args.model}/{args.profile} {index+1}/{steps} alive={active.sum()}',flush=True)
            if not active.any(): break
    assert all(r is not None for r in records)
    assert actor_hash()==before and sha(MODELS[args.model])==model_hash
    np.savez_compressed(args.output/'trace.npz',**{k:np.asarray(v) for k,v in buffers.items()})
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(args.output/'rollout.mp4',frames,fps=10,codec='libx264',quality=7,macro_block_size=2)
        imageio.imwrite(args.output/'snapshot.png',frames[min(60,len(frames)-1)])
    report=dict(status='passed',model=args.model,profile=args.profile,model_sha256=model_hash,actor_unchanged=True,
        episodes=n,steps=index+1,probe=args.steps is not None,records=records,reward_names=reward_names,
        step_dt=dt,video_env=args.video_env,video_frame_steps=frame_steps)
    save(args.output/'report.json',report)
    print(json.dumps({case:sum(r['outcome']=='orientation_failure' for r in records if r['case']==case) for case in CASES}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('update4001','update6001','update8001','update10000'),required=True)
    p.add_argument('--profile',choices=('nominal','training_randomized'),required=True)
    p.add_argument('--repeats',type=int,default=20); p.add_argument('--steps',type=int)
    p.add_argument('--video-env',type=int); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),default=0); p.add_argument('--child',action='store_true')
    args=p.parse_args(); args.output=args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],gpu=args.gpu,output=args.output,timeout=2400)
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=args.video_env is not None,device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0').app
        collect(args,app); sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__=='__main__': sys.exit(main())
