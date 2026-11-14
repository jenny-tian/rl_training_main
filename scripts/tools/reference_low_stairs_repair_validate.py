"""Small frozen integration check of each independent repair, with no optimizer."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import load_reference, resolve_lite3_asset, source_manifest
from reference_low_stairs_evaluate import EXPECTED_SHA, MODEL, TRAINING
from reference_stairs_eval import MODELS, load_actor, save, sha


def collect(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    reference,_=load_reference()
    from reference_six_terrain_cfg import replace_terrain, column_names, FAMILIES
    from reference_low_stairs_cfg import apply_terrain, apply_rewards
    from reference_low_stairs_rewards import sample
    from reference_low_stairs_repair import (apply_repair, wall_sample, stair_riser_contact,
        yaw_airtime_without_short_step_penalty, pure_yaw, VARIANTS)
    from reference_reward_variants import plain
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as original

    cfg,_=apply_terrain(replace_terrain(reference));cfg,_=apply_rewards(cfg)
    baseline=plain(cfg)
    audits={name:apply_repair(cfg,name)[1] for name in VARIANTS}
    assert plain(cfg)==baseline
    cfg,_=apply_repair(cfg,args.variant)
    save(args.output/'repair_audits.json',audits)
    cfg.seed=20261114;cfg.scene.num_envs=120;cfg.sim.device='cuda:0'
    resolve_lite3_asset(cfg)
    gen=cfg.scene.terrain.terrain_generator
    gen.num_rows=1;gen.use_cache=False;gen.seed=20261114
    for name in ('stairs_up','stairs_down'):
        gen.sub_terrains[name].step_height_range=(.05,.05)
    cfg.curriculum.terrain_levels=cfg.curriculum.gait_level=None
    cfg.commands.base_velocity.debug_vis=False
    cfg.observations.policy.enable_corruption=False
    for name in ('randomize_rigid_body_mass','randomize_rigid_body_inertia','randomize_com_positions','randomize_actuator_gains'):
        setattr(cfg.events,name,None)
    cfg.events.randomize_rigid_body_material.params.update(static_friction_range=(.9,.9),dynamic_friction_range=(.9,.9),
                                                          restitution_range=(0.,0.),make_consistent=True)
    cfg.events.randomize_reset_base.params['pose_range']={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.events.randomize_reset_base.params['velocity_range']={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    save(args.output/'config.json',dict(configuration=plain(cfg),variant=args.variant,
        overrides=['120 environments; one native 20-column row; exact 5 cm up/down test stairs',
                   'nominal physics, native actuator delay, no terrain or gait curriculum',
                   'fixed command groups; default PD preparation 1 s, then 12 s frozen probe'],
        optimizer_updates=0,training=False))
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    base=ManagerBasedRLEnv(cfg=cfg);obs,_=base.reset()
    robot=base.scene['robot'];sensor=base.scene['contact_forces'];command=base.command_manager.get_term('base_velocity')
    names=column_names(cfg)
    families=[names[int(col)] for col in base.scene.terrain.terrain_types.cpu()]
    assert set(FAMILIES[name] for name in families)=={'flat','rough','slope','stairs','friction','slope_rough'}
    target=torch.zeros(120,3,device=base.device);target[:,0]=.4
    cases=['forward']*120
    flat_ids=[i for i,name in enumerate(families) if name=='flat']
    assert len(flat_ids)==24
    flat_commands=[('low_left',(0.,0.,.2)),('low_right',(0.,0.,-.2)),
        ('turn_left',(0.,0.,.5)),('turn_right',(0.,0.,-.5)),
        ('fast_left',(0.,0.,.8)),('fast_right',(0.,0.,-.8)),
        ('arc_left',(.3,0.,.5)),('arc_right',(.3,0.,-.5)),
        ('standing',(0.,0.,0.)),('standing',(0.,0.,0.)),('forward',(.4,0.,0.)),('forward',(.4,0.,0.))]
    for k,i in enumerate(flat_ids):
        name,value=flat_commands[k//2];target[i]=target.new_tensor(value);cases[i]=name
    for i,name in enumerate(families):
        if name.startswith('stairs'):
            target[i,0]=(.32,.4,.6)[i%3]
    roots=robot.data.default_root_state.clone();roots[:,:3]+=base.scene.env_origins
    roots[:,0]+=.25;roots[:,7:]=0.
    robot.write_root_state_to_sim(roots)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_vel))
    base.sim.forward();base.scene.update(base.step_dt)
    tick=-1
    def set_command(next_tick):
        command.is_standing_env[:]=command.is_heading_env[:]=False
        command.vel_command_b[:]=target*min(max((next_tick*base.step_dt-.5)/.5,0.),1.) if next_tick>=0 else 0.
    native_command=base.command_manager.compute
    def command_compute(dt):
        native_command(dt);set_command(tick+1 if tick>=0 else -1)
    base.command_manager.compute=command_compute;set_command(-1)
    original.update_gait_level_from_terrain_mean(3.) # Explicit fixed unit scale for accounting, not historical replay.
    with torch.inference_mode():
        for _ in range(50):
            obs,_,done,trunc,_=base.step(torch.zeros(120,12,device=base.device))
            assert not (done|trunc).any()
    base.episode_length_buf[:]=0;base.action_manager.reset();tick=0;set_command(0)
    obs=base.observation_manager.compute()
    initial=dict(root=robot.data.root_state_w.cpu().numpy(),joint_pos=robot.data.joint_pos.cpu().numpy(),
        observation=obs['policy'].cpu().numpy(),origins=base.scene.env_origins.cpu().numpy(),
        materials=robot.root_physx_view.get_material_properties().cpu().numpy())
    np.savez_compressed(args.output/'initial.npz',**initial)
    checkpoint=TRAINING/'model_updates_10000.pt';assert sha(checkpoint)==EXPECTED_SHA
    MODELS[MODEL]=checkpoint;actor=load_actor(MODEL).to(base.device)
    digest=lambda:hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    before=digest();active=torch.ones(120,dtype=torch.bool,device=base.device)
    keys=('active','command','root','gravity','feet','force','wall_distance','wall_normal','wall_hit',
          'old_clearance','riser_cost','yaw_original','yaw_candidate','first_contact','last_air_time',
          'joint_pos','joint_torque','joint_velocity','action','support','terminated','truncated')
    arrays={k:[] for k in keys};errors=[]
    native_reward=base.reward_manager.compute
    reward_names=list(base.reward_manager.active_terms)
    yaw_term=base.reward_manager.get_term_cfg('feet_air_time_ang_z_lite3')
    foot_sensor_ids,foot_names=sensor.find_bodies(['FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT'],preserve_order=True)
    assert foot_names==['FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT']
    joint_limits=robot.data.soft_joint_pos_limits.clone()
    def capture(dt):
        total=native_reward(dt);rates=base.reward_manager._step_reward
        assert torch.allclose(total,rates.sum(-1)*dt,atol=1e-5,rtol=1e-4)
        x=sample(base);wall=wall_sample(base);cost=stair_riser_contact(base)
        yo=original.feet_air_time_ang_z_cmd_lite3(base,**yaw_term.params)*yaw_term.weight
        yn=yaw_airtime_without_short_step_penalty(base,**yaw_term.params)*yaw_term.weight
        pure=pure_yaw(command.command)
        assert (yn[pure]>=0.).all() and torch.equal(yn[~pure],yo[~pure])
        assert (cost>=0.).all() and (cost<=1.).all() and (cost[~x['enabled']]==0.).all()
        expected=yn if args.variant=='yaw_airtime' else yo
        error=(rates[:,reward_names.index('feet_air_time_ang_z_lite3')]-expected).abs().max()
        if args.variant=='riser_contact':
            error=torch.maximum(error,(rates[:,reward_names.index('stair_riser_contact')]+cost).abs().max())
        assert error<1e-6;errors.append(float(error))
        finite=torch.isfinite(wall['distance'])
        values=dict(active=active,command=command.command,root=robot.data.root_pos_w-base.scene.env_origins,
            gravity=robot.data.projected_gravity_b,feet=x['feet']-base.scene.env_origins[:,None],force=x['force'],
            wall_distance=torch.where(finite,wall['distance'],-1.),
            wall_normal=torch.where(finite[...,None],wall['normal'],0.),wall_hit=finite,
            old_clearance=rates[:,reward_names.index('stair_clearance')],riser_cost=cost,yaw_original=yo,yaw_candidate=yn,
            first_contact=sensor.compute_first_contact(dt)[:,foot_sensor_ids],last_air_time=sensor.data.last_air_time[:,foot_sensor_ids],
            joint_pos=robot.data.joint_pos,joint_torque=robot.data.applied_torque,joint_velocity=robot.data.joint_vel,
            action=base.action_manager.action,support=x['support'],terminated=base.reset_terminated,truncated=base.reset_time_outs)
        for k,v in values.items():
            assert torch.isfinite(v).all(),k
            arrays[k].append(v.cpu().numpy().copy())
        active.logical_and_(~(base.reset_terminated|base.reset_time_outs))
        return total
    base.reward_manager.compute=capture
    with torch.inference_mode():
        for tick in range(600):
            assert torch.equal(obs['policy'][:,6:9],command.command)
            obs,_,done,trunc,_=base.step(actor(obs['policy']).clamp(-100,100))
            assert obs['policy'].shape==(120,45) and obs['critic'].shape==(120,393)
            assert all(torch.isfinite(v).all() for v in obs.values())
            if (tick+1)%200==0:print(f'FROZEN REPAIR {args.variant} {tick+1}/600',flush=True)
    data={k:np.asarray(v) for k,v in arrays.items()}
    np.savez_compressed(args.output/'trace.npz',**data)
    valid=data['active'];pure=(np.linalg.norm(data['command'][...,:2],axis=-1)<.1)&(np.abs(data['command'][...,2])>.1)
    changed=(data['yaw_candidate']>data['yaw_original']+1e-6)&valid
    collisions=(data['riser_cost']>0)&valid
    assert collisions.any() and changed.any() and not (changed&~pure).any()
    assert digest()==before and sha(checkpoint)==EXPECTED_SHA
    save(args.output/'report.json',dict(status='passed',variant=args.variant,environments=120,steps=600,
        optimizer_updates=0,checkpoint_sha256=EXPECTED_SHA,actor_unchanged=True,observations=[45,393],
        exact_reward_max_error=max(errors),riser_cost_active_frames=int(collisions.sum()),
        yaw_repaired_frames=int(changed.sum()),yaw_outside_pure_turn_changed_frames=int((changed&~pure).sum()),
        first_episode_failures=int((~active).sum()),cases=cases,terrain_names=families,
        joint_names=robot.joint_names,soft_joint_limits=joint_limits[0].cpu().tolist(),
        fixed_gait_scale=1.,sources=source_manifest(),
        note='Configuration/reward integration only. Frozen weights cannot establish training improvement; native reset is excluded from first-episode counts.'))
    print('FROZEN REPAIR PASSED; no optimizer constructed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('riser_contact','yaw_airtime'),required=True)
    parser.add_argument('--gpu',type=int,choices=(1,),default=1)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=args.gpu,output=args.output,timeout=1800)
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        collect(args);sys.stdout.flush();os._exit(0)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__':sys.exit(main())
