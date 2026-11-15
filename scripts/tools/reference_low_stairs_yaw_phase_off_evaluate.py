"""核验固定预算训练，六次原协议仿真中同步补录运动数据，共 1600 回合。"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace as NS

import numpy as np

from reference_low_stairs_yaw_phase_off_train import (
    ROOT,OUTPUT,DOCS,ZERO_OUTPUT,VARIANT,NATIVE_SHA,directory,sources,preflight,expected_configuration,read,save,sha)
from reference_low_stairs_evaluate import OUTPUT as PARENT_RUNS, EXPECTED_SHA
from reference_low_stairs_guarded_bank import OUTPUT as GUARDED_OUTPUT
from reference_low_stairs_zero_preservation_evaluate import verify_run
from reference_low_stairs_trial import CHECKPOINT
from reference_low_stairs_clearance_train import check_algorithm
from reference_low_stairs_resume import STATE_KEY
from reference_low_stairs_guarded_ppo import KEY
from reference_reward_resume import equal_state
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_eval_checkpoint import register_checkpoint

EVALUATION=OUTPUT/'evaluation_10100'
ZERO_RUNS=ZERO_OUTPUT/'evaluation_10100'
GUARDED_RUNS=GUARDED_OUTPUT/'evaluation_10100'
MODEL='lowstairs_yaw_phase_off10100'
PROTOCOLS={
    'turn':(ROOT/'scripts/tools/reference_turn_diagnose.py','e53803a8f3f4716e7752cb1b0c1afb79938a072d60c92ff021907f3f9c780337'),
    'stairs':(ROOT/'scripts/tools/reference_isaac_stairs_diagnose.py','c155ab8d2050e53eaabda24cb00bee91092a97b8b60a4303638c1a363157bab9')}
LEGS=('FL','FR','HL','HR')
JOBS=[(f'turn_{p}','turn',p,None) for p in ('nominal','training_randomized')]+[
    (f'{d}_{p}','stairs',p,d) for p in ('nominal','training_randomized') for d in ('up','down')]


def checked_training():
    import torch
    preflight(1);run=directory(1);r=read(run/'report.json');execution=read(run/'execution.json')
    assert execution['status']=='completed' and execution['exit_code']==0 and execution['physical_gpu']==1
    assert r['status']=='passed' and r['max_updates']==r['optimizer_updates']==r['phase_updates']==100
    assert r['cumulative_updates']==10100 and r['environment_steps']==2400 and r['interactions']==9830400
    assert r['variant']==VARIANT and r['auxiliary_coefficient']==0. and not r['automatic_extension']
    assert r['parent_sha256']==r['original_parent_sha256']==EXPECTED_SHA
    assert r['implementation_sources']==sources() and r['native_trainer_sha256']==NATIVE_SHA
    assert [v['update'] for v in r['updates']]==list(range(1,101))
    assert all(v['losses']['capability_preservation']==0. for v in r['updates'])
    assert read(run/'configuration.json')==expected_configuration()
    audit=r['phase_gate_audit'];assert audit['steps']==2400 and audit['frames']==9830400
    assert audit['removed_nonzero_frames']>0 and audit['pure_yaw_frames']>=audit['removed_nonzero_frames']
    assert audit['non_yaw_exact'] and audit['reference_initialized']
    parent=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    zero=torch.load(ZERO_OUTPUT/'train_001_100/model_updates_10000.pt',map_location='cpu',weights_only=True)
    assert [v['phase_updates'] for v in r['checkpoints']]==[0,50,100]
    initial=None
    for row in r['checkpoints']:
        path=Path(row['path']);n=row['phase_updates'];assert sha(path)==row['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True);check_algorithm(state,n);info=state['infos']
        assert state['iter']==9999+n and info['completed_updates']==10000+n and info['phase_updates']==n
        assert info['variant']==VARIANT and info['trial_budget']==100 and info['parent_sha256']==EXPECTED_SHA
        assert info['validation_only']==(n==0) and info[KEY]['coefficient']==0. and info[KEY]['calls']==n*20
        assert info[KEY]['bank_sha256']==r['bank_sha256'] and info[KEY]['per_group']==128
        curriculum=info[STATE_KEY];levels=curriculum['terrain_levels'];types=curriculum['terrain_types']
        assert levels.shape==types.shape==(4096,) and levels.min()>=0 and levels.max()<10
        assert equal_state(curriculum['env_origins'],curriculum['terrain_origins'][levels,types])
        if n==0:
            initial=state
            assert all(equal_state(state[k],parent[k]) and equal_state(state[k],zero[k]) for k in
                       ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
            assert equal_state(curriculum,zero['infos'][STATE_KEY]) and info[KEY]==zero['infos'][KEY]
        else:
            assert all(not equal_state(state[k],initial[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
    assert path==Path(r['output_checkpoint']) and sha(path)==r['output_sha256'] and sha(CHECKPOINT)==EXPECTED_SHA
    return path,r['output_sha256']


def protocol_equal(left,right):
    left,right=copy.deepcopy(left),copy.deepcopy(right)
    sources_left,sources_right=(c.pop('sources',[]) for c in (left,right))
    assert same_configuration(left,right),'有效评估配置发生变化'
    a,b=({r['path']:r['sha256'] for r in rows} for rows in (sources_left,sources_right))
    assert all(a.get(k)==v and sha(k)==v for k,v in b.items())
    watcher=str(ROOT/'scripts/tools/reference_six_watch.py')
    extra=set(a)-set(b);assert extra<={watcher}
    assert all(a[k]==sha(k)=='21060b75a6b34cb49c29cb868b5fb18cede274c4481e3e13616e72b6ddb049ad' for k in extra)
    return dict(effective_config_exact=True,extra_readonly_sources=sorted(extra))


def collect(args,app):
    import isaaclab.envs as envs
    if args.kind=='turn':import reference_turn_diagnose as protocol
    else:import reference_isaac_stairs_diagnose as protocol
    native_env,native_loader=envs.ManagerBasedRLEnv,protocol.load_actor
    arrays,preparation,observations={}, {}, [];state={'steps':0}
    def snap(values):return {k:v.detach().cpu().numpy().copy() for k,v in values.items()}
    def append(target,values):
        for k,a in snap(values).items():
            assert np.isfinite(a).all(),k
            target.setdefault(k,[]).append(a)
    def instrument(base):
        robot,sensor=base.scene['robot'],base.scene['contact_forces']
        feet=[robot.body_names.index(f'{leg}_FOOT') for leg in LEGS]
        contacts=[sensor.body_names.index(f'{leg}_FOOT') for leg in LEGS]
        native_step,native_reward=base.step,base.reward_manager.compute
        def capture(dt):
            value=native_reward(dt);data,contact=robot.data,sensor.data
            append(preparation if state['steps']<50 else arrays,dict(
                root=data.root_state_w,gravity=data.projected_gravity_b,joint_pos=data.joint_pos,joint_vel=data.joint_vel,
                joint_target=data.joint_pos_target,torque=data.applied_torque,
                feet=data.body_pos_w[:,feet]-base.scene.env_origins[:,None],foot_velocity=data.body_lin_vel_w[:,feet],
                force=contact.net_forces_w[:,contacts],air_time=contact.current_air_time[:,contacts],
                contact_time=contact.current_contact_time[:,contacts],last_air_time=contact.last_air_time[:,contacts],
                last_contact_time=contact.last_contact_time[:,contacts],episode_length=base.episode_length_buf,
                command=base.command_manager.get_command('base_velocity'),terminated=base.reset_terminated,truncated=base.reset_time_outs))
            return value
        def step(action):
            if state['steps']==50:
                values=dict(masses=robot.root_physx_view.get_masses(),inertias=robot.root_physx_view.get_inertias(),
                    coms=robot.root_physx_view.get_coms(),default_joint_pos=robot.data.default_joint_pos,
                    effort_limits=robot.data.joint_effort_limits,velocity_limits=robot.data.joint_vel_limits,origins=base.scene.env_origins)
                for name,actuator in robot.actuators.items():
                    values[f'delay_{name}']=actuator.positions_delay_buffer.time_lags
                    values[f'pd_stiffness_{name}']=actuator.stiffness;values[f'pd_damping_{name}']=actuator.damping
                np.savez_compressed(args.output/'physics_initial.npz',**snap(values))
                from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards
                save(args.output/'names.json',dict(legs=LEGS,joint_names=robot.joint_names,
                    action_joint_ids=base.action_manager.get_term('joint_pos')._joint_ids,
                    phase_body_ids=base.reward_manager.get_term_cfg('phase_foot_trajectory_exp').params['asset_cfg'].body_ids,
                    feet_body_ids=feet,contact_body_ids=contacts,gait_level=float(rewards.gait_level),
                    sensor_force_threshold=float(sensor.cfg.force_threshold),
                    state_timing='post reward, pre reset; actor input copied before step',
                    wrapper_sha256=sha(__file__),protocol_sha256=sha(PROTOCOLS[args.kind][0])))
            assert state['steps']<1050
            result=native_step(action);state['steps']+=1;return result
        base.step=step;base.reward_manager.compute=capture
    class ObservedEnv(native_env):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);instrument(self)
    def loader(name):
        actor=native_loader(name)
        actor.register_forward_pre_hook(lambda module,inputs:observations.append(inputs[0].detach().cpu().numpy().copy()))
        return actor
    envs.ManagerBasedRLEnv=ObservedEnv;protocol.load_actor=loader
    try:protocol.collect(args,app)
    finally:envs.ManagerBasedRLEnv=native_env;protocol.load_actor=native_loader
    report=read(args.output/'report.json');steps=report['steps'];n=report['episodes']
    assert state['steps']==50+steps and len(observations)==len(arrays['root'])==steps<=1000
    assert len(preparation['root'])==50
    m={k:np.asarray(v) for k,v in arrays.items()};m['observation']=np.asarray(observations)
    assert m['observation'].shape==(steps,n,45)
    with np.load(args.output/'initial.npz') as initial:assert np.array_equal(m['observation'][0],initial['observation'])
    with np.load(args.output/'trace.npz') as trace:
        for key in ('command','gravity','terminated','truncated'):assert np.array_equal(m[key],trace[key])
        assert np.array_equal(m['observation'][...,6:9],trace['command'])
    np.savez_compressed(args.output/'motion.npz',**m)
    np.savez_compressed(args.output/'preparation.npz',**{k:np.asarray(v) for k,v in preparation.items()})
    save(args.output/'motion_report.json',dict(status='passed',steps=steps,episodes=n,preparation_steps=50,
        optimizer_updates=0,protocol_sha256=sha(PROTOCOLS[args.kind][0]),source_sha256=sha(__file__),
        checkpoint_sha256=args.checkpoint_sha256,observation_and_trace_aligned=True,pre_reset_capture=True))


def matrix():
    checkpoint,digest=checked_training()
    for path,expected in PROTOCOLS.values():assert sha(path)==expected
    EVALUATION.mkdir(parents=True,exist_ok=True)
    save(EVALUATION/'plan.json',dict(model=MODEL,checkpoint=str(checkpoint),sha256=digest,
        formal_episodes=1600,heading_episodes=0,optimizer_updates=0,source_sha256=sha(__file__)))
    rows=[]
    for name,kind,profile,direction in JOBS:
        dest=EVALUATION/name
        if not dest.exists():
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',name],cwd=ROOT,check=True)
        report=verify_run(dest,kind,profile,digest,MODEL,direction)
        assert read(dest/'motion_report.json')['source_sha256']==sha(__file__)
        for old in (PARENT_RUNS/name,ZERO_RUNS/name,GUARDED_RUNS/name):
            assert same_initial(dest,old),'初始数组不一致'
            protocol_equal(read(dest/'config.json'),read(old/'config.json'))
        assert sha(checkpoint)==digest
        rows.append(dict(name=name,kind=kind,profile=profile,direction=direction,episodes=report['episodes'],
            candidate=str(dest),parent=str(PARENT_RUNS/name),zero=str(ZERO_RUNS/name),guarded=str(GUARDED_RUNS/name)))
        save(EVALUATION/'matrix.json',rows)
    assert sum(r['episodes'] for r in rows)==1600
    save(EVALUATION/'complete.json',dict(status='passed',scope='运行和来源完整性；能力筛查另行分析',
        formal_episodes=1600,heading_episodes=0,optimizer_updates=0,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        matrix=rows,source_sha256=sha(__file__)))
    print('YAW PHASE OFF EVALUATION COMPLETE: 1600 frozen episodes',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',choices=[r[0] for r in JOBS]);parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    if args.job is None:return matrix()
    name,args.kind,args.profile,args.direction=next(row for row in JOBS if row[0]==args.job)
    args.output=EVALUATION/name
    if not args.child:
        checked_training()
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=1,output=args.output,timeout=2400,minimum_free_mib=10240)
    try:
        checkpoint,digest=checked_training();args.checkpoint_sha256=digest
        for path,expected in PROTOCOLS.values():assert sha(path)==expected
        args.model=MODEL;args.repeats=20;args.steps=None;args.video_env=None;args.gpu=1
        args.height_set='low_stairs';args.action_mode='mean'
        register_checkpoint(MODEL,checkpoint,digest)
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cuda:0',kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        collect(args,app);sys.stdout.flush();os._exit(0)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__':sys.exit(main())
