"""已批准的四组冻结转向补录；只读原状态，原协议和奖励代码不变。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_eval_checkpoint import register_checkpoint
from reference_low_stairs_zero_preservation_evaluate import (
    checked_training, EVALUATION, PARENT_RUNS, GUARDED_RUNS, MODEL, verify_run, read)
from reference_low_stairs_evaluate import EXPECTED_SHA
from reference_low_stairs_trial import CHECKPOINT
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration

OUTPUT=ROOT/'logs/reference_six/low_stairs_motion_diagnosis_20261115'
DOCS=ROOT/'docs/lite3-low-stairs-motion-diagnosis-20261115'
PROTOCOL=ROOT/'scripts/tools/reference_turn_diagnose.py'
PROTOCOL_SHA='e53803a8f3f4716e7752cb1b0c1afb79938a072d60c92ff021907f3f9c780337'
ZERO_SHA='290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497'
LEGS=('FL','FR','HL','HR')


def identity(policy):
    if policy=='parent':return CHECKPOINT,EXPECTED_SHA,'lowstairs10000',PARENT_RUNS
    if policy!='zero':raise ValueError(policy)
    checkpoint,digest=checked_training()
    assert digest==ZERO_SHA
    return checkpoint,digest,MODEL,EVALUATION


def exact_replay(path,baseline):
    a,b=read(path/'report.json'),read(baseline/'report.json')
    assert a==b,'报告和终止结果必须逐项一致'
    config_a,config_b=read(path/'config.json'),read(baseline/'config.json')
    sources_a,sources_b=(c.pop('sources',[]) for c in (config_a,config_b))
    assert config_a==config_b,'有效配置必须完全一致'
    source_a,source_b=({r['path']:r['sha256'] for r in rows} for rows in (sources_a,sources_b))
    assert all(source_a.get(k)==v and sha(k)==v for k,v in source_b.items())
    extra=set(source_a)-set(source_b)
    watcher=ROOT/'scripts/tools/reference_six_watch.py'
    assert extra<={str(watcher)}
    assert all(source_a[k]==sha(k)=='21060b75a6b34cb49c29cb868b5fb18cede274c4481e3e13616e72b6ddb049ad' for k in extra)
    assert same_initial(path,baseline),'初始数组必须完全一致'
    checks={}
    with np.load(path/'trace.npz') as left,np.load(baseline/'trace.npz') as right:
        assert left.files==right.files
        for key in left.files:
            checks[key]=bool(np.array_equal(left[key],right[key]))
        assert all(checks.values()),checks
    return dict(status='passed',report_exact=True,effective_config_exact=True,initial_exact=True,
                extra_readonly_sources=[dict(path=k,sha256=source_a[k]) for k in sorted(extra)],
                trace_exact=checks,baseline=str(baseline),optimizer_updates=0)


def collect(args,app):
    import torch
    import isaaclab.envs as envs
    import reference_turn_diagnose as protocol
    native_env,native_loader=envs.ManagerBasedRLEnv,protocol.load_actor
    arrays,preparation,observations={}, {}, []
    state={'steps':0}
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
            value=native_reward(dt)
            data,contact=robot.data,sensor.data
            values=dict(root=data.root_state_w,gravity=data.projected_gravity_b,
                joint_pos=data.joint_pos,joint_vel=data.joint_vel,joint_target=data.joint_pos_target,
                torque=data.applied_torque,feet=data.body_pos_w[:,feet]-base.scene.env_origins[:,None],
                foot_velocity=data.body_lin_vel_w[:,feet],force=contact.net_forces_w[:,contacts],
                air_time=contact.current_air_time[:,contacts],contact_time=contact.current_contact_time[:,contacts],
                last_air_time=contact.last_air_time[:,contacts],last_contact_time=contact.last_contact_time[:,contacts],
                episode_length=base.episode_length_buf,command=base.command_manager.get_command('base_velocity'),
                terminated=base.reset_terminated,truncated=base.reset_time_outs)
            append(preparation if state['steps']<50 else arrays,values)
            return value
        def step(action):
            if state['steps']==50:
                values=dict(masses=robot.root_physx_view.get_masses(),inertias=robot.root_physx_view.get_inertias(),
                    coms=robot.root_physx_view.get_coms(),default_joint_pos=robot.data.default_joint_pos,
                    effort_limits=robot.data.joint_effort_limits,velocity_limits=robot.data.joint_vel_limits,
                    origins=base.scene.env_origins)
                for name,actuator in robot.actuators.items():
                    values[f'delay_{name}']=actuator.positions_delay_buffer.time_lags
                    values[f'pd_stiffness_{name}']=actuator.stiffness
                    values[f'pd_damping_{name}']=actuator.damping
                np.savez_compressed(args.output/'physics_initial.npz',**snap(values))
                from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards
                save(args.output/'names.json',dict(legs=LEGS,joint_names=robot.joint_names,
                    action_joint_ids=base.action_manager.get_term('joint_pos')._joint_ids,
                    phase_body_ids=base.reward_manager.get_term_cfg('phase_foot_trajectory_exp').params['asset_cfg'].body_ids,
                    feet_body_ids=feet,contact_body_ids=contacts,
                    gait_level=float(rewards.gait_level),sensor_force_threshold=float(sensor.cfg.force_threshold),
                    state_timing='post reward, pre reset; phase reference and counters reconstructed offline',
                    wrapper_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL)))
            assert state['steps']<1050,'超过固定补录步数'
            result=native_step(action);state['steps']+=1
            return result
        base.step=step;base.reward_manager.compute=capture

    class ObservedEnv(native_env):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);instrument(self)

    def loader(name):
        actor=native_loader(name)
        def capture_input(module,inputs):observations.append(inputs[0].detach().cpu().numpy().copy())
        actor.register_forward_pre_hook(capture_input)
        return actor
    envs.ManagerBasedRLEnv=ObservedEnv;protocol.load_actor=loader
    try:protocol.collect(args,app)
    finally:envs.ManagerBasedRLEnv=native_env;protocol.load_actor=native_loader
    assert state['steps']==1050 and len(observations)==len(arrays['root'])==1000
    assert len(preparation['root'])==50
    np.savez_compressed(args.output/'motion.npz',**{k:np.asarray(v) for k,v in arrays.items()},observation=np.asarray(observations))
    np.savez_compressed(args.output/'preparation.npz',**{k:np.asarray(v) for k,v in preparation.items()})
    checkpoint,digest,_,baseline=identity(args.policy)
    assert sha(checkpoint)==digest and sha(PROTOCOL)==PROTOCOL_SHA
    result=exact_replay(args.output,baseline/f'turn_{args.profile}')
    save(args.output/'motion_report.json',dict(**result,steps=1000,episodes=320,preparation_steps=50,
        diagnostic_only=True,training=False,policy=args.policy,profile=args.profile,
        checkpoint=str(checkpoint),checkpoint_sha256=digest,source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL)))


def matrix():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    OUTPUT.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for policy in ('parent','zero'):
        checkpoint,digest,model,baseline=identity(policy)
        for profile in ('nominal','training_randomized'):
            dest=OUTPUT/f'{policy}_{profile}'
            verify_run(baseline/f'turn_{profile}','turn',profile,digest,model)
            if not dest.exists():
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--policy',policy,'--profile',profile],cwd=ROOT,check=True)
            execution=read(dest/'execution.json')
            if execution['status']=='failed':
                # 首组已完成 320 回合，只因来源清单多出只读模块而核验退出。
                # 保留 execution/error 原文件；用修正后的核验器验证全量已有数据，不重跑。
                assert policy=='parent' and profile=='nominal' and execution['exit_code']==1
                assert execution['failure'] is None and not execution['fatal_messages']
                assert execution['source_commit']=='75262dd8dc8d4205943805b550d7b496584a7770'
                assert read(dest/'error.json')['error'].endswith('AssertionError: 配置必须完全一致\n')
                result=exact_replay(dest,baseline/f'turn_{profile}')
                with np.load(dest/'motion.npz') as data:
                    assert all(np.isfinite(data[k]).all() for k in data.files)
                    assert data['root'].shape==(1000,320,13) and data['observation'].shape==(1000,320,45)
                import hashlib
                old_source=subprocess.check_output(['git','show',execution['source_commit']+':scripts/tools/reference_low_stairs_motion_probe.py'],cwd=ROOT)
                report=dict(**result,steps=1000,episodes=320,preparation_steps=50,diagnostic_only=True,
                    training=False,policy=policy,profile=profile,checkpoint=str(checkpoint),checkpoint_sha256=digest,
                    source_sha256=hashlib.sha256(old_source).hexdigest(),verification_source_sha256=sha(__file__),
                    protocol_sha256=sha(PROTOCOL),completion_type='capture_completed_verifier_corrected',
                    note='仿真已完整完成；原退出由额外只读来源条目引起，保留原失败文件，未重跑。')
                save(dest/'motion_report.json',report)
            else:
                assert execution['status']=='completed' and execution['exit_code']==0
                report=read(dest/'motion_report.json');assert report['source_sha256']==sha(__file__)
            assert execution['physical_gpu']==1 and report['status']=='passed' and report['optimizer_updates']==0
            exact_replay(dest,baseline/f'turn_{profile}')
            jobs.append(dict(policy=policy,profile=profile,path=str(dest),baseline=str(baseline/f'turn_{profile}'),episodes=320))
            save(OUTPUT/'matrix.json',jobs)
    assert sum(x['episodes'] for x in jobs)==1280
    save(OUTPUT/'complete.json',dict(status='passed',episodes=1280,optimizer_updates=0,matrix=jobs,source_sha256=sha(__file__)))
    print('MOTION CAPTURE COMPLETE: 1280 episodes, all original arrays exact',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',action='store_true');parser.add_argument('--child',action='store_true')
    parser.add_argument('--policy',choices=('parent','zero'));parser.add_argument('--profile',choices=('nominal','training_randomized'))
    args=parser.parse_args()
    if args.matrix:
        assert not args.child and args.policy is None and args.profile is None
        return matrix()
    assert args.policy and args.profile and sha(PROTOCOL)==PROTOCOL_SHA
    checkpoint,digest,args.model,_=identity(args.policy)
    register_checkpoint(args.model,checkpoint,digest)
    args.output=OUTPUT/f'{args.policy}_{args.profile}'
    args.repeats,args.steps,args.video_env,args.gpu=20,None,None,1
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
            gpu=1,output=args.output,timeout=2400,minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cuda:0',kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        collect(args,app);sys.stdout.flush();os._exit(0)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__':sys.exit(main())
