"""已批准的 480 回合奖励旁路补录；原轨迹必须逐字段完全相同。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_eval_checkpoint import register_checkpoint
from reference_low_stairs_heading_probe import CHECKPOINT, DIGEST, MODEL, PROTOCOL, PROTOCOL_SHA, attach_feedback
from reference_low_stairs_heading_probe import OUTPUT as HEADING
from reference_low_stairs_heading_evaluate import configuration_equal
from reference_low_stairs_zero_preservation_evaluate import EVALUATION, checked_training, read, verify_run
from reference_core500_evaluate import same_initial
from reference_low_stairs_equal100_analyze import heading_error, validate_trace
from reference_low_stairs_analyze import check_log

OUTPUT=ROOT/'logs/reference_six/low_stairs_reward_state_20261116'
DOCS=ROOT/'docs/lite3-low-stairs-reward-state-20261116'
BASELINES={'nominal':EVALUATION/'heading_up_nominal',
    'training_randomized':HEADING/'heading_up_training_randomized'}
REWARD_FILES=('reference_low_stairs_rewards.py','reference_low_stairs_clearance.py',
    'reference_low_stairs_cfg.py','reference_low_stairs_clearance_cfg.py','reference_low_stairs_guarded_cfg.py')


def sources():
    names=REWARD_FILES+('reference_low_stairs_reward_observer.py','reference_low_stairs_reward_state_probe.py',
        'reference_low_stairs_heading_probe.py','reference_isaac_stairs_diagnose.py')
    return [dict(path=str(Path(__file__).with_name(name)),sha256=sha(Path(__file__).with_name(name))) for name in names]


def exact_replay(path,baseline):
    assert read(path/'report.json')==read(baseline/'report.json'), '补录报告改变'
    assert configuration_equal(read(path/'config.json'),read(baseline/'config.json')), '补录配置改变'
    assert same_initial(path,baseline), '补录初始数组改变'
    checks={}
    with np.load(path/'trace.npz') as a,np.load(baseline/'trace.npz') as b:
        assert a.files==b.files
        for k in a.files:checks[k]=bool(np.array_equal(a[k],b[k]))
    assert all(checks.values()), checks
    return dict(report_exact=True,configuration_exact=True,initial_exact=True,trace_exact=checks)


def collect(args,app):
    import torch
    import isaaclab.envs as envs
    import reference_isaac_stairs_diagnose as protocol
    from reference_low_stairs_rewards import sample
    from reference_low_stairs_clearance import clearance_sample
    from reference_low_stairs_reward_observer import RewardObserver,LANDING_RULES,state_copy
    native_env=envs.ManagerBasedRLEnv
    arrays,preparation={},{}
    step_count=0

    def append(target,values):
        for k,v in values.items():
            a=v.detach().cpu().numpy().copy()
            if k not in ('wall_distance','wall_normal'):assert np.isfinite(a).all(),k
            target.setdefault(k,[]).append(a)

    def instrument(base):
        robot,sensor=base.scene['robot'],base.scene['contact_forces']
        names=[f'{leg}_FOOT' for leg in ('FL','FR','HL','HR')]
        feet=[robot.body_names.index(name) for name in names]
        contacts=[sensor.body_names.index(name) for name in names]
        subs=base.cfg.scene.terrain.terrain_generator.sub_terrains
        assert list(subs)==['flat','stairs_02cm','stairs_03cm','stairs_05cm']
        assert base.num_envs==240 and base.step_dt==.02
        base._low_stairs_bindings=(feet,contacts,torch.tensor([False,True,True,True],device=base.device))
        observer=RewardObserver(base.num_envs,base.device)
        original_reward,original_reset,original_step=base.reward_manager.compute,base._reset_idx,base.step

        def reset(ids):
            result=original_reset(ids)
            observer.reset(ids)
            for key in ('_low_stairs_sample','_persistent_clearance_sample'):
                if hasattr(base,key):delattr(base,key)
            return result

        def capture(dt):
            value=original_reward(dt)
            x=sample(base);walls=clearance_sample(base)
            values=observer.observe(x,walls,dt,base.common_step_counter)
            values.update(command=base.command_manager.get_command('base_velocity'),
                gravity=robot.data.projected_gravity_b,episode_length=base.episode_length_buf,
                tick=torch.full((base.num_envs,),base.common_step_counter,device=base.device,dtype=torch.long),
                terminated=base.reset_terminated,truncated=base.reset_time_outs,
                current_contact_time=sensor.data.current_contact_time[:,contacts],
                current_air_time=sensor.data.current_air_time[:,contacts])
            append(preparation if step_count<50 else arrays,values)
            return value

        def step(action):
            nonlocal step_count
            if step_count==50:
                values=state_copy(observer.clearance,'clear_') | state_copy(observer.landing,'land_')
                np.savez_compressed(args.output/'shadow_initial.npz',**{k:v.cpu().numpy() for k,v in values.items()})
                save(args.output/'shadow_schema.json',dict(landing_rules=LANDING_RULES,
                    coordinates='world',sidecar_only=True,initial_preparation_steps=50,
                    columns=['flat','stairs','stairs','stairs'],foot_names=names,step_dt=.02,
                    clearance_weight=-1.,landing_weight=1.,progress_weight=1.,
                    timing='after original reward, before reset; preparation state retained; native reset synchronized',
                    wall_nonfinite_values='ray misses are retained; finite distance and normal define valid hits',
                    sources=sources()))
            assert step_count<1050, '超过固定补录步数'
            result=original_step(action);step_count+=1
            return result
        base._reset_idx=reset;base.reward_manager.compute=capture;base.step=step

    class ObservedEnv(native_env):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            attach_feedback(self)
            instrument(self)

    envs.ManagerBasedRLEnv=ObservedEnv
    try:protocol.collect(args,app)
    finally:envs.ManagerBasedRLEnv=native_env
    report=read(args.output/'report.json')
    assert step_count==report['steps']+50 and len(preparation['bottom'])==50
    assert len(arrays['bottom'])==report['steps']
    np.savez_compressed(args.output/'reward_state.npz',**{k:np.asarray(v) for k,v in arrays.items()})
    np.savez_compressed(args.output/'preparation_state.npz',**{k:np.asarray(v) for k,v in preparation.items()})
    save(args.output/'capture_complete.json',dict(status='passed',episodes=240,steps=report['steps'],
        optimizer_updates=0,checkpoint_sha256=DIGEST,sources=sources(),sidecar_only=True,
        scope='旁路数据已保存，逐帧一致性在父进程核验'))


def verify(path,profile):
    verify_run(path,'stairs',profile,DIGEST,MODEL,'up')
    cap=read(path/'capture_complete.json')
    assert cap['sources']==sources() and cap['checkpoint_sha256']==DIGEST and cap['optimizer_updates']==0
    result=exact_replay(path,BASELINES[profile])
    report=read(path/'report.json')
    with np.load(path/'trace.npz') as archive:data={k:archive[k] for k in archive.files}
    validate_trace(report,data)
    error=heading_error(path,data)
    with np.load(path/'reward_state.npz') as obs,np.load(path/'initial.npz') as initial:
        assert obs['bottom'].shape==(report['steps'],240,4)
        assert np.array_equal(obs['command'],data['command']) and np.array_equal(obs['force'],data['forces'])
        assert np.array_equal(obs['gravity'],data['gravity'])
        assert np.array_equal(obs['feet']-initial['origins'][None,:,None],data['feet'])
        assert np.array_equal(obs['terminated'],data['terminated']) and np.array_equal(obs['truncated'],data['truncated'])
        assert all(np.isfinite(obs[k]).all() for k in obs.files if k not in ('wall_distance','wall_normal'))
        assert np.all(obs['clearance_rate'][data['active']]<=0.)
    assert sha(CHECKPOINT)==DIGEST and sha(PROTOCOL)==PROTOCOL_SHA
    return dict(status='passed',profile=profile,path=str(path),baseline=str(BASELINES[profile]),episodes=240,
        optimizer_updates=0,feedback_max_error=error,**result,**check_log(path))


def matrix():
    checkpoint,digest=checked_training()
    assert checkpoint==CHECKPOINT and digest==DIGEST and sha(PROTOCOL)==PROTOCOL_SHA
    historic={r['path']:r['sha256'] for r in read(ROOT/'docs/lite3-low-stairs-stability-priority-20261116/raw_manifest.json')}
    reused=[]
    for p in BASELINES.values():
        for f in sorted(p.iterdir()):
            if f.is_file():
                assert sha(f)==historic[str(f)]
                reused.append(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size))
    cfg=read(CHECKPOINT.parent/'configuration.json')['environment']['rewards']
    expected={'stair_clearance':('reference_low_stairs_clearance_cfg:StairClearanceReward',-1.),
        'stair_progress':('reference_low_stairs_rewards:stair_progress',1.),
        'stair_landing':('reference_low_stairs_cfg:StairLandingReward',1.)}
    for key,(func,weight) in expected.items():assert cfg[key]==dict(func=func,params={},weight=weight)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    save(OUTPUT/'plan.json',dict(new_episodes=480,optimizer_updates=0,physical_gpu=1,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=DIGEST,training_terms={k:cfg[k] for k in expected},
        sources=sources(),automatic_extension=False,reused=reused))
    jobs=[]
    for profile in BASELINES:
        dest=OUTPUT/profile
        if not dest.exists():
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--profile',profile],cwd=ROOT,check=True)
        result=verify(dest,profile);jobs.append(result)
        save(OUTPUT/'matrix.json',jobs)
        print('VERIFIED EXACT REPLAY',profile,flush=True)
    assert all(sha(r['path'])==r['sha256'] for r in reused)
    save(OUTPUT/'complete.json',dict(status='passed',new_episodes=480,optimizer_updates=0,matrix=jobs,
        checkpoint_sha256=DIGEST,sources=sources(),automatic_extension=False))
    print('REWARD STATE CAPTURE COMPLETE: 480 episodes',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',action='store_true')
    parser.add_argument('--profile',choices=tuple(BASELINES))
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    if args.matrix:
        assert not args.child and args.profile is None
        return matrix()
    if args.profile is None:parser.error('需要指定 profile')
    assert sha(CHECKPOINT)==DIGEST and sha(PROTOCOL)==PROTOCOL_SHA
    args.model,args.direction,args.action_mode=MODEL,'up','mean'
    args.height_set,args.repeats,args.steps,args.video_env='low_stairs',20,None,None
    args.output=OUTPUT/args.profile
    register_checkpoint(MODEL,CHECKPOINT,DIGEST)
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
            gpu=1,output=args.output,minimum_free_mib=10240,timeout=2400)
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cuda:0',kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        collect(args,app);sys.stdout.flush();os._exit(0)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__':sys.exit(main())
