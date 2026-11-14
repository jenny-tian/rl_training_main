"""Explicitly bounded A-only continuation of the reviewed low-stair checkpoint.

The original scratch-only entry point and historical training sources remain
unchanged. No high-stair checkpoint or variant B can enter through this CLI.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import ROOT, REFERENCE, load_reference, resolve_lite3_asset, source_manifest
from reference_low_stairs_evaluate import TRAINING, EXPECTED_SHA, verify_training
from reference_reward_resume import equal_state
from reference_stairs_eval import sha
from reference_six_watch import save

CHECKPOINT=TRAINING/'model_updates_10000.pt'
VARIANT='riser_contact'


def checked_state(device='cpu'):
    import torch
    if sha(CHECKPOINT)!=EXPECTED_SHA:
        raise ValueError('reviewed low-stair checkpoint identity changed')
    state=torch.load(CHECKPOINT,map_location=device,weights_only=True)
    info=state['infos']
    if (state['iter']!=9999 or info['completed_updates']!=10000 or info['phase_updates']!=10000
            or info['terrain_profile']!='low_stairs' or info['reward_variant']!='low_stairs'
            or info['validation_only'] or not info['from_scratch']):
        raise ValueError('checkpoint is not the approved completed low-stair run')
    if state['actor_state_dict']['mlp.0.weight'].shape!=(512,45) or state['critic_state_dict']['mlp.0.weight'].shape!=(512,393):
        raise ValueError('checkpoint observation dimensions differ')
    optimizer=state['optimizer_state_dict']
    rates={group['lr'] for group in optimizer['param_groups']}
    if {int(v['step']) for v in optimizer['state'].values()}!={200000} or len(rates)!=1:
        raise ValueError('checkpoint optimizer state differs')
    if not all(torch.isfinite(v).all() for key in ('actor_state_dict','critic_state_dict') for v in state[key].values()):
        raise ValueError('checkpoint parameters are not finite')
    return state,next(iter(rates))


def restore(runner):
    state,rate=checked_state(runner.device)
    runner.alg.load(state,load_cfg=None,strict=True)
    runner.alg.learning_rate=rate
    runner.current_learning_iteration=10000
    actual=runner.alg.save()
    for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):
        if not equal_state(state[key],actual[key]):
            raise ValueError(f'inexact low-stair restore: {key}')
    return dict(status='passed',checkpoint=str(CHECKPOINT),sha256=EXPECTED_SHA,
        completed_updates=10000,next_iteration=10000,optimizer_steps=200000,learning_rate=rate,
        actor_exact=True,critic_exact=True,optimizer_exact=True,environment_state_restored=False,
        note='Native low-stair terrain/reset/RNG and gait curriculum restart from the approved initial configuration.')


def source_check():
    verify_training()
    old=json.loads((TRAINING/'report.json').read_text())
    for row in old['sources']:
        if sha(row['path'])!=row['sha256']:
            raise ValueError(f'training source changed: {row["path"]}')
    reward=json.loads((TRAINING/'reward_override_audit.json').read_text())
    if sha(Path(__file__).with_name('reference_reward_variants.py'))!=reward['source_sha256']:
        raise ValueError('original reward variants changed')
    for name,digest in reward['low_stairs_sources'].items():
        if sha(Path(__file__).with_name(name))!=digest:
            raise ValueError(f'original low-stair source changed: {name}')
    return dict(status='passed',common_training_sources=len(old['sources']),reward_sources_unchanged=True)


def cpu_restore_check(output):
    import torch
    import yaml
    from types import SimpleNamespace
    from tensordict import TensorDict
    from rsl_rl.runners import OnPolicyRunner
    torch.set_num_threads(1)
    source=source_check()
    cfg=yaml.safe_load((TRAINING/'params/agent.yaml').read_text())
    cfg['device']='cpu'
    observations=TensorDict({'policy':torch.zeros(2,45),'critic':torch.zeros(2,393)},batch_size=[2])
    env=SimpleNamespace(cfg={},num_envs=2,num_actions=12,device='cpu',get_observations=lambda:observations)
    runner=OnPolicyRunner(env,cfg,log_dir=None,device='cpu')
    report=restore(runner)
    before=copy.deepcopy(runner.alg.save())
    with torch.inference_mode():
        action=runner.get_inference_policy()(observations)
        assert action.shape==(2,12) and torch.isfinite(action).all()
    assert equal_state(before,runner.alg.save())
    report.update(optimizer_updates=0,native_ppo_restore=True,synthetic_observations_only=True,source_check=source)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'report.json',report)
    return 0


def worker(args):
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    save(args.output/'source_check.json',source_check())
    reference,agent=load_reference()
    from reference_six_terrain_cfg import replace_terrain,column_names
    from reference_six_train import configuration_audit,audit_simulation
    from reference_low_stairs_cfg import apply_terrain,apply_rewards
    from reference_low_stairs_repair import apply_repair
    from reference_low_stairs_validate import geometry_audit
    cfg=replace_terrain(reference)
    save(args.output/'configuration_audit.json',configuration_audit(reference,cfg,agent))
    cfg,terrain_audit=apply_terrain(cfg);cfg,reward_audit=apply_rewards(cfg)
    cfg,repair_audit=apply_repair(cfg,VARIANT)
    save(args.output/'terrain_audit.json',dict(**terrain_audit,
        trial_initialization='explicit pinned low-stair continuation; original scratch entry unchanged'))
    save(args.output/'reward_audit.json',reward_audit);save(args.output/'repair_audit.json',repair_audit)
    save(args.output/'asset_resolution.json',resolve_lite3_asset(cfg))
    cfg.seed=agent.seed=42;cfg.scene.num_envs=120 if args.mode=='smoke' else 4096
    cfg.sim.device=agent.device='cuda:0';agent.max_iterations=args.updates;agent.logger='tensorboard'
    agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.deterministic=torch.backends.cudnn.benchmark=False
    save(args.output/'runtime_overrides.json',dict(seed=42,num_envs=cfg.scene.num_envs,device='cuda:0',
        max_iterations=args.updates,mode=args.mode,checkpoint=str(CHECKPOINT),resume=True,
        terrain_profile='low_stairs',reward_variant='low_stairs_riser_contact',repair_variant=VARIANT,
        automatic_extension=False,physical_gpu=1))
    files=[Path(__file__),Path(__file__).with_name('reference_low_stairs_repair.py')]
    save(args.output/'implementation_sources.json',[dict(path=str(p),sha256=sha(p)) for p in files])
    dump_yaml(str(args.output/'params/env.yaml'),cfg);dump_yaml(str(args.output/'params/agent.yaml'),agent)
    base=ManagerBasedRLEnv(cfg=cfg);env=RslRlVecEnvWrapper(base,clip_actions=agent.clip_actions)
    audit_simulation(base,cfg,args.output,make_plot=False)
    save(args.output/'geometry.json',geometry_audit(base,cfg))
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(args.output/'training'),device='cuda:0')
    runner.add_git_repo_to_log(__file__)
    runner.add_git_repo_to_log(str(REFERENCE/'scripts/reinforcement_learning/rsl_rl/train.py'))
    save(args.output/'restore.json',restore(runner))
    initial=copy.deepcopy(runner.alg.save())
    assert int(base.scene.terrain.terrain_levels.max())<=1
    names=column_names(cfg)
    terrain=base.scene.terrain
    coverage=torch.zeros(200,dtype=torch.long,device=base.device)
    collisions=torch.zeros_like(coverage)
    translating=torch.zeros_like(coverage)
    report=dict(status='running',mode=args.mode,seed=42,from_scratch=False,checkpoint_loaded=str(CHECKPOINT),
        initial_sha256=EXPECTED_SHA,initial_iteration=10000,initial_learning_rate=runner.alg.learning_rate,
        reward_variant='low_stairs_riser_contact',repair_variant=VARIANT,terrain_profile='low_stairs',
        completed_updates=0,environment_steps=0,resets=0,terminated=0,truncated=0,updates=[],checkpoints=[],
        automatic_extension=False,terrain_column_names=names,
        initial_level_histogram=torch.bincount(terrain.terrain_levels,minlength=10).cpu().tolist())
    save(args.output/'report.json',report)
    step_budget=100 if args.mode=='smoke' else args.updates*agent.num_steps_per_env
    native_reward=base.reward_manager.compute
    riser_index=list(base.reward_manager.active_terms).index('stair_riser_contact')
    def reward_capture(dt):
        total=native_reward(dt)
        slot=terrain.terrain_types*10+terrain.terrain_levels
        coverage.add_(torch.bincount(slot,minlength=200))
        active=base.reward_manager._step_reward[:,riser_index]<0.
        collisions.add_(torch.bincount(slot[active],minlength=200))
        moving=base.command_manager.get_command('base_velocity')[:,:2].norm(dim=-1)>.1
        translating.add_(torch.bincount(slot[moving],minlength=200))
        return total
    base.reward_manager.compute=reward_capture
    native_step,native_log=env.step,runner.logger.log
    def checked_step(action):
        if report['environment_steps']>=step_budget:raise RuntimeError('trial interaction budget exceeded')
        if not torch.isfinite(action).all():raise FloatingPointError('nonfinite trial actions')
        obs,rewards,dones,extras=native_step(action)
        if not torch.isfinite(rewards).all() or not all(torch.isfinite(v).all() for v in obs.values()):
            raise FloatingPointError('nonfinite trial observations or rewards')
        for name in ('height_scanner','height_scanner_base'):
            if not torch.isfinite(base.scene[name].data.ray_hits_w).all():raise FloatingPointError('nonfinite terrain rays')
        report['environment_steps']+=1;report['resets']+=int(dones.sum())
        report['terminated']+=int(base.reset_terminated.sum());report['truncated']+=int(base.reset_time_outs.sum())
        return obs,rewards,dones,extras
    def model_check(updates):
        state=runner.alg.save()
        assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()}=={200000+updates*20}
        for key in ('actor_state_dict','critic_state_dict'):
            if not all(torch.isfinite(v).all() for v in state[key].values()):raise FloatingPointError('nonfinite trial parameters')
        return state
    def checkpoint(updates):
        model_check(updates)
        path=args.output/f'model_updates_{10000+updates:05d}.pt'
        runner.save(str(path),infos=dict(from_scratch=False,completed_updates=10000+updates,phase_updates=updates,
            reward_variant='low_stairs_riser_contact',repair_variant=VARIANT,terrain_profile='low_stairs',
            validation_only=False,parent_sha256=EXPECTED_SHA,trial_budget=args.updates))
        report['checkpoints'].append(dict(phase_updates=updates,path=str(path),sha256=sha(path)))
    def record_coverage():
        report['coverage_by_column_level']=coverage.reshape(20,10).cpu().tolist()
        report['collision_frames_by_column_level']=collisions.reshape(20,10).cpu().tolist()
        report['translation_frames_by_column_level']=translating.reshape(20,10).cpu().tolist()
    def checked_log(**kwargs):
        losses={k:float(v) for k,v in kwargs['loss_dict'].items()}
        if not all(math.isfinite(v) for v in losses.values()):raise FloatingPointError('nonfinite PPO loss')
        report['completed_updates']+=1;n=report['completed_updates']
        if n>args.updates:raise RuntimeError('trial update budget exceeded')
        report['updates'].append(dict(update=n,losses=losses,learning_rate=runner.alg.learning_rate,
            collect_s=kwargs['collect_time'],learn_s=kwargs['learn_time'],
            mean_terrain_level=float(terrain.terrain_levels.float().mean())))
        native_log(**kwargs)
        if n==20:
            model_check(n)
            save(args.output/'first20_check.json',dict(status='passed',updates_included_in_budget=20,
                observations_and_losses_finite=True,adam_steps=200400,
                average_seconds_per_update=sum(x['collect_s']+x['learn_s'] for x in report['updates'])/20))
        if n in (100,250,args.updates):checkpoint(n)
        if n%20==0 or n==args.updates:record_coverage()
        save(args.output/'report.json',report)
        print(f"REFERENCE SIX train: {n}/{args.updates}",flush=True)
    env.step,runner.logger.log=checked_step,checked_log
    if args.mode=='smoke':
        policy=runner.get_inference_policy()
        obs=env.get_observations()
        with torch.inference_mode():
            for _ in range(step_budget):obs,*_=env.step(policy(obs))
        assert equal_state(initial,runner.alg.save())
    else:
        runner.learn(num_learning_iterations=args.updates,init_at_random_ep_len=True)
        current=model_check(args.updates)
        assert report['completed_updates']==args.updates and runner.current_learning_iteration==9999+args.updates
        assert all(any(not torch.equal(initial[key][k],v) for k,v in current[key].items())
                   for key in ('actor_state_dict','critic_state_dict'))
        last=report['checkpoints'][-1]
        report.update(output_checkpoint=last['path'],output_sha256=last['sha256'])
    assert report['environment_steps']==step_budget and sha(CHECKPOINT)==EXPECTED_SHA
    record_coverage()
    report.update(status='passed',interactions=step_budget*base.num_envs,
        cumulative_updates=10000+report['completed_updates'],final_learning_rate=runner.alg.learning_rate,
        actor_unchanged=args.mode=='smoke',sources=source_manifest(),
        final_terrain_level_histogram=torch.bincount(terrain.terrain_levels,minlength=10).cpu().tolist())
    save(args.output/'report.json',report)
    print('LOW STAIRS A TRIAL COMPLETE; no automatic extension',flush=True)
    sys.stdout.flush();os._exit(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('restore-check','smoke','train'),required=True)
    p.add_argument('--updates',type=int)
    p.add_argument('--gpu',type=int,choices=(1,),default=1)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',action='store_true')
    args=p.parse_args()
    if args.mode=='train' and (args.updates is None or not 1<=args.updates<=500):
        p.error('A trial requires an explicit budget of 1..500 updates')
    if args.mode!='train' and args.updates not in (None,0):p.error('checks must perform zero updates')
    args.updates=args.updates or 0;args.output=args.output.resolve()
    if args.mode=='restore-check':return cpu_restore_check(args.output)
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],gpu=1,
                   output=args.output,timeout=21600 if args.mode=='train' else 1800)
    try:worker(args)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__':sys.exit(main())
