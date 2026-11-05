"""Bounded live verification of the corrected Source reward profile; no PPO training."""
import argparse
from dataclasses import asdict
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
OLD = Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt')
NEW = ROOT/'docs/lite3-source-training-delivery-revision-20261104/deliverables/actor_update_90.pt'
HASHES = ['87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c',
          '14d11c8b57392c3ac0ac1e08b26de1cc1254ee059d1f653172433df9a05bdf42']
STATE_NAMES = ['vx_mps','vy_mps','vz_mps','velocity_error_squared','height_above_scan_m',
               'progress_m','joint_deviation_l2','joint_velocity_rms','tilt_xy_norm']
ACTION_NAMES = ['mean_action_rms','clipped_joint_fraction','projected','paused','executed_delta_max']
STEPS, NUM_ENVS = 400, 120


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def plain(value):
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    if isinstance(value,slice): return dict(slice=[value.start,value.stop,value.step])
    if callable(value): return value.__module__+':'+value.__qualname__
    if hasattr(value,'__dict__'): return plain(vars(value))
    raise TypeError(f'unsupported configuration type: {type(value)}')


def make_environment():
    import gymnasium as gym
    import rl_training.tasks
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rl_training.source_scan.training_env import SourceTrainingVecEnv
    from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_adaptation_env_cfg import DeeproboticsLite3SourceAdaptationEnvCfg
    cfg = DeeproboticsLite3SourceAdaptationEnvCfg()
    cfg.sim.device = 'cuda:0'
    raw = gym.make('SourceTerrain-Deeprobotics-Lite3-v0',cfg=cfg)
    return SourceTrainingVecEnv(RslRlVecEnvWrapper(raw,clip_actions=None))


def environment_identity(base):
    data = base.scene['robot'].data
    return dict(assignments=[asdict(t) for t in base._source_parallel_terrain_allocator.assignments],
        terrain_origins=base.scene.terrain.terrain_origins.cpu().tolist(),
        initial_state_sha256=hashlib.sha256(data.root_link_state_w.cpu().numpy().tobytes()
            +data.joint_pos.cpu().numpy().tobytes()+data.joint_vel.cpu().numpy().tobytes()).hexdigest())


def collect_case(wrapped, actor, label, folder, app, report):
    """Record existing manager outputs before reset; never evaluate a reward twice."""
    import numpy as np
    import torch
    from source_delivery_light import check_scans
    from rl_training.source_scan.adaptation_profile import validate_source_adaptation_config
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as reward_mdp
    base = wrapped.unwrapped
    manager = base.reward_manager
    names = list(manager.active_terms)
    resolved = {name:manager.get_term_cfg(name) for name in names}
    validated = validate_source_adaptation_config(SimpleNamespace(rewards=resolved,
        source_reward_profile=base.cfg.source_reward_profile,reward_gait_level=base.cfg.reward_gait_level))
    identity = environment_identity(base)
    observations = wrapped.get_observations()
    before = {k:v.detach().clone() for k,v in actor.state_dict().items()}
    original_compute = manager.compute
    captured = {}
    chunks = {key:[] for key in ('rewards','state','columns','actions','done','gate')}
    snapshots = {}
    endings = {}
    reconstruction = 0.0

    def compute(dt):
        value = original_compute(dt)
        data = base.scene['robot'].data
        velocity = data.root_lin_vel_b
        err2 = (velocity[:,:2]-base.command_manager.get_command('base_velocity')[:,:2]).square().sum(-1)
        ground = base.scene.sensors['height_scanner_base'].data.ray_hits_w[:,:,2].mean(-1)
        state = torch.stack((velocity[:,0],velocity[:,1],velocity[:,2],err2,data.root_link_pos_w[:,2]-ground,
            data.root_link_pos_w[:,0]-base.scene.terrain.env_origins[:,0],
            (data.joint_pos-data.default_joint_pos).norm(dim=-1),data.joint_vel.square().mean(-1).sqrt(),
            data.projected_gravity_b[:,:2].norm(dim=-1)),dim=-1)
        captured.update(rewards=manager._step_reward.clone(),state=state,
            columns=base.scene.terrain.terrain_types.clone(),gate=reward_mdp.get_gait_level_tensor(base).clone())
        return value

    manager.compute = compute
    try:
        with torch.inference_mode():
            for step in range(1,STEPS+1):
                if not app.is_running(): raise RuntimeError('simulator stopped')
                action = actor(observations)
                saved_action = action.clone()
                if step in (1,100,200,400):
                    snapshots[step] = dict(observation=observations['policy'].cpu().clone(),mean=action.cpu().clone())
                observations,reward,done,extras = wrapped.step(action)
                trace = wrapped.last_trace
                if not torch.equal(action,saved_action): raise RuntimeError('input action changed in place')
                if not bool((captured['gate']==1).all()): raise RuntimeError('environment reward gate changed')
                reconstruction = max(reconstruction,float((captured['rewards'].sum(-1)*base.step_dt-reward).abs().max()))
                actions = torch.stack((action.square().mean(-1).sqrt(),
                    (trace.raw_policy_action!=trace.base_action).float().mean(-1),
                    trace.decision.projected[:,0].float(),trace.decision.paused[:,0].float(),
                    (trace.executed_action-action).abs().amax(-1)),dim=-1)
                for key in ('rewards','state','columns','gate'):
                    chunks[key].append(captured[key].cpu().numpy().copy())
                chunks['actions'].append(actions.cpu().numpy())
                chunks['done'].append(done.cpu().numpy())
                terminal = extras['source_terminal']
                if terminal is not None:
                    for name,mask in terminal['terms'].items(): endings[name] = endings.get(name,0)+int(mask.sum())
                check_scans(base)
                report['diagnostic_transitions'] += NUM_ENVS
                if step%100==0: print(f'[adaptation] {label} {step}/{STEPS}',flush=True)
    finally:
        manager.compute = original_compute
        # Preserve partial arrays even when validation or a later environment fails.
        arrays = {key:np.asarray(value) for key,value in chunks.items()}
        np.savez_compressed(folder/(label+'.npz'),**arrays)
        torch.save(snapshots,folder/(label+'_observations.pt'))
    if any(not np.isfinite(value).all() for value in arrays.values()): raise FloatingPointError('nonfinite diagnostic data')
    if reconstruction>1e-5: raise RuntimeError('reward components do not reconstruct the total')
    if any(not torch.equal(v,before[k]) for k,v in actor.state_dict().items()): raise RuntimeError('actor parameters changed')
    def summarize(start):
        rewards = arrays['rewards'][start:].astype(np.float64)
        state = arrays['state'][start:].astype(np.float64)
        actions = arrays['actions'][start:].astype(np.float64)
        return dict(vx_mps=float(state[:,:,0].mean()),velocity_rms_error_mps=float(np.sqrt(state[:,:,3].mean())),
            reward_per_control=float(rewards.sum(-1).mean()*base.step_dt),clipped_joint_fraction=float(actions[:,:,1].mean()),
            weighted_reward_per_second=dict(zip(names,rewards.mean((0,1)).tolist())))
    return dict(label=label,initial=identity,profile=validated,reward_names=names,
        resolved_reward_config=plain(resolved),control_dt=base.step_dt,all_400=summarize(0),after_100=summarize(100),
        done_count=int(arrays['done'].sum()),terminal_counts=endings,gate_min=float(arrays['gate'].min()),
        gate_max=float(arrays['gate'].max()),reward_reconstruction_max_abs=reconstruction,
        actor_parameters_unchanged=True,input_actions_unchanged=True)


def child(folder):
    from source_delivery_light import record_phase,close_comparison_environment
    from rl_training.source_scan.training_budget_ledger import canonical_budget_path
    budget = canonical_budget_path(ROOT)
    report = dict(status='starting',diagnostic_transitions=0,training_transitions=0,ppo_updates=0,cases=[],
        budget_before=digest(budget),state_names=STATE_NAMES,action_names=ACTION_NAMES)
    app = wrapped = None
    stack_file = (folder/'python_stacks.log').open('w')
    faulthandler.enable(file=stack_file)
    def phase(name,label=None,limit=120):
        faulthandler.cancel_dump_traceback_later()
        record_phase(folder,name,label,limit)
        if limit: faulthandler.dump_traceback_later(max(1,limit-5),file=stack_file)
    try:
        if [digest(p) for p in (OLD,NEW)]!=HASHES: raise ValueError('checkpoint hash mismatch')
        phase('application_start')
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,device='cuda:0').app
        import torch
        from rl_training.source_scan.frozen_actor import FrozenSourceActor
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        for label,path in [('old_mean',OLD),('new_mean',NEW)]:
            phase('environment_start',label)
            wrapped = make_environment()
            state = torch.load(path,map_location='cpu',weights_only=True)
            actor = FrozenSourceActor(state['actor_state_dict'] if label=='old_mean' else state).to('cuda:0')
            phase('actor_controls',label,120)
            case = collect_case(wrapped,actor,label,folder,app,report)
            report['cases'].append(case)
            write_json(folder/'result.json',report)
            if len(report['cases'])>1 and case['initial']!=report['cases'][0]['initial']:
                raise ValueError('paired initial states or geometry differ')
            phase('environment_close',label,15)
            close_comparison_environment(wrapped)
            wrapped = None
            if label=='old_mean':
                phase('stage_replace',label)
                from isaaclab.sim import create_new_stage
                create_new_stage()
        old,new = report['cases']
        report['behavior_checks'] = dict(original_walks=old['after_100']['vx_mps']>=0.2,
            original_has_no_endings=old['done_count']==0,original_mean_clip_below_one_percent=old['after_100']['clipped_joint_fraction']<=0.01,
            walking_reward_exceeds_stationary_candidate=old['after_100']['reward_per_control']>new['after_100']['reward_per_control'],
            candidate_remains_stationary=abs(new['after_100']['vx_mps'])<0.05)
        report['eligible_for_two_update_trial'] = all(report['behavior_checks'].values())
        if digest(budget)!=report['budget_before'] or [digest(p) for p in (OLD,NEW)]!=HASHES:
            raise RuntimeError('checkpoint or budget changed')
        report.update(status='passed',budget_after=digest(budget))
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(folder/'result.json',report)
        if wrapped is not None:
            phase('environment_close',limit=15)
            close_comparison_environment(wrapped)
        if app is not None:
            phase('application_close',limit=15)
            app.close()
        phase('finished',limit=None)
        faulthandler.disable()
        stack_file.close()
    return 0 if report['status']=='passed' else 1


def supervise(folder):
    from source_delivery_light import lifecycle_deadline
    from rl_training.source_scan.training_budget_ledger import canonical_budget_path
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise ValueError('verification is restricted to GPU0')
    folder.mkdir(parents=True,exist_ok=False)
    before = digest(canonical_budget_path(ROOT))
    write_json(folder/'manifest.json',dict(schema='source_adaptation_live_verification_v1',
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=digest(__file__),profile='lite3_source_lowspeed_v1',num_envs=120,steps_per_actor=400,
        actor_labels=['old_mean','new_mean'],checkpoint_sha256=HASHES,diagnostic_transition_cap=96000,
        wall_limit_s=300,startup_limit_s=120,close_limit_s=15,training_transitions=0,ppo_updates=0,
        behavior_limits=dict(original_min_vx_mps=0.2,original_max_clipped_joint_fraction=0.01,candidate_max_abs_vx_mps=0.05),
        authorization_basis='user agreed to the bounded next-stage plan',budget_before=before,python=sys.executable))
    started = time.monotonic()
    reason = None
    with (folder/'runtime.log').open('w') as log:
        proc = subprocess.Popen([sys.executable,'-u',__file__,'--child','--output',str(folder)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            now = time.monotonic()
            reason = '300s wall deadline exceeded' if now-started>300 else lifecycle_deadline(folder,now)
            if reason:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL)
                break
            time.sleep(1)
        code = proc.wait()
    try:
        os.killpg(proc.pid,0)
        remaining = True
        os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:
        remaining = False
    receipt = dict(exit_code=code,deadline_failure=reason,elapsed_s=time.monotonic()-started,
        process_group_had_remaining_children=remaining,budget_before=before,budget_after=digest(canonical_budget_path(ROOT)))
    write_json(folder/'execution.json',receipt)
    print(json.dumps(receipt),flush=True)
    return code if code else int(reason is not None or remaining or before!=receipt['budget_after'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    raise SystemExit(child(args.output) if args.child else supervise(args.output))
