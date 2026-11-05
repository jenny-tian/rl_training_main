"""Approved frozen-actor first-episode comparison; at most 480000 diagnostic transitions."""
import argparse
import faulthandler
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
sys.path[:0] = [str(ROOT/'source/rl_training'), str(ROOT/'scripts/tools')]
from source_adaptation_runtime import OLD, digest, write_json, plain, make_environment, environment_identity, STATE_NAMES, ACTION_NAMES
from source_delivery_light import record_phase, close_comparison_environment, lifecycle_deadline
from rl_training.source_scan.training_budget_ledger import canonical_budget_path

PREVIOUS = ROOT/'docs/lite3-source-fixed-low-lr-trial-20261105'
PROPOSAL = PREVIOUS/'deliverables/next_stage_plan.json'
NEW = PREVIOUS/'deliverables/actor_update_2.pt'
BASELINE = ROOT/'docs/lite3-source-reward-live-verification-20261105'
HASHES = ['87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c',
          'e7fc084e6beb8cc8a4b5d9d4aefd75ae97a7160a695079cbbf36216719bf822f']
BUDGET_SHA256 = '9e54797c258c4d7c658bb07b9b3fe39d400019805944c44fe9d055946e1357b8'
FULL_STATE_NAMES = STATE_NAMES + ['lateral_m','gravity_roll_proxy_rad','gravity_pitch_proxy_rad','yaw_rad','yaw_rate_rps']


def collect(wrapped, actor, label, folder, app, report):
    import numpy as np
    import torch
    from source_delivery_light import check_scans
    from rl_training.source_scan.delivery_metrics import append_json
    from rl_training.source_scan.adaptation_profile import validate_source_adaptation_config
    from rl_training.source_scan.first_episode_comparison import first_episodes, episode_totals
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as reward_mdp
    base = wrapped.unwrapped
    if base.num_envs != 120 or base.step_dt != 0.02:
        raise ValueError('environment scale or control period changed')
    manager = base.reward_manager
    names = list(manager.active_terms)
    resolved = {name:manager.get_term_cfg(name) for name in names}
    profile = validate_source_adaptation_config(SimpleNamespace(rewards=resolved,
        source_reward_profile=base.cfg.source_reward_profile,reward_gait_level=base.cfg.reward_gait_level))
    identity = environment_identity(base)
    before = {k:v.clone() for k,v in actor.state_dict().items()}
    observations = wrapped.get_observations()
    original_compute = manager.compute
    captured = {}
    chunks = {key:[] for key in ('rewards','state','columns','gate','hold','episode_steps','actions','policy_action','done')}
    events = []
    first_ended = np.zeros(120,dtype=bool)
    first_completed = 0
    reconstruction = 0.0

    def compute(dt):
        reward = original_compute(dt)
        data = base.scene['robot'].data
        velocity = data.root_lin_vel_b
        gravity = data.projected_gravity_b
        err2 = (velocity[:,:2]-base.command_manager.get_command('base_velocity')[:,:2]).square().sum(-1)
        ground = base.scene.sensors['height_scanner_base'].data.ray_hits_w[:,:,2].mean(-1)
        relative = data.root_link_pos_w-base.scene.terrain.env_origins
        w,x,y,z = data.root_link_quat_w.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y),1-2*(y.square()+z.square()))
        state = torch.stack((velocity[:,0],velocity[:,1],velocity[:,2],err2,data.root_link_pos_w[:,2]-ground,
            relative[:,0],(data.joint_pos-data.default_joint_pos).norm(dim=-1),
            data.joint_vel.square().mean(-1).sqrt(),gravity[:,:2].norm(dim=-1),relative[:,1],
            torch.asin(gravity[:,1].clamp(-1,1)),torch.asin(gravity[:,0].clamp(-1,1)),yaw,data.root_ang_vel_b[:,2]),dim=-1)
        captured.update(rewards=manager._step_reward.clone(),state=state,
            columns=base.scene.terrain.terrain_types.clone(),gate=reward_mdp.get_gait_level_tensor(base).clone(),
            hold=wrapped.runtime.lifecycle.hold.clone(),episode_steps=base.episode_length_buf.clone())
        return reward

    def progress(step):
        write_json(folder/'progress.json',dict(actor=label,actor_control=step,
            attempted_diagnostic_transitions=report['attempted_diagnostic_transitions'],
            diagnostic_transitions=report['diagnostic_transitions'],first_episodes_ended=int(first_ended.sum()),
            first_episodes_completed=first_completed))

    manager.compute = compute
    try:
        with torch.inference_mode():
            for step in range(1,2001):
                if not app.is_running():
                    raise RuntimeError('simulator stopped during full-episode comparison')
                action = actor(observations)
                saved_action = action.clone()
                report['attempted_diagnostic_transitions'] += 120
                if report['attempted_diagnostic_transitions'] > 480000:
                    raise RuntimeError('diagnostic transition cap exceeded')
                progress(step)
                observations,reward,done,extras = wrapped.step(action)
                report['diagnostic_transitions'] += 120
                trace = wrapped.last_trace
                if not torch.equal(saved_action,action):
                    raise RuntimeError('input action changed in place')
                if not bool((captured['gate']==1).all()):
                    raise RuntimeError('reward gate changed')
                reconstruction = max(reconstruction,float((captured['rewards'].sum(-1)*base.step_dt-reward).abs().max()))
                action_metrics = torch.stack((action.square().mean(-1).sqrt(),
                    (trace.raw_policy_action!=trace.base_action).float().mean(-1),
                    trace.decision.projected[:,0].float(),trace.decision.paused[:,0].float(),
                    (trace.executed_action-action).abs().amax(-1)),dim=-1)
                for key,value in captured.items():
                    chunks[key].append(value.cpu().numpy().copy())
                chunks['actions'].append(action_metrics.cpu().numpy().copy())
                chunks['policy_action'].append(action.cpu().numpy().copy())
                chunks['done'].append(done.cpu().numpy().copy())
                terminal = extras['source_terminal']
                if terminal is not None:
                    ids = terminal['ids'].cpu().tolist()
                    ending = {k:terminal[k].cpu().tolist() for k in ('steps','hold','completed')}
                    terms = {k:v.cpu().tolist() for k,v in terminal['terms'].items()}
                    for j,i in enumerate(ids):
                        event = dict(env_id=i,control_step=step,episode_step=ending['steps'][j],
                            hold=ending['hold'][j],completed=ending['completed'][j],
                            terms=[k for k,v in terms.items() if v[j]],is_first=not bool(first_ended[i]))
                        events.append(event)
                        append_json(folder/(label+'_endings.jsonl'),event)
                        if not first_ended[i]:
                            first_ended[i] = True
                            first_completed += int(event['completed'])
                check_scans(base)
                progress(step)
                if step%200==0 or first_ended.all():
                    print(f'[full-episode] {label} controls={step}/2000 first_ended={int(first_ended.sum())}/120 '
                          f'completed={first_completed}',flush=True)
                if first_ended.all():
                    break
    finally:
        manager.compute = original_compute
        arrays = {key:np.asarray(value) for key,value in chunks.items()}
        np.savez_compressed(folder/(label+'.npz'),**arrays)
        write_json(folder/(label+'_events.json'),events)
    if any(not np.isfinite(value).all() for value in arrays.values()) or reconstruction>1e-5:
        raise ValueError('nonfinite data or reward reconstruction mismatch')
    if any(not torch.equal(value,before[key]) for key,value in actor.state_dict().items()):
        raise RuntimeError('frozen actor changed')
    episodes = first_episodes(arrays,events,identity['assignments'],control_dt=base.step_dt)
    # The original 400 controls and candidate's 400 controls already have archived references.
    reference = np.load(BASELINE/'raw/runtime/old_mean.npz' if label=='original'
                        else PREVIOUS/'raw/runtime/actor_update_2.npz')
    prefix = dict(state=np.array_equal(arrays['state'][:400,:,:9],reference['state']),
        columns=np.array_equal(arrays['columns'][:400],reference['columns']),
        done=np.array_equal(arrays['done'][:400],reference['done']),
        rewards=np.array_equal(arrays['rewards'][:400],reference['rewards']))
    if not all(prefix.values()):
        raise ValueError('first 400 controls differ from the archived short comparison')
    return dict(label=label,initial=identity,profile=profile,reward_names=names,
        resolved_reward_config=plain(resolved),control_dt=base.step_dt,controls=len(arrays['done']),
        diagnostic_transitions=len(arrays['done'])*120,first_episodes=episodes,totals=episode_totals(episodes),
        prefix_matches_short_trial=prefix,reward_reconstruction_max_abs=reconstruction,
        actor_parameters_unchanged=True,input_actions_unchanged=True)


def child(folder):
    report = dict(status='starting',attempted_diagnostic_transitions=0,diagnostic_transitions=0,
        training_transitions=0,ppo_updates=0,cases=[],state_names=FULL_STATE_NAMES,action_names=ACTION_NAMES)
    manifest = json.loads((folder/'manifest.json').read_text())
    app = wrapped = None
    stack_file = (folder/'python_stacks.log').open('w')
    faulthandler.enable(file=stack_file)
    def phase(name,label=None,limit=120):
        faulthandler.cancel_dump_traceback_later()
        record_phase(folder,name,label,limit)
        if limit:
            faulthandler.dump_traceback_later(max(1,limit-5),file=stack_file)
    try:
        if os.getppid()!=manifest['supervisor_pid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
            raise ValueError('comparison requires its GPU0 supervisor')
        for name,expected in manifest['input_sha256'].items():
            if digest(name)!=expected:
                raise ValueError('comparison input changed: '+name)
        if digest(canonical_budget_path(ROOT))!=BUDGET_SHA256:
            raise ValueError('training ledger changed')
        phase('application_start')
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,device='cuda:0').app
        import numpy as np
        import torch
        from rl_training.source_scan.frozen_actor import FrozenSourceActor
        from rl_training.source_scan.first_episode_comparison import compare_episodes
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        expected_initial = json.loads((BASELINE/'raw/runtime/result.json').read_text())['cases'][0]['initial']
        for label,path in [('original',OLD),('candidate',NEW)]:
            phase('environment_start',label)
            wrapped = make_environment()
            if environment_identity(wrapped.unwrapped)!=expected_initial:
                raise ValueError('full-episode initial state or geometry differs from the baseline')
            state = torch.load(path,map_location='cpu',weights_only=True)
            actor = FrozenSourceActor(state['actor_state_dict'] if label=='original' else state).to('cuda:0')
            phase('actor_controls',label,480)
            report['cases'].append(collect(wrapped,actor,label,folder,app,report))
            write_json(folder/'result.json',report)
            phase('environment_close',label,15)
            close_comparison_environment(wrapped)
            wrapped = None
            if label=='original':
                phase('stage_replace',label)
                from isaaclab.sim import create_new_stage
                create_new_stage()
        phase('comparison_finalize',limit=30)
        old,new = report['cases']
        with np.load(folder/'original.npz') as a, np.load(folder/'candidate.npz') as b:
            report['comparison'] = compare_episodes(old['first_episodes'],new['first_episodes'],a,b,
                manifest['plan']['comparison_gates'])
        if digest(canonical_budget_path(ROOT))!=BUDGET_SHA256 or [digest(p) for p in (OLD,NEW)]!=HASHES:
            raise ValueError('read-only comparison changed a checkpoint or training ledger')
        report['status'] = 'passed' if report['comparison']['passed'] else 'behavior_regressed'
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
    return 0 if report['status'] in ('passed','behavior_regressed') else 1


def supervise(folder):
    budget = canonical_budget_path(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or digest(budget)!=BUDGET_SHA256:
        raise ValueError('GPU assignment or training ledger differs from the agreed baseline')
    if [digest(p) for p in (OLD,NEW)]!=HASHES:
        raise ValueError('comparison checkpoint changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('commit comparison changes before launch')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch = subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=ROOT,text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','--exit-code','origin','refs/heads/'+branch],
        cwd=ROOT,text=True,timeout=40).split()
    if remote!=[revision,'refs/heads/'+branch]:
        raise ValueError('push comparison revision before launch')
    plan = json.loads(PROPOSAL.read_text())
    inputs = [OLD,NEW,PROPOSAL,Path(__file__),ROOT/'scripts/tools/source_adaptation_runtime.py',
        ROOT/'scripts/tools/source_delivery_light.py',BASELINE/'raw/runtime/result.json',
        BASELINE/'raw/runtime/old_mean.npz',PREVIOUS/'raw/runtime/actor_update_2.npz']
    inputs += list((ROOT/'source/rl_training/rl_training/source_scan').glob('*.py'))
    inputs += list((ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3').glob('*.py'))
    inputs += list((ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp').glob('*.py'))
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'budget_before.json').write_bytes(budget.read_bytes())
    write_json(folder/'manifest.json',dict(schema='source_fixed_lr_full_episode_comparison_v1',
        authorization_basis='user explicitly agreed to the detailed full-first-episode comparison proposal',
        user_approved=True,
        plan=plan,source_revision=revision,supervisor_pid=os.getpid(),python=sys.executable,
        input_sha256={str(p):digest(p) for p in sorted(set(inputs))},budget_before=BUDGET_SHA256,
        diagnostic_transition_cap=480000,training_transitions=0,ppo_updates=0,wall_limit_s=600,
        startup_limit_s=120,close_limit_s=15,automatic_retry=False))
    started = time.monotonic()
    proc = None
    reason = None
    code = 1
    remaining = False
    try:
        with (folder/'runtime.log').open('w') as log:
            proc = subprocess.Popen([sys.executable,'-u',__file__,'--child','--output',str(folder)],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                now = time.monotonic()
                reason = '600s wall deadline exceeded' if now-started>600 else lifecycle_deadline(folder,now)
                if reason:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                    break
                time.sleep(1)
            code = proc.wait()
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid,0)
                remaining = True
                os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        progress_path = folder/'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        (folder/'budget_after.json').write_bytes(budget.read_bytes())
        receipt = dict(exit_code=code,deadline_failure=reason,elapsed_s=time.monotonic()-started,
            process_group_had_remaining_children=remaining,budget_before=BUDGET_SHA256,budget_after=digest(budget),
            attempted_diagnostic_transitions=progress.get('attempted_diagnostic_transitions',0),
            diagnostic_transitions=progress.get('diagnostic_transitions',0),training_transitions=0,ppo_updates=0)
        write_json(folder/'execution.json',receipt)
        print(json.dumps(receipt),flush=True)
    return code if code else int(reason is not None or remaining or receipt['budget_after']!=BUDGET_SHA256)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    raise SystemExit(child(args.output) if args.child else supervise(args.output))
