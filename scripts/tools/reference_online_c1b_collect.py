"""Bounded paired C1-b collection with explicit reset and full control accounting."""
from __future__ import annotations

from dataclasses import fields
import hashlib

import numpy as np
import torch

from reference_online_c1b import OUTPUT, rows, protocol
from reference_online_targets import MODEL_PATHS, HASHES
from reference_stairs_eval import save, sha


def collect(args):
    from reference_online_c1b_env import Environment
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    specs = rows(args.kind, args.smoke_index)
    env = Environment(args, specs)
    from rl_training.ea_lpl.c1b import Session, FeatureHistory, exploration_table, digest, rewards
    from rl_training.ea_lpl.basis import basis
    from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
    from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
    from rl_training.online_adaptation.safety import SafetyLimits
    from reference_online_interface import clip_action
    n, device = len(specs), env.device
    noise = exploration_table()
    noise_sha = hashlib.sha256(noise.tobytes()).hexdigest()
    np.save(args.output / 'noise_table.npy', noise)
    heads = [Session(args.kind == 'train' and s['arm'] == 'Updated') for s in specs]
    if args.kind in ('target','retention'):
        checkpoint = torch.load(OUTPUT / 'train/final.pt', map_location='cpu', weights_only=True)
        for head, spec in zip(heads, specs):
            source = checkpoint['heads'][spec['session']*2+int(spec['arm']=='Updated')]
            head.model.load_state_dict(source['state_dict'])
            head.initial_digest = digest(head.model)
            head.begin()
    initial_digests = [digest(h.model) for h in heads]
    base_digest = digest(env.actor)
    limits = SafetyLimits(residual_limits=torch.tensor(protocol()['residual_limits'], device=device))
    safety = CurrentModelSafetyFilter(limits)
    bridge = ResidualSafetyBridge(env.adapter, safety, action_clip=clip_action)
    v = basis('sag', device=device)
    history = FeatureHistory(n)
    previous = torch.zeros(n,12,device=device)
    code = np.zeros((n,2),np.float32)
    timeout = torch.zeros(n,1,dtype=torch.bool,device=device)
    # PREP=0, POLICY=1, WAIT=2. Every simulator step belongs to one phase.
    phase, preparation = np.zeros(n,int), np.zeros(n,int)
    episodes, age = np.zeros(n,int), np.zeros(n,int)
    counts = {k:np.zeros(n,int) for k in ('preparation','policy','waiting')}
    dwell, progress, x0 = np.zeros(n), np.zeros(n), np.zeros(n)
    positions = [[] for _ in specs]
    records, resets, pair_checks, update_logs = [], [], [], []
    buffers = {}
    trace = None
    completed_steps = 0
    initial_physics_counter = env.base._sim_step_counter
    steps = 600 if args.kind == 'smoke' else 20000 if args.kind == 'train' else 1050
    resets.append(env.reset(list(range(n)), episodes))
    initial_snapshots = []

    def checkpoint(name, control_steps):
        torch.save(dict(heads=[h.snapshot() for h in heads], control_steps=control_steps,
            base_sha256=HASHES['B'], noise_sha256=noise_sha, kind=args.kind), args.output / name)

    def persist(name, control_steps):
        np.savez_compressed(args.output / 'trace.npz', **{k:np.asarray(value) for k,value in buffers.items()})
        save(args.output / name, dict(kind=args.kind, control_steps=control_steps, records=records,
            resets=resets, pair_checks=pair_checks, updates=update_logs,
            counts={k:value.tolist() for k,value in counts.items()}, phase=phase.tolist(),
            episode_index=episodes.tolist(), policy_age=age.tolist(), noise_sha256=noise_sha))

    checkpoint('initial.pt', 0)
    tick = -1
    try:
        for tick in range(steps):
            trace = None
            policy = phase == 1
            prep = phase == 0
            env.policy_active[:] = policy; env.policy_age[:] = age
            env.refresh()
            env.adapter.check_observation(env.obs['policy'], previous)
            before = env.frame()
            history.append(before['gravity'][:,:2],before['yaw_rate'],before['mean_joint_speed'])
            features = history.features(code)
            decisions = policy & (age % 10 == 0)
            for i in np.flatnonzero(decisions):
                exploration = noise[specs[i]['session'],episodes[i],age[i]//10] if args.kind in ('smoke','train') else None
                code[i] = heads[i].choose(features[i], exploration)
            with torch.no_grad():
                raw = env.actor(env.obs['policy'])
                raw[torch.as_tensor(~policy,device=device)] = 0
                request = torch.einsum('jk,nk->nj', v, torch.tanh(torch.tensor(code,device=device)))
                request[torch.as_tensor(~policy,device=device)] = 0
                trace = bridge.step(raw,request,timeout)
                state = trace.safety_state
                target = state.default_joint_pos+state.action_scale*trace.base_action
                projected_base = ((target < state.joint_lower+.02) | (target > state.joint_upper-.02)).any(-1)
                zero_history = (trace.filter_state_before['filter_previous_residual_action']==0).all(-1)
                zero_expected = (request==0).all(-1) & zero_history & ~projected_base
                if ((trace.executed_action != trace.base_action).any(-1) & zero_expected).any():
                    raise ValueError('zero residual with zero history changes feasible base action')
                if torch.count_nonzero(trace.executed_action[torch.as_tensor(~policy,device=device)]):
                    raise ValueError('preparation or waiting action is not zero')
                step_data = {f'state.{f.name}':getattr(state,f.name).detach().cpu().numpy().copy() for f in fields(state)}
                step_data.update({f'memory.{k}':value.cpu().numpy().copy() for k,value in trace.filter_state_before.items()})
                components = safety.execution_components()
                if args.kind in ('target','retention'):
                    frozen_ids = torch.tensor([i for i,s in enumerate(specs) if s['arm']=='Frozen'],device=device)
                    if torch.count_nonzero(components['learned_effect'][frozen_ids]):
                        raise ValueError('Frozen zero adapter is not equivalent to projected base')
                step_data.update({k:value.cpu().numpy().copy() for k,value in components.items()})
                step_data.update(base_action=trace.base_action.cpu().numpy().copy(), action=trace.executed_action.cpu().numpy().copy(),
                    residual_request=request.cpu().numpy().copy(), observation=env.obs['policy'].cpu().numpy().copy(),
                    phase=phase.copy(), episode=episodes.copy(), age=age.copy(), code=code.copy(), features=features.copy(),
                    decision=decisions.copy(), projected=trace.decision.projected[:,0].cpu().numpy().copy(),
                    paused=trace.decision.paused[:,0].cpu().numpy().copy(), base_projection=projected_base.cpu().numpy().copy())
                env.obs, _, term, trunc, _ = env.base.step(trace.executed_action)
                completed_steps += 1
                assert torch.equal(env.base.action_manager.action,trace.executed_action)
                previous = trace.executed_action.clone()
            after = env.frame()
            terminated, truncated = term.cpu().numpy().copy(), trunc.cpu().numpy().copy()
            step_data.update(terminated=terminated,truncated=truncated,root=after['root'],gravity=after['gravity'],feet=after['feet'])
            reset_filter = np.zeros(n,bool)
            score = np.zeros(n)
            for key, mask in (('preparation',prep),('policy',policy),('waiting',phase==2)):
                counts[key] += mask
            if ((terminated | truncated) & prep).any():
                raise RuntimeError('standing preparation terminated')
            preparation[prep] += 1
            finished = []
            for i in np.flatnonzero(policy):
                age[i] += 1
                spec = specs[i]; root = after['root'][i]
                geometry = env.geometry[f'{spec["condition"]}:{spec["row"]}']
                positions[i].append(float(root[0]))
                progress[i] = max(progress[i],float(root[0]-x0[i]))
                is_stair = bool(geometry['fronts'])
                if is_stair:
                    tilt = np.degrees(np.arccos(np.clip(-after['gravity'][i,2],-1.,1.)))
                    clear = (root[0] > geometry['goal_x'] and .18 < after['clearance'][i] < .55 and tilt < 44.
                        and after['feet'][i,:,0].min() > geometry['fronts'][-1]+.05
                        and after['feet'][i,:,2].min() > geometry['tops'][-1]-.01
                        and after['supports'][i] >= 2 and not after['torso_contact'][i])
                else:
                    clear = root[0]-x0[i] >= 3. and after['supports'][i] >= 2
                dwell[i] = dwell[i]+.02 if clear else 0.
                cap = (100 if args.smoke_index == 2 and spec['arm']=='Frozen' else 150) if args.kind == 'smoke' else 1000
                outcome = None
                if terminated[i]: outcome = 'posture_failure'
                elif abs(root[1]) > 1.: outcome = 'lateral_exit'
                elif dwell[i] >= (.26 if is_stair else .25)-1.e-8: outcome = 'success'
                elif truncated[i] or age[i] >= cap:
                    outcome = ('smoke_cutoff' if args.kind == 'smoke' else
                        'stuck' if is_stair and root[0]-positions[i][max(0,len(positions[i])-251)] < .15 else 'timeout')
                score[i] = rewards(np.array([root[0]-before['root'][i,0]]),
                    np.array([root[1]-before['root'][i,1]]),step_data['learned_effect'][i:i+1],
                    np.asarray(protocol()['residual_limits']),[outcome=='posture_failure'],[outcome=='lateral_exit'])[0]
                heads[i].rewards.append(float(score[i]))
                if outcome:
                    update = heads[i].finish(allow_update=args.kind=='train')
                    update_logs.append(dict(env_id=i,session=spec['session'],arm=spec['arm'],episode=int(episodes[i]),
                                            control_step=tick+1,**update))
                    length = geometry['goal_x']-x0[i] if is_stair else 3.
                    records.append(dict(**spec,episode=int(episodes[i]),outcome=outcome,policy_steps=int(age[i]),
                        final_root=root.tolist(),mean_progress=float(min(1.,progress[i]/length)),**update))
                    phase[i] = 2; code[i] = 0; finished.append(i)
            if finished:
                safety.reset_temporal_state(torch.tensor(finished,device=device))
                reset_filter[finished] = True
            ready = np.flatnonzero(prep & (preparation == 50))
            if len(ready):
                index = torch.tensor(ready,device=device)
                env.base.episode_length_buf[index] = 0
                env.base.action_manager.reset(index)
                safety.reset_temporal_state(index); reset_filter[ready] = True
                previous[index] = 0
                history.reset(ready)
                phase[ready] = 1; age[ready] = 0
                x0[ready] = after['root'][ready,0]
                env.policy_active[:] = phase == 1; env.policy_age[:] = age
                env.refresh()
                prepared = env.prepared_state(ready.tolist())
                initial_snapshots.append(dict(control_step=tick+1,ids=ready.tolist(),episode=episodes[ready].tolist(),
                    **{k:value.tolist() for k,value in prepared.items()}))
                for k in range(0,len(ready),2):
                    assert ready[k] % 2 == 0 and ready[k+1] == ready[k]+1
                    errors = {name:float(np.abs(value[k]-value[k+1]).max()) for name,value in prepared.items()}
                    pair_checks.append(dict(session=specs[ready[k]]['session'],env_ids=ready[k:k+2].tolist(),
                        episode=int(episodes[ready[k]]),errors=errors))
                    if max(errors.values()) > 1.e-6:
                        raise ValueError(f'paired preparation states differ: {errors}')
            reset_ids = []
            if args.kind in ('train','smoke') and steps-tick-1 > 50:
                for i in range(0,n,2):
                    if phase[i] == phase[i+1] == 2 and episodes[i] < 63:
                        reset_ids.extend([i,i+1])
                if reset_ids:
                    episodes[reset_ids] += 1
                    resets.append(env.reset(reset_ids,episodes))
                    phase[reset_ids] = 0; preparation[reset_ids] = age[reset_ids] = 0
                    dwell[reset_ids] = progress[reset_ids] = 0; code[reset_ids] = 0
                    history.reset(reset_ids)
                    for i in reset_ids:
                        heads[i].begin(); positions[i] = []
                    index = torch.tensor(reset_ids,device=device)
                    safety.reset_temporal_state(index); previous[index] = 0
                    reset_filter[reset_ids] = True
            step_data.update(score=score,filter_reset_after=reset_filter)
            for key,value in step_data.items():
                if not np.isfinite(value).all(): raise ValueError(f'nonfinite trace {key}')
                buffers.setdefault(key,[]).append(value)
            if tick+1 in (10000,20000): checkpoint(f'at_{tick+1}.pt',tick+1)
            if (tick+1) % 1000 == 0:
                print(f'C1B {args.kind}: {tick+1}/{steps}, episodes={len(records)}, updates={sum(h.updates for h in heads)}',flush=True)
            if args.kind in ('target','retention') and (phase == 2).all(): break
            if (phase == 2).all() and (episodes >= 63).all(): break
    except Exception:
        persist('partial_report.json',completed_steps)
        checkpoint('failure.pt',completed_steps)
        save(args.output / 'failure_clock.json',dict(loop_index=tick,completed_control_steps=completed_steps,
            native_physics_counter_delta=env.base._sim_step_counter-initial_physics_counter))
        save(args.output / 'prepared_states.json',initial_snapshots)
        raise
    assert digest(env.actor) == base_digest and sha(MODEL_PATHS['B']) == HASHES['B']
    assert all(p.grad is None and not p.requires_grad for p in env.actor.parameters())
    assert all(digest(h.model) == initial_digests[i] for i,h in enumerate(heads) if specs[i]['arm']=='Frozen' or args.kind!='train')
    assert sum(int(value.sum()) for value in counts.values()) == (tick+1)*n
    assert (tick+1)*n <= protocol()['budgets'][args.kind]
    if args.kind in ('target','retention'): assert len(records) == n
    if args.kind == 'smoke':
        assert all(episodes >= 2) and sum(h.updates for h in heads) == 0
        if args.smoke_index == 2: assert counts['waiting'][::2].min() >= 50
    if args.kind == 'train':
        assert all(h.updates > 0 for i,h in enumerate(heads) if specs[i]['arm']=='Updated')
    checkpoint('final.pt',tick+1)
    persist('episodes.json',tick+1)
    save(args.output / 'prepared_states.json',initial_snapshots)
    save(args.output / 'report.json',dict(status='passed',kind=args.kind,environments=n,control_steps=tick+1,
        interactions=(tick+1)*n,complete_episodes=len(records),adapter_updates=sum(h.updates for h in heads),
        session_updates=[h.updates for h in heads],base_optimizer_updates=0,base_unchanged=True,frozen_unchanged=True,
        base_actor_digest=base_digest,base_checkpoint_sha256=HASHES['B'],initial_head_digests=initial_digests,
        final_head_digests=[digest(h.model) for h in heads],noise_sha256=noise_sha,
        paired_preparations=len(pair_checks),paired_max_error=max(max(x['errors'].values()) for x in pair_checks),
        frozen_zero_equivalence_checked=args.kind in ('target','retention'),
        deferred_native_reset_calls=env.deferred_events,explicit_resets=sum(len(r['ids']) for r in resets),
        counts={k:value.tolist() for k,value in counts.items()},
        incomplete_policy_rows=np.flatnonzero(phase==1).tolist(),incomplete_episodes_used_for_update=False,
        final_sha256=sha(args.output / 'final.pt'),trace_sha256=sha(args.output / 'trace.npz')))
    print(f'COMPLETE {args.kind}: episodes={len(records)}, adapter_updates={sum(h.updates for h in heads)}',flush=True)
