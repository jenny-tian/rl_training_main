"""C1-a collector: one requested code per episode, complete four-episode batches."""
from __future__ import annotations

from dataclasses import fields
import hashlib

import numpy as np
import torch

from reference_online_c1a import OUTPUT, MARKS, rows, protocol
from reference_online_targets import MODEL_PATHS, HASHES
from reference_stairs_eval import save as native_save, sha


def json_values(value):
    if isinstance(value,np.ndarray): return json_values(value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {key:json_values(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [json_values(item) for item in value]
    return value


def save(path,value):
    native_save(path,json_values(value))


class TraceWriter:
    """Bound RAM; each chunk carries an unambiguous contiguous time range."""
    def __init__(self,directory):
        self.directory = directory
        self.buffer, self.chunks = {}, []
        self.steps = 0

    def append(self,values):
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError('nonfinite trace')
        if self.buffer and set(self.buffer) != set(values):
            raise ValueError('trace schema changed')
        for key,value in values.items(): self.buffer.setdefault(key,[]).append(value)
        self.steps += 1
        if self.steps % 1000 == 0: self.flush()

    def flush(self):
        if not self.buffer: return
        arrays = {k:np.asarray(v) for k,v in self.buffer.items()}
        count = len(arrays['phase'])
        path = self.directory/f'trace_{len(self.chunks):04d}.npz'
        np.savez_compressed(path,**arrays)
        self.chunks.append(dict(file=path.name,start=self.steps-count,stop=self.steps,sha256=sha(path)))
        self.buffer = {}
        save(self.directory/'trace_index.json',dict(control_steps=self.steps,chunks=self.chunks))


def collect(args):
    from reference_online_c1a_env import Environment
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    specs = rows(args.kind,part=args.part,smoke_index=args.smoke_index)
    env = Environment(args,specs)
    from rl_training.ea_lpl.c1a import Session, exploration_table, episode_return
    from rl_training.ea_lpl.c1b import digest
    from rl_training.ea_lpl.basis import basis
    from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
    from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
    from rl_training.online_adaptation.safety import SafetyLimits
    from reference_online_interface import clip_action
    n, device = len(specs), env.device
    noise = exploration_table()
    noise_sha = hashlib.sha256(noise.tobytes()).hexdigest()
    np.save(args.output/'noise_table.npy',noise)
    heads = [Session(args.kind=='train' and s['arm']=='Updated') for s in specs]
    if args.kind in ('target','retention'):
        path = OUTPUT/'train'/('initial.pt' if args.mark==0 else f'at_{args.mark}.pt')
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['control_steps'] != args.mark: raise ValueError('wrong evaluation snapshot')
        for head,spec in zip(heads,specs):
            head.load_evaluation_mean(checkpoint['heads'][2*spec['session']+int(spec['arm']=='Updated')])
    initial_digests = [h.digest() for h in heads]
    base_digest = digest(env.actor)
    safety = CurrentModelSafetyFilter(SafetyLimits(residual_limits=torch.tensor(protocol()['residual_limits'],device=device)))
    bridge = ResidualSafetyBridge(env.adapter,safety,action_clip=clip_action)
    v = basis('sag',device=device)
    previous = torch.zeros(n,12,device=device)
    timeout = torch.zeros(n,1,dtype=torch.bool,device=device)
    code = np.zeros((n,2),np.float64)
    phase, preparation, episodes, age = (np.zeros(n,int) for _ in range(4))
    counts = {k:np.zeros(n,int) for k in ('preparation','policy','waiting')}
    dwell, progress, x0, velocity_error_sum = (np.zeros(n) for _ in range(4))
    positions = [[] for _ in specs]
    records, resets, pairs, prepared_states = [], [], [], []
    writer = TraceWriter(args.output)
    completed_steps = 0
    physics_start = env.base._sim_step_counter
    resets.append(env.reset(list(range(n)),episodes))

    def checkpoint(name,step):
        torch.save(dict(heads=[h.snapshot() for h in heads],control_steps=step,
            base_sha256=HASHES['B'],noise_sha256=noise_sha,kind=args.kind),args.output/name)

    def persist(name):
        writer.flush()
        save(args.output/name,dict(kind=args.kind,control_steps=completed_steps,records=records,resets=resets,
            pair_checks=pairs,counts=counts,phase=phase,episode_index=episodes,policy_age=age,noise_sha256=noise_sha))
        save(args.output/'prepared_states.json',prepared_states)

    checkpoint('initial.pt',0)
    try:
        for tick in range(args.steps):
            policy, prep = phase==1, phase==0
            env.policy_active[:] = policy; env.policy_age[:] = age
            env.refresh()
            env.adapter.check_observation(env.obs['policy'],previous)
            before = env.frame()
            decisions = policy & (age==0)
            for i in np.flatnonzero(decisions):
                exploration = noise[specs[i]['session'],episodes[i]] if args.kind in ('train','smoke') else None
                code[i] = heads[i].begin(exploration)
            for i in np.flatnonzero(policy):
                if not np.array_equal(code[i],heads[i].requested_code()): raise ValueError('code changed inside episode')
            with torch.no_grad():
                raw = env.actor(env.obs['policy'])
                raw[torch.as_tensor(~policy,device=device)] = 0
                request = torch.einsum('jk,nk->nj',v,torch.tanh(torch.tensor(code,dtype=torch.float32,device=device)))
                request[torch.as_tensor(~policy,device=device)] = 0
                trace = bridge.step(raw,request,timeout)
                state = trace.safety_state
                if torch.count_nonzero(trace.executed_action[torch.as_tensor(~policy,device=device)]):
                    raise ValueError('nonzero preparation or waiting action')
                components = safety.execution_components()
                if args.kind in ('target','retention') and torch.count_nonzero(components['learned_effect'][::2]):
                    raise ValueError('Frozen not equivalent to projected base')
                values = {f'state.{f.name}':getattr(state,f.name).cpu().numpy().copy() for f in fields(state)}
                values.update({f'memory.{k}':value.cpu().numpy().copy() for k,value in trace.filter_state_before.items()})
                values.update({k:value.cpu().numpy().copy() for k,value in components.items()})
                values.update(base_action=trace.base_action.cpu().numpy().copy(),action=trace.executed_action.cpu().numpy().copy(),
                    residual_request=request.cpu().numpy().copy(),observation=env.obs['policy'].cpu().numpy().copy(),
                    phase=phase.copy(),episode=episodes.copy(),age=age.copy(),code=code.copy(),decision=decisions.copy(),
                    paused=trace.decision.paused[:,0].cpu().numpy().copy(),projected=trace.decision.projected[:,0].cpu().numpy().copy(),
                    before_root=before['root'],command=env.command.vel_command_b.cpu().numpy().copy())
                env.obs,_,term,trunc,_ = env.base.step(trace.executed_action)
                completed_steps += 1
                if not torch.equal(env.base.action_manager.action,trace.executed_action): raise ValueError('executed action mismatch')
                previous = trace.executed_action.clone()
            after = env.frame()
            terminated, truncated = term.cpu().numpy().copy(), trunc.cpu().numpy().copy()
            values.update(terminated=terminated,truncated=truncated,root=after['root'],gravity=after['gravity'],
                          feet=after['feet'],body_velocity=after['body_velocity'])
            reset_filter = np.zeros(n,bool)
            terminal_return = np.zeros(n)
            for key,mask in (('preparation',prep),('policy',policy),('waiting',phase==2)): counts[key] += mask
            if ((terminated | truncated) & prep).any(): raise RuntimeError('standing preparation terminated')
            preparation[prep] += 1
            finished = []
            for i in np.flatnonzero(policy):
                age[i] += 1
                spec, root = specs[i], after['root'][i]
                geometry = env.geometry[f'{spec["condition"]}:{spec["row"]}']
                positions[i].append(float(root[0])); progress[i] = max(progress[i],float(root[0]-x0[i]))
                velocity_error_sum[i] += float(((after['body_velocity'][i]-values['command'][i,:2])**2).sum())
                stair = bool(geometry['fronts'])
                tilt = np.degrees(np.arccos(np.clip(-after['gravity'][i,2],-1.,1.)))
                clear = (root[0]>geometry['goal_x'] and .18<after['clearance'][i]<.55 and tilt<44.
                    and after['feet'][i,:,0].min()>geometry['fronts'][-1]+.05
                    and after['feet'][i,:,2].min()>geometry['tops'][-1]-.01
                    and after['supports'][i]>=2 and not after['torso_contact'][i]) if stair else (root[0]-x0[i]>=3. and after['supports'][i]>=2)
                dwell[i] = dwell[i]+.02 if clear else 0.
                cap = (100 if args.smoke_index==2 and spec['arm']=='Frozen' else 150) if args.kind=='smoke' else 1000
                outcome = None
                if terminated[i]: outcome = 'posture_failure'
                elif abs(root[1])>1.: outcome = 'lateral_exit'
                elif dwell[i]>=(.26 if stair else .25)-1.e-8: outcome = 'success'
                elif truncated[i] or age[i]>=cap:
                    outcome = 'smoke_cutoff' if args.kind=='smoke' else 'stuck' if stair and root[0]-positions[i][max(0,len(positions[i])-251)]<.15 else 'timeout'
                if outcome:
                    displacement = float(root[0]-x0[i])
                    value = episode_return(displacement,outcome=='success',outcome=='posture_failure')
                    terminal_return[i] = value
                    stats = heads[i].finish(value,learn=args.kind=='train')
                    length = geometry['goal_x']-x0[i] if stair else 3.
                    records.append(dict(**spec,episode=int(episodes[i]),outcome=outcome,policy_steps=int(age[i]),
                        control_step=tick+1,final_root=root.tolist(),start_x=float(x0[i]),final_forward_displacement=displacement,
                        mean_progress=float(min(1.,progress[i]/length)),xy_velocity_error_sum=float(velocity_error_sum[i]),
                        xy_velocity_rmse=float(np.sqrt(velocity_error_sum[i]/age[i])),**stats))
                    phase[i] = 2; code[i] = 0; finished.append(i)
            if finished:
                safety.reset_temporal_state(torch.tensor(finished,device=device)); reset_filter[finished] = True
            ready = np.flatnonzero(prep & (preparation==50))
            if len(ready):
                index = torch.tensor(ready,device=device)
                env.base.episode_length_buf[index] = 0; env.base.action_manager.reset(index)
                safety.reset_temporal_state(index); reset_filter[ready] = True; previous[index] = 0
                phase[ready] = 1; age[ready] = 0; x0[ready] = after['root'][ready,0]
                env.policy_active[:] = phase==1; env.policy_age[:] = age; env.refresh()
                prepared = env.prepared_state(ready.tolist())
                prepared_states.append(dict(control_step=tick+1,ids=ready,episode=episodes[ready],**prepared))
                for k in range(0,len(ready),2):
                    if ready[k]%2 or ready[k+1]!=ready[k]+1: raise ValueError('unpaired preparation')
                    errors = {key:float(np.abs(value[k]-value[k+1]).max()) for key,value in prepared.items()}
                    pairs.append(dict(session=specs[ready[k]]['session'],episode=int(episodes[ready[k]]),errors=errors))
                    if max(errors.values())>1.e-6: raise ValueError('paired preparation mismatch')
            reset_ids = []
            if args.kind in ('train','smoke') and args.steps-tick-1>50:
                for i in range(0,n,2):
                    if phase[i]==phase[i+1]==2 and episodes[i]<399: reset_ids.extend([i,i+1])
                if reset_ids:
                    episodes[reset_ids] += 1; resets.append(env.reset(reset_ids,episodes))
                    phase[reset_ids] = 0; preparation[reset_ids] = age[reset_ids] = 0
                    dwell[reset_ids] = progress[reset_ids] = velocity_error_sum[reset_ids] = 0
                    code[reset_ids] = 0
                    for i in reset_ids: positions[i] = []
                    index = torch.tensor(reset_ids,device=device)
                    safety.reset_temporal_state(index); previous[index] = 0; reset_filter[reset_ids] = True
            values.update(terminal_return=terminal_return,filter_reset_after=reset_filter)
            writer.append(values)
            if args.kind=='train' and tick+1 in MARKS: checkpoint(f'at_{tick+1}.pt',tick+1)
            if (tick+1)%1000==0:
                print(f'C1A {args.name}: {tick+1}/{args.steps}, episodes={len(records)}, updates={sum(h.updates for h in heads)}',flush=True)
            if args.kind in ('target','retention') and (phase==2).all(): break
    except Exception:
        persist('partial_report.json'); checkpoint('failure.pt',completed_steps)
        save(args.output/'failure_clock.json',dict(completed_control_steps=completed_steps,trace_steps=writer.steps,
            native_physics_counter_delta=env.base._sim_step_counter-physics_start))
        raise
    if digest(env.actor)!=base_digest or sha(MODEL_PATHS['B'])!=HASHES['B']: raise ValueError('base changed')
    if any(p.grad is not None or p.requires_grad for p in env.actor.parameters()): raise ValueError('base gradients enabled')
    if any(h.digest()!=initial_digests[i] for i,h in enumerate(heads) if specs[i]['arm']=='Frozen' or args.kind!='train'):
        raise ValueError('frozen parameters changed')
    if sum(int(value.sum()) for value in counts.values())!=completed_steps*n: raise ValueError('missing control accounting')
    if args.kind in ('target','retention') and len(records)!=n: raise ValueError('incomplete evaluation')
    if args.kind=='smoke' and (not (episodes>=2).all() or any(h.updates for h in heads)): raise ValueError('smoke incomplete')
    if args.kind=='smoke' and args.smoke_index==2 and counts['waiting'][::2].min()<50: raise ValueError('no asynchronous waiting')
    if args.kind=='train' and any(h.updates==0 for i,h in enumerate(heads) if specs[i]['arm']=='Updated'): raise ValueError('no training updates')
    checkpoint('final.pt',completed_steps); persist('episodes.json')
    save(args.output/'report.json',dict(status='passed',name=args.name,kind=args.kind,environments=n,
        control_steps=completed_steps,interactions=completed_steps*n,complete_episodes=len(records),
        adapter_updates=sum(h.updates for h in heads),session_updates=[h.updates for h in heads],
        base_optimizer_updates=0,base_unchanged=True,frozen_unchanged=True,base_actor_digest=base_digest,
        base_checkpoint_sha256=HASHES['B'],initial_head_digests=initial_digests,final_head_digests=[h.digest() for h in heads],
        paired_max_error=max(max(p['errors'].values()) for p in pairs),counts=counts,noise_sha256=noise_sha,
        incomplete_policy_rows=np.flatnonzero(phase==1),incomplete_episodes_used_for_update=False,
        pending_complete_episodes=[len(h.batch) for h in heads],final_sha256=sha(args.output/'final.pt'),
        trace_index_sha256=sha(args.output/'trace_index.json')))
    print(f'COMPLETE {args.name}: episodes={len(records)}, updates={sum(h.updates for h in heads)}',flush=True)
