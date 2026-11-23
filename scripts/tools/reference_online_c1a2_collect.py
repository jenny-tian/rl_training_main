"""由 reference_online_c1a_collect.py（57ad0de）按固定替换生成：航向闭环、箱体终点、按路线长度归一化的回报。"""
from __future__ import annotations

from dataclasses import fields
import hashlib

import numpy as np
import torch

from reference_online_c1a2 import OUTPUT, MARKS, rows, protocol, episode_return, NOISE_SEED, VARIANT, REPLICAS, SWING, SWING_BASIS, SOURCE, WINDOWED, WINDOW_INITIAL, LIBRARY, SELECT, SELECT_SEED, CANDIDATE, LEAD, RATE, STALL
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
    from reference_online_c1a2_env import Environment
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    script = getattr(args,'script',None)
    specs = rows(args.kind,part=args.part,smoke_index=args.smoke_index,script=script)
    env = Environment(args,specs)
    from rl_training.ea_lpl.c1a import Session, exploration_table
    from rl_training.ea_lpl.c1b import digest
    ea, bh, bs = VARIANT=='ea', VARIANT=='bh', SWING_BASIS
    bsea = VARIANT=='s2ea'  # C1-b-EA：摆动幅度的探索标准差按执行响应调整
    if bh or bs: from rl_training.ea_lpl import c1b_h
    if bs: from rl_training.ea_lpl import c1b_s
    if bsea: from rl_training.ea_lpl import c1b_s_ea
    if WINDOWED: from rl_training.ea_lpl import c1b_w
    if ea: from rl_training.ea_lpl.c1a_ea import SessionEA as Session
    from rl_training.ea_lpl.basis import basis
    from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
    from rl_training.ea_lpl.residual_bridge import ResidualSafetyBridge
    from rl_training.online_adaptation.safety import SafetyLimits
    from reference_online_interface import clip_action
    n, device = len(specs), env.device
    noise = c1b_w.exploration_table(seed=NOISE_SEED) if WINDOWED else c1b_s.exploration_table(seed=NOISE_SEED) if bs else c1b_h.exploration_table(seed=NOISE_SEED) if bh else exploration_table(seed=NOISE_SEED)
    noise_sha = hashlib.sha256(noise.tobytes()).hexdigest()
    np.save(args.output/'noise_table.npy',noise)
    parallel = REPLICAS>1 and args.kind in ('train','smoke')  # 一个会话的并行环境共用一个均值
    windowed = WINDOWED and parallel  # 回合内按窗口换扰动并更新；评估时与 C1-b-S 相同
    width = 2*REPLICAS if parallel else 2  # 一起重置的相邻环境数
    if parallel:
        if bsea: ParallelSession, Slot = c1b_s_ea.ParallelSessionEA, c1b_s_ea.SlotEA
        elif bs:
            import functools
            ParallelSession, Slot = functools.partial(c1b_s.ParallelSession,active=protocol().get('active_parameters')), c1b_s.Slot
        elif bh: ParallelSession, Slot = c1b_h.ParallelSession, c1b_h.Slot
        else: from rl_training.ea_lpl.c1a_m4 import ParallelSession, Slot
        if windowed:
            Slot, initial = c1b_w.Slot, None
            if WINDOW_INITIAL:  # 只有 Updated 组从迁移来的平均参数出发；Frozen 组保持为 0
                source = torch.load(WINDOW_INITIAL,map_location='cpu',weights_only=True)
                if source['control_steps']!=60000 or source['kind']!='train' or len(source['heads'])!=120: raise ValueError('unexpected initial snapshot')
                initial = torch.stack([torch.as_tensor(h['w'],dtype=torch.float64) for h in source['heads'][1::2]]).mean(0)
            if LIBRARY or SELECT:  # Updated 组先从参数库里选；Frozen 组与 C1-b-W 相同
                from rl_training.ea_lpl import c1b_l, c1b_d
                from reference_online_transfer import library
                entries = library()
            if STALL: from rl_training.ea_lpl import c1b_r
            make = lambda s,arm: (c1b_r.StallSession(args.kind=='train',noise[s]) if STALL and arm=='Updated' else
                c1b_d.SelectSession(args.kind=='train',noise[s],entries,seed=[SELECT_SEED,s],expand=VARIANT=='d2') if SELECT and arm=='Updated' else
                c1b_l.LibrarySession(args.kind=='train',noise[s],entries) if LIBRARY and arm=='Updated' else
                c1b_w.WindowSession(args.kind=='train' and arm=='Updated',noise[s],active=protocol().get('active_parameters'),initial=initial if arm=='Updated' else None))
            groups = {(s,arm):make(s,arm) for s in range(60) for arm in ('Frozen','Updated')}
        else: groups = {(s,arm):ParallelSession(args.kind=='train' and arm=='Updated') for s in range(60) for arm in ('Frozen','Updated')}
        heads = [Slot(groups[s['session'],s['arm']],s['repeat']) for s in specs]
        for k in range(0,n,width):
            if len({s['session'] for s in specs[k:k+width]})!=1 or k//width!=specs[k]['session']: raise ValueError('session rows not contiguous')
    elif bs:
        heads = [c1b_s.Slot(c1b_s.ParallelSession(False,replicas=1),0) for s in specs]
    elif bh:
        heads = [c1b_h.Slot(c1b_h.ParallelSession(False,replicas=1),0) for s in specs]
    else:
        heads = [Session(args.kind=='train' and s['arm']=='Updated') for s in specs]
    if script:
        for head,spec in zip(heads,specs): head.load_evaluation_mean(c1b_h.scripted(spec['script']))
    elif args.kind in ('target','retention'):
        if SOURCE:  # 参数迁移：Updated 组用在另一个地形上学到的参数
            from reference_online_transfer import load_source
            checkpoint = load_source()
        elif LEAD or RATE:  # 提前起抬、每步变化上限的脚本化检查：Updated 组都用库里的一组参数
            from reference_online_c1a2 import FIXED_ENTRY
            from reference_online_transfer import library_heads
            checkpoint = dict(control_steps=0,heads=library_heads(FIXED_ENTRY))
        elif CANDIDATE:  # 候选参数的不学习评估：Updated 组都用同一组固定参数
            from reference_online_transfer import candidate_heads
            checkpoint = dict(control_steps=0,heads=candidate_heads(CANDIDATE))
        else:
            path = OUTPUT/'train'/('initial.pt' if args.mark==0 else f'at_{args.mark}.pt')
            checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['control_steps'] != args.mark: raise ValueError('wrong evaluation snapshot')
        for head,spec in zip(heads,specs):
            head.load_evaluation_mean(checkpoint['heads'][2*spec['session']+int(spec['arm']=='Updated')])
    initial_digests = [h.digest() for h in heads]
    base_digest = digest(env.actor)
    swing = dict(SWING[script]) if script in SWING else None
    if swing:
        # 只在脚本化检查里放大残差上限或放开每步变化上限；确认任务的两个值由命令行给出。
        from rl_training.ea_lpl.swing_lift import swing_request
        if swing['scale'] is None: swing.update(scale=args.swing_scale,rate=args.swing_rate)
        limit = torch.tensor(protocol()['residual_limits'],device=device)*swing['scale']
        safety = CurrentModelSafetyFilter(SafetyLimits(residual_limits=limit,residual_rate_limit=swing['rate'] or float(limit.max())))
        swing_groups = [s['swing'] for s in specs]
        save(args.output/'swing.json',dict(swing,residual_limits=limit.tolist(),residual_rate_limit=safety.limits.residual_rate_limit))
    else:
        safety = CurrentModelSafetyFilter(SafetyLimits(residual_limits=torch.tensor(protocol()['residual_limits'],device=device),
            **(dict(residual_rate_limit=protocol()['residual_rate_limit']) if bs else {})))
    bridge = ResidualSafetyBridge(env.adapter,safety,action_clip=clip_action)
    lead = None
    if LEAD:
        from reference_online_c1a2 import LEAD_RULE
        from rl_training.ea_lpl.swing_lead import SwingLead
        lead = SwingLead(LEAD_RULE,n,device)
    v = basis('sag',device=device)
    projector = (v/(v**2).sum(0)).double()  # 执行残差在两个基方向上的系数
    executed_requested, requested_squared = np.zeros((n,2)), np.zeros((n,2))
    previous = torch.zeros(n,12,device=device)
    timeout = torch.zeros(n,1,dtype=torch.bool,device=device)
    code = np.zeros((n,4 if bs else 2),np.float64)
    if bh or bs:
        scanner = env.base.scene['height_scanner']
        terrain_features = c1b_h.TerrainFeatures(scanner.ray_starts[0,:,:2])
        if not torch.equal(scanner.ray_starts[:,:,:2],scanner.ray_starts[:1,:,:2].expand(n,-1,-1)): raise ValueError('ray grids differ')
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
        saved = [groups[s,arm].snapshot() for s in range(60) for arm in ('Frozen','Updated')] if parallel else [h.snapshot() for h in heads]
        torch.save(dict(heads=saved,control_steps=step,
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
                exploration = noise[specs[i]['session'],REPLICAS*episodes[i]+specs[i]['repeat']] if args.kind in ('train','smoke') and not windowed else None
                code[i] = heads[i].begin(exploration)
            if windowed:
                for i in np.flatnonzero(policy): code[i] = heads[i].requested_code()
            if bh or bs: features = terrain_features(scanner.data.ray_hits_w[:,:,2]).cpu().numpy()
            if bh:
                for i in np.flatnonzero(policy & (age%c1b_h.INTERVAL==0)): code[i] = heads[i].choose(features[i])
            for i in np.flatnonzero(policy):
                if not np.array_equal(code[i],heads[i].requested_code()): raise ValueError('code changed inside episode')
            with torch.no_grad():
                raw = env.actor(env.obs['policy'])
                raw[torch.as_tensor(~policy,device=device)] = 0
                if bs:
                    stance = env.sensor.data.net_forces_w[:,env.contact_ids,2]>1.
                    request_stance = lead.step(stance,clip_action(raw)) if lead else stance
                    request = c1b_s.residual_request(torch.tensor(code,dtype=torch.float32,device=device),request_stance,torch.as_tensor(features[:,0],device=device))
                else: request = torch.einsum('jk,nk->nj',v,torch.tanh(torch.tensor(code,dtype=torch.float32,device=device)))
                if swing:
                    stance = env.sensor.data.net_forces_w[:,env.contact_ids,2]>1.
                    request = request+swing_request(stance,torch.as_tensor(features[:,0],device=device),swing_groups,swing['scale'])
                request[torch.as_tensor(~policy,device=device)] = 0
                trace = bridge.step(raw,request,timeout)
                state = trace.safety_state
                if torch.count_nonzero(trace.executed_action[torch.as_tensor(~policy,device=device)]):
                    raise ValueError('nonzero preparation or waiting action')
                components = safety.execution_components()
                executed = (components['learned_effect'].double()@projector).cpu().numpy()
                if bsea:
                    done, squared = c1b_s_ea.execution_sums(request,components['learned_effect'])
                    executed_requested[policy] += done[policy]; requested_squared[policy] += squared[policy]
                elif not bs:
                    requested = np.tanh(code)
                    executed_requested[policy] += (executed*requested)[policy]; requested_squared[policy] += (requested**2)[policy]
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
                if bh or bs: values['features'] = features.astype(np.float32)
                if swing or bs: values['stance'] = stance.cpu().numpy().copy()
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
                if windowed:
                    heads[i].observe(values['stance'][i],(after['feet'][i,:,0]-before['feet'][i,:,0])/.02,
                                     features[i,0],after['root'][i,0]-before['root'][i,0])
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
                    and after['supports'][i]>=2 and not after['torso_contact'][i]) if stair else ((root[0]>geometry['goal_x'] if geometry['goal_x'] else root[0]-x0[i]>=3.) and after['supports'][i]>=2)
                dwell[i] = dwell[i]+.02 if clear else 0.
                cap = (100 if args.smoke_index==2 and spec['arm']=='Frozen' else 150) if args.kind=='smoke' else 1000
                outcome = None
                if terminated[i]: outcome = 'posture_failure'
                elif abs(root[1])>1.: outcome = 'lateral_exit'
                elif dwell[i]>=(.26 if stair else .25)-1.e-8: outcome = 'success'
                elif truncated[i] or age[i]>=cap:
                    outcome = 'smoke_cutoff' if args.kind=='smoke' else 'stuck' if geometry['goal_x'] and root[0]-positions[i][max(0,len(positions[i])-251)]<.15 else 'timeout'
                if outcome:
                    displacement = float(root[0]-x0[i])
                    length = geometry['goal_x']-x0[i] if geometry['goal_x'] else 3.
                    value = episode_return(displacement,length,outcome=='success',outcome=='posture_failure')
                    terminal_return[i] = value
                    learn = args.kind=='train'
                    stats = heads[i].finish(value,learn=learn,**(dict(executed_requested=executed_requested[i],
                        requested_squared=requested_squared[i],steps=int(age[i])) if (ea or bsea) and learn else {}),
                        **(dict(posture=outcome=='posture_failure') if windowed else {}),
                        **(dict(success=outcome=='success') if windowed and SELECT and spec['arm']=='Updated' else {}))
                    executed_requested[i] = requested_squared[i] = 0
                    records.append(dict(**spec,episode=int(episodes[i]),outcome=outcome,policy_steps=int(age[i]),
                        control_step=tick+1,final_root=root.tolist(),start_x=float(x0[i]),final_forward_displacement=displacement,
                        mean_progress=float(min(1.,progress[i]/length)),xy_velocity_error_sum=float(velocity_error_sum[i]),
                        xy_velocity_rmse=float(np.sqrt(velocity_error_sum[i]/age[i])),**stats))
                    phase[i] = 2; code[i] = 0; finished.append(i)
            if windowed:
                for group in groups.values(): group.tick(learn=args.kind=='train')
            if finished:
                safety.reset_temporal_state(torch.tensor(finished,device=device)); reset_filter[finished] = True
            ready = np.flatnonzero(prep & (preparation==50))
            if len(ready):
                index = torch.tensor(ready,device=device)
                env.base.episode_length_buf[index] = 0; env.base.action_manager.reset(index)
                safety.reset_temporal_state(index); reset_filter[ready] = True; previous[index] = 0
                phase[ready] = 1; age[ready] = 0; x0[ready] = after['root'][ready,0]
                if lead: lead.reset(index)
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
                for i in range(0,n,width):
                    if (phase[i:i+width]==2).all() and episodes[i]<400//REPLICAS-1: reset_ids.extend(range(i,i+width))
                if reset_ids:
                    episodes[reset_ids] += 1; resets.append(env.reset(reset_ids,episodes))
                    phase[reset_ids] = 0; preparation[reset_ids] = age[reset_ids] = 0
                    dwell[reset_ids] = progress[reset_ids] = velocity_error_sum[reset_ids] = 0
                    code[reset_ids] = 0
                    executed_requested[reset_ids] = requested_squared[reset_ids] = 0
                    for i in reset_ids: positions[i] = []
                    index = torch.tensor(reset_ids,device=device)
                    safety.reset_temporal_state(index); previous[index] = 0; reset_filter[reset_ids] = True
            values.update(terminal_return=terminal_return,filter_reset_after=reset_filter)
            writer.append(values)
            if args.kind=='train' and tick+1 in MARKS: checkpoint(f'at_{tick+1}.pt',tick+1)
            if (tick+1)%1000==0:
                print(f'C1A {args.name}: {tick+1}/{args.steps}, episodes={len(records)}, updates={sum(h.updates for h in heads)}'
                      +(f', mean_std={np.mean([h.std.numpy() for h in heads[1::2]],0).round(3).tolist()}' if ea else '')
                      +(f', mean_std={np.mean([g.std.numpy()[2:] for (_,arm),g in groups.items() if arm=="Updated"],0).round(3).tolist()}' if bsea and parallel else ''),flush=True)
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
    if windowed and any(g.windows==0 for g in groups.values()): raise ValueError('no window closed')
    if args.kind=='smoke' and args.smoke_index==2 and counts['waiting'][::2].min()<50: raise ValueError('no asynchronous waiting')
    if args.kind=='train' and any(h.updates==0 for i,h in enumerate(heads) if specs[i]['arm']=='Updated'): raise ValueError('no training updates')
    checkpoint('final.pt',completed_steps); persist('episodes.json')
    if windowed: save(args.output/'windows.json',{f'{s}:{arm}':group.log for (s,arm),group in groups.items()})
    if windowed and (LIBRARY or SELECT):
        save(args.output/'selection.json',dict(library=entries.tolist(),sessions=[groups[s,'Updated'].selection for s in range(60)]))
        if any(groups[s,'Updated'].selection is None for s in range(60)): raise ValueError('selection incomplete')
    save(args.output/'report.json',dict(status='passed',name=args.name,kind=args.kind,environments=n,
        control_steps=completed_steps,interactions=completed_steps*n,complete_episodes=len(records),
        adapter_updates=sum(h.updates for h in heads),session_updates=[h.updates for h in heads],
        base_optimizer_updates=0,base_unchanged=True,frozen_unchanged=True,base_actor_digest=base_digest,
        base_checkpoint_sha256=HASHES['B'],initial_head_digests=initial_digests,final_head_digests=[h.digest() for h in heads],
        paired_max_error=max(max(p['errors'].values()) for p in pairs),counts=counts,noise_sha256=noise_sha,
        incomplete_policy_rows=np.flatnonzero(phase==1),incomplete_episodes_used_for_update=False,
        pending_complete_episodes=[len(h.batch) for h in heads],final_sha256=sha(args.output/'final.pt'),
        trace_index_sha256=sha(args.output/'trace_index.json'),feedback_max_error=env.feedback_error,
        **(dict(missing_rays=terrain_features.missing) if bh or bs else {})))
    assert env.feedback_error<=1e-5
    print(f'COMPLETE {args.name}: episodes={len(records)}, updates={sum(h.updates for h in heads)}',flush=True)
