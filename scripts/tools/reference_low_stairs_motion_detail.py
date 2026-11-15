"""冻结数据的起步时序、共同下阶窗口及图表；不启动仿真或训练。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from reference_low_stairs_motion_analyze import (
    OUTPUT, DOCS, ROOT, LEGS, DT, PARENT_RUNS, EVALUATION, GUARDED_RUNS,
    npz, read, save, sha, identity, yaw, first_time, lateral_components,
)
from reference_core500_analyze import write_csv
from reference_stairs_eval import MODELS, load_actor


def common_window(traces, start, end):
    """所有策略都完整存活的同一起点；终止帧不作为完整窗口。"""
    a, b = round(start / DT), round(end / DT)
    return np.logical_and.reduce([
        (d['active'][a:b] & ~d['terminated'][a:b]).all(0) for d in traces
    ])


def tilt(gravity):
    return np.rad2deg(np.arccos(np.clip(-gravity[..., 2], -1, 1)))


def recovery():
    data = {}; checks = {}; i = 57
    for policy in ('parent', 'zero'):
        path = OUTPUT / f'{policy}_training_randomized'
        data[policy] = dict(m=npz(path/'motion.npz'), t=npz(path/'trace.npz'),
                            initial=npz(path/'initial.npz'), physics=npz(path/'physics_initial.npz'),
                            prep=npz(path/'preparation.npz'), names=read(path/'names.json'),
                            report=read(path/'report.json'))
    old, new = data['parent'], data['zero']
    assert all(np.array_equal(old['prep'][k], new['prep'][k]) for k in old['prep'])
    assert np.array_equal(old['m']['observation'][0], new['m']['observation'][0])
    action_ids = old['names']['action_joint_ids']; joint_names = old['names']['joint_names']
    scales = np.tile([.125, .25, .25], 4)
    actions = {}
    for policy, d in data.items():
        checkpoint, digest, _, _ = identity(policy); MODELS['motion_detail_'+policy] = checkpoint
        actor = load_actor('motion_detail_'+policy)
        with torch.inference_mode():
            actions[policy] = actor(torch.from_numpy(old['m']['observation'][0].copy())).numpy()
        err = float(np.abs(actions[policy]-d['t']['action'][0]).max())
        assert err < 2e-5, err
        target = d['physics']['default_joint_pos'][:, action_ids] + d['t']['action'][0]*scales
        target_err = float(np.abs(target-d['m']['joint_target'][0][:, action_ids]).max())
        assert target_err < 2e-6, target_err
        checks[policy] = dict(cpu_actor_action_max_error=err, requested_target_max_error=target_err,
                              model_sha256=digest)
    differences = (actions['zero'][i]-actions['parent'][i])*scales
    targets = [dict(joint=joint_names[j], parent=float(old['m']['joint_target'][0,i,j]),
                    zero=float(new['m']['joint_target'][0,i,j]), delta=float(differences[k]))
               for k,j in enumerate(action_ids)]
    targets.sort(key=lambda r:abs(r['delta']),reverse=True)
    result = dict(env_id=i, preparation_exact=True, initial_observation_exact=True, checks=checks,
                  first_action_target_changes=targets, policies={})
    for policy,d in data.items():
        m,t = d['m'],d['t']; n = min(100, int(t['active'][:,i].sum()))
        support = m['force'][:n,i,:,2] > 1
        angle = tilt(m['gravity'][:n,i])
        root0 = d['initial']['root'][i]; q = root0[3:7]
        result['policies'][policy] = dict(
            record=d['report']['records'][i], initial_tilt_deg=float(np.rad2deg(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))),
            preparation_initial_tilt_deg=float(tilt(d['prep']['gravity'][0,i])),
            max_tilt_first_082s_deg=float(angle[:41].max()),
            first_tilt30_s=first_time(angle>30),first_tilt40_s=first_time(angle>40),
            first_front_support_s=first_time(support[:,:2].any(-1)),
            first_three_support_s=first_time(support.sum(-1)>=3),
            first_halfsecond_support_fraction=support[:25].mean(0).tolist(),
            zero_command_until_failure=bool((m['command'][:n,i]==0).all()),
            initial_mass_kg=float(d['physics']['masses'][i].sum()),
            initial_torso_com=d['physics']['coms'][i,0,:3].tolist(),
            delay_physics_steps={k:int(v[i]) for k,v in d['physics'].items() if k.startswith('delay_')},
            timeline=[dict(time_s=(k+1)*DT, tilt_deg=float(angle[k]), support=support[k].tolist(),
                           foot_z=m['feet'][k,i,:,2].tolist(), joint_target=m['joint_target'][k,i].tolist())
                      for k in (0,4,9,14,19,24,29,39,40) if k<n])
    save(DOCS/'recovery_detail.json', result)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for policy,d in data.items():
        m,t=d['m'],d['t']; n=min(100,int(t['active'][:,i].sum())); time=np.arange(1,n+1)*DT
        axes[0,0].plot(time,tilt(m['gravity'][:n,i]),label=policy)
        axes[0,1].plot(time,(m['force'][:n,i,:,2]>1).sum(-1),label=policy)
        j=joint_names.index(targets[0]['joint'])
        axes[1,0].plot(time,m['joint_target'][:n,i,j],label=policy+' requested')
        axes[1,0].plot(time,m['joint_pos'][:n,i,j],ls='--',label=policy+' actual')
        axes[1,1].plot(time,m['feet'][:n,i,:2,2].min(-1)-.022,label=policy)
    for ax,title,ylabel in zip(axes.flat,('Env 57: body tilt','Vertical-force support > 1 N',targets[0]['joint'], 'Lowest front foot bottom'),
                               ('deg','feet','rad','m')):
        ax.set(title=title,xlabel='Time since actor start (s)',ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same initial state and physics; zero command before 2 s')
    fig.savefig(DOCS/'recovery57.png',dpi=160); plt.close(fig)
    return result


def down():
    groups=[]; cases=[]; details={}
    for profile in ('nominal','training_randomized'):
        all_data={}
        for policy, root in (('parent',PARENT_RUNS),('guarded',GUARDED_RUNS),('zero',EVALUATION)):
            path=root/f'down_{profile}'
            all_data[policy]=dict(t=npz(path/'trace.npz'),initial=npz(path/'initial.npz'),
                                  cfg=read(path/'config.json'),report=read(path/'report.json'))
        for start,end in ((0,2),(2,4),(4,6)):
            common=common_window([d['t'] for d in all_data.values()],start,end)
            a,b=round(start/DT),round(end/DT)
            for height in (0.,.02,.03,.05):
                ids=np.array([r['env_id'] for r in all_data['zero']['report']['records'] if r['height']==height and common[r['env_id']]])
                if not len(ids):continue
                for policy,d in all_data.items():
                    t,ini=d['t'],d['initial'];q=t['quat'][a:b,ids];vel=t['velocity'][a:b,ids]
                    heading=yaw(q);h0=yaw(ini['root'][ids,3:7]);dh=(heading-h0+np.pi)%(2*np.pi)-np.pi
                    y0=ini['root'][ids,1]-ini['origins'][ids,1];displacement=t['root'][b-1,ids,1]-y0
                    support=t['forces'][a:b,ids,:,2]>1
                    groups.append(dict(profile=profile,policy=policy,height=height,start=start,end=end,n=len(ids),
                        mean_signed_yaw_deg=float(np.rad2deg(dh).mean()),mean_abs_yaw_deg=float(np.abs(np.rad2deg(dh)).mean()),
                        end_signed_displacement_y=float(displacement.mean()),end_abs_displacement_y=float(np.abs(displacement).mean()),
                        yaw_velocity_mean=float(vel[...,5].mean()),body_lateral_speed_abs_mean=float(np.abs(vel[...,1]).mean()),
                        support_left=float(support[:,:,[0,2]].mean()),support_right=float(support[:,:,[1,3]].mean())))
        if profile!='nominal':continue
        for i in (135,181):
            details[i]={}
            for policy,d in all_data.items():
                t,ini=d['t'],d['initial'];r=d['report']['records'][i];n=int(t['active'][:,i].sum())
                edge=d['cfg']['geometry'][r['column']]['fronts'][0]
                h0=yaw(ini['root'][i,3:7]);h=yaw(t['quat'][:n,i]);delta=(h-h0+np.pi)%(2*np.pi)-np.pi
                first10=np.flatnonzero(np.abs(delta)>np.deg2rad(10))
                k=int(first10[0]) if len(first10) else None
                row=dict(policy=policy,env_id=i,height=r['height'],outcome=r['outcome'],
                    first_front_foot_past_edge_s=first_time((t['feet'][:n,i,:2,0]>edge).any(-1)),
                    first_yaw10_s=first_time(np.abs(delta)>np.deg2rad(10)),
                    root_x_at_yaw10=float(t['root'][k,i,0]) if k is not None else None,
                    maximum_front_foot_x_at_yaw10=float(t['feet'][k,i,:2,0].max()) if k is not None else None,
                    first_edge_x=edge,initial_yaw_deg=float(np.rad2deg(h0)),
                    final_yaw_deg=float(np.rad2deg(h[-1])))
                cases.append(row);details[i][policy]=(d,n,delta)
    write_csv(DOCS/'down_common_windows.csv',groups);write_csv(DOCS/'down_case_timing.csv',cases)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for column,i in enumerate((135,181)):
        for policy,(d,n,delta) in details[i].items():
            t=d['t'];ini=d['initial'];time=np.arange(1,n+1)*DT
            y0=ini['root'][i,1]-ini['origins'][i,1]
            axes[0,column].plot(time,np.rad2deg(delta),label=policy)
            axes[1,column].plot(time,t['root'][:n,i,1]-y0,label=policy)
        axes[0,column].set(title=f'Env {i}: heading change',ylabel='deg')
        axes[1,column].set(title=f'Env {i}: lateral displacement',ylabel='m')
    for ax in axes.flat:ax.set(xlabel='Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(DOCS/'down_drift.png',dpi=160);plt.close(fig)
    return dict(common_windows=groups,case_timing=cases)


def turn_plot():
    import csv
    import matplotlib.pyplot as plt
    with (DOCS/'feet_paired.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,profile in enumerate(('nominal','training_randomized')):
        selected=[r for r in rows if r['case']=='fast_left' and r['window']=='steady' and r['profile']==profile]
        assert len(selected)==4 and all(int(r['n'])==20 for r in selected)
        selected.sort(key=lambda r:LEGS.index(r['leg']))
        x=np.arange(4)
        for policy,offset in (('parent',-.18),('zero',.18)):
            axes[0,col].bar(x+offset,[float(r[policy+'_clearance_p95_m'])*1000 for r in selected],.36,label=policy)
            axes[1,col].bar(x+offset,[float(r[policy+'_support_slip_mean_m_s']) for r in selected],.36,label=policy)
        axes[0,col].set(title=profile+'; fast left, 20 paired starts',ylabel='Per-episode clearance p95, mean (mm)')
        axes[1,col].set(ylabel='Support slip proxy (m/s)')
    for ax in axes.flat:ax.set_xticks(np.arange(4),LEGS);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('4-8 s; clearance from flat ground; force-z > 1 N support proxy')
    fig.savefig(DOCS/'turn_feet.png',dpi=160);plt.close(fig)


def verify_archive():
    summary=read(DOCS/'summary.json');manifest=read(DOCS/'raw_manifest.json')
    for row in manifest:
        path=Path(row['path'])
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    for row in summary['implementation_sources']:
        assert sha(row['path'])==row['sha256'],row['path']
    for profile in ('nominal','training_randomized'):
        old=npz(OUTPUT/f'parent_{profile}'/'preparation.npz')
        new=npz(OUTPUT/f'zero_{profile}'/'preparation.npz')
        assert old.keys()==new.keys() and all(np.array_equal(old[k],new[k]) for k in old)
    json_copies=0
    for path in OUTPUT.rglob('*.json'):
        assert sha(path)==sha(DOCS/'runs'/path.relative_to(OUTPUT));json_copies+=1
    runs=[read(Path(e['path'])/'execution.json') for e in read(OUTPUT/'complete.json')['matrix']]
    assert [r['exit_code'] for r in runs]==[1,0,0,0]
    detail=read(DOCS/'detail_summary.json');assert detail['source_sha256']==sha(__file__)
    save(DOCS/'verification.json',dict(status='passed',scope='来源与冻结数据完整性，不表示能力通过',
        episodes=1280,optimizer_updates=0,raw_files_verified=len(manifest),reused_files_verified=summary['reused_files_verified'],
        tests_passed=summary['tests_passed'],original_json_copies_verified=json_copies,
        preparation_exact_both_profiles=True,original_execution_exit_codes=[r['exit_code'] for r in runs],
        first_run_verifier_failure_preserved=True,simulations_elapsed_s=sum(r['elapsed_s'] for r in runs),
        peak_total_gpu_used_mib=max(r['peak_total_gpu_used_mib'] for r in runs),
        source_sha256=sha(__file__),sources=summary['implementation_sources']))
    save(DOCS/'artifact_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(DOCS.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'])


def main():
    assert read(DOCS/'summary.json')['status']=='passed'
    import matplotlib
    matplotlib.use('Agg')
    result=dict(recovery=recovery(),down=down(),optimizer_updates=0,source_sha256=sha(__file__))
    turn_plot();save(DOCS/'detail_summary.json',result);verify_archive()
    print('MOTION DETAIL COMPLETE',flush=True)


if __name__=='__main__':main()
