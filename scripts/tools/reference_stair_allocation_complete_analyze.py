"""Describe the full frozen capability tradeoff without a zero-failure acceptance gate."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from reference_stair_allocation_complete import OUTPUT, DOCS, PREVIOUS, LIMIT, verify_core, verify_retention
from reference_stair_allocation_evaluate import read, checked_training
from reference_stair_allocation import PARENT, DIGEST
from reference_stairs_eval import save, sha
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_core500_analyze import write_csv
from reference_low_stairs_equal100_analyze import validate_trace, heading_error
from reference_low_stairs_heading_analyze import episode_metrics
from reference_stair_allocation_analyze import retention_metrics, actor_adapter
from reference_turn_analyze import analyze as analyze_turn, WINDOWS
from reference_turn_diagnose import CASES


def counts(records):
    result=Counter(r['outcome'] for r in records)
    return dict(starts=len(records),success=result['success'],survived=result['survived'],
        orientation_failure=result['orientation_failure'],stuck=result['stuck'],timeout=result['timeout'],
        lateral_exit=result['lateral_exit'],success_rate=result['success']/len(records) if records else None)


def outcome_pairs(old,new,kind):
    a={r['env_id']:r for r in old}; b={r['env_id']:r for r in new}
    assert a.keys()==b.keys() and len(a)==len(old) and len(b)==len(new)
    target='survived' if kind=='turn' else 'success'
    result=[]
    for i in sorted(a):
        x,y=a[i],b[i]
        result.append(dict(env_id=i,before=x['outcome'],after=y['outcome'],
            before_duration_s=x['duration_s'],after_duration_s=y['duration_s'],
            gained_success=x['outcome']!=target and y['outcome']==target,
            lost_success=x['outcome']==target and y['outcome']!=target,
            new_orientation_failure=x['outcome']!='orientation_failure' and y['outcome']=='orientation_failure',
            earlier_orientation_failure=x['outcome']==y['outcome']=='orientation_failure' and y['duration_s']<x['duration_s']-1e-8,
            resolved_orientation_failure=x['outcome']=='orientation_failure' and y['outcome']!='orientation_failure'))
    return result


def paired_turn_windows(old,new,profile):
    """Keep empty/lost windows visible, even when no paired mean can be computed."""
    result=[]
    for case in CASES:
        for window in WINDOWS:
            a={r['env_id']:r for r in old if r['case']==case and r['window']==window}
            b={r['env_id']:r for r in new if r['case']==case and r['window']==window}
            assert a.keys()==b.keys()
            ai={i for i,r in a.items() if r['complete_window']}
            bi={i for i,r in b.items() if r['complete_window']}
            common=sorted(ai&bi)
            eligible=case not in ('standing','forward','low_left','low_right')
            row=dict(profile=profile,case=case,window=window,starts=len(a),
                before_complete=len(ai),after_complete=len(bi),paired_complete=len(common),
                lost_complete=len(ai-bi),gained_complete=len(bi-ai),tracking_in_scope=eligible,
                before_actual=None,after_actual=None,before_mae=None,after_mae=None,mae_delta=None,
                descriptive_regression=False)
            if common:
                for label,values in [('before',a),('after',b)]:
                    row[f'{label}_actual']=float(np.mean([values[i]['yaw_actual'] for i in common]))
                    row[f'{label}_mae']=float(np.mean([values[i]['yaw_mae'] for i in common]))
                delta=row['after_mae']-row['before_mae'];row['mae_delta']=delta
                row['descriptive_regression']=eligible and delta>.01 and (row['before_mae']==0. or delta>.1*row['before_mae'])
            result.append(row)
    return result


def copy_reports(path,out,prefix,raw):
    for filename in ('report.json','execution.json','config.json'):
        shutil.copy2(path/filename,out/f'{prefix}_{filename}')
    raw.update(p for p in path.iterdir() if p.is_file())


def plot_summary(out,stairs,turn_pairs,retention):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policies=('zero10100','A','B'); names=list(dict.fromkeys(r['name'] for r in stairs))
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for j,policy in enumerate(policies):
        rr=[next(r for r in stairs if r['name']==name and r['policy']==policy) for name in names]
        axes[0].bar(np.arange(len(names))+(j-1)*.25,[r['positive_success']/180 for r in rr],width=.25,label=policy)
        axes[1].bar(np.arange(len(names))+(j-1)*.25,[r['orientation_failure'] for r in rr],width=.25,label=policy)
    labels=[n.replace('heading_','H ').replace('training_randomized','random').replace('_','\n') for n in names]
    for ax in axes:ax.set_xticks(range(len(names)),labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.legend()
    axes[0].set(ylabel='Actual-stair success rate (180 starts/group)',ylim=(0,1.05))
    axes[1].set(ylabel='Posture failures (all 240 starts/group)')
    fig.savefig(out/'stairs_overview.png',dpi=150);plt.close(fig)
    cases=('turn_left','turn_right','fast_left','fast_right')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,profile in zip(axes,('nominal','training_randomized')):
        rr=[next(r for r in turn_pairs if r['comparator']=='zero10100' and r['profile']==profile and r['case']==case and r['window']=='steady') for case in cases]
        for side,label in [('before','zero10100'),('after','B')]:
            ax.plot(range(4),[r[f'{side}_mae'] for r in rr],marker='o',label=label)
        ax.set_xticks(range(4),[f'{case}\nn={r["paired_complete"]}/20' for case,r in zip(cases,rr)],fontsize=8)
        ax.set(title=profile,ylabel='Paired yaw MAE, 4-8 s (rad/s)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'turn_comparison.png',dpi=150);plt.close(fig)
    terrains=list(dict.fromkeys(r['terrain'] for r in retention))
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,profile in zip(axes,('nominal','training_randomized')):
        for j,policy in enumerate(policies):
            rates=[]
            for terrain in terrains:
                rr=[r for r in retention if r['model']==policy and r['profile']==profile and r['terrain']==terrain]
                rates.append(sum(r['success'] for r in rr)/sum(r['episodes'] for r in rr))
            ax.bar(np.arange(len(terrains))+(j-1)*.25,rates,width=.25,label=policy)
        ax.set_xticks(range(len(terrains)),[t.replace('_','\n') for t in terrains],fontsize=8)
        ax.set(title=profile,ylabel='Non-stair completion rate (20 starts/type)',ylim=(0,1.05));ax.grid(axis='y',alpha=.2);ax.legend()
    fig.savefig(out/'retention_comparison.png',dpi=150);plt.close(fig)


def main():
    complete=read(OUTPUT/'complete.json');plan=read(OUTPUT/'plan.json')
    assert complete['status']=='completed' and complete['new_evaluation_episodes']==LIMIT
    assert complete['new_optimizer_updates']==0 and not complete['promoted']
    assert sha(PREVIOUS/'complete.json')==plan['predecessor']['sha256']
    assert all(sha(s['path'])==s['sha256'] for s in plan['sources'])
    checkpoints={v:checked_training(v) for v in ('A','B')}
    out=DOCS/'results';out.mkdir(parents=True,exist_ok=True)
    by_name={}
    for entry in complete['matrix']:by_name.setdefault(entry['name'],{})[entry['variant']]=entry
    assert len(by_name)==10
    stairs=[];stair_groups=[];episodes=[];turns=[];turn_groups=[];turn_windows=[];turn_pairs=[];pairs=[]
    checks=[];raw=set();adapters={}
    for name,entries in by_name.items():
        assert set(entries)=={'A','B'};entry=entries['A']
        paths={'zero10100':Path(entry['baseline']),**{v:Path(entries[v]['path']) for v in ('A','B')}}
        reports={v:read(p/'report.json') for v,p in paths.items()};windows={}
        for policy,path in paths.items():
            report=reports[policy];cfg=read(path/'config.json')
            assert same_initial(path,paths['zero10100']) and same_configuration(cfg,read(paths['zero10100']/'config.json'))
            if policy!='zero10100':verify_core(entry,path,checkpoints[policy][1],f'allocation_{policy.lower()}10200',paths['zero10100'])
            copy_reports(path,out,f'{name}_{policy}',raw)
            if entry['kind']=='turn':
                _,data,wr,gr=analyze_turn(path)
                assert all(np.isfinite(v).all() for v in data.values())
                for record in report['records']:
                    active=data['active'][:,record['env_id']];n=int(active.sum())
                    assert active[:n].all() and not active[n:].any() and abs(n*.02-record['duration_s'])<1e-5
                windows[policy]=wr
                turn_windows.extend(dict(policy=policy,**r) for r in wr)
                turn_groups.extend(dict(policy=policy,**r) for r in gr)
                turns.append(dict(policy=policy,profile=entry['profile'],**counts(report['records']),
                    posture_before_command=sum(r['outcome']=='orientation_failure' and r['duration_s']<=2. for r in report['records']),
                    posture_after_command=sum(r['outcome']=='orientation_failure' and r['duration_s']>2. for r in report['records'])))
            else:
                with np.load(path/'trace.npz') as archive:data={k:archive[k] for k in archive.files}
                validate_trace(report,data)
                if entry['heading']:heading_error(path,data)
                episodes.extend(dict(name=name,policy=policy,**episode_metrics(r,data,cfg,entry['heading'])) for r in report['records'])
                positive=[r for r in report['records'] if r['height']>0]
                stairs.append(dict(name=name,policy=policy,profile=entry['profile'],direction=entry['direction'],heading=entry['heading'],
                    **counts(report['records']),positive_starts=len(positive),positive_success=sum(r['outcome']=='success' for r in positive)))
                for height in (0.,.02,.03,.05):
                    for speed in (.32,.4,.6):
                        selected=[r for r in report['records'] if r['height']==height and r['speed']==speed]
                        stair_groups.append(dict(name=name,policy=policy,height=height,speed=speed,**counts(selected)))
            del data
            checks.append(dict(name=name,policy=policy,initial_exact=True,configuration_exact=True,finite_trace=True,first_episodes=True))
        for comparator in ('zero10100','A'):
            pairs.extend(dict(name=name,kind=entry['kind'],profile=entry['profile'],comparator=comparator,**r)
                for r in outcome_pairs(reports[comparator]['records'],reports['B']['records'],entry['kind']))
            if entry['kind']=='turn':turn_pairs.extend(dict(comparator=comparator,**r) for r in paired_turn_windows(windows[comparator],windows['B'],entry['profile']))
    retention=retention_metrics(complete,out);retention_summary=[];retention_pairs=[]
    for item in complete['retention']['matrix']:
        policy=item['model'];profile=item['profile'];path=Path(item['path']);baseline=OUTPUT/'retention/zero10100'/profile
        digest=DIGEST if policy=='zero10100' else checkpoints[policy][1]
        report=verify_retention(path,f'allocation_retention_{policy.lower()}',profile,digest,baseline)
        with np.load(path/'trace.npz') as data:assert all(np.isfinite(data[k]).all() for k in data.files)
        retention_summary.append(dict(policy=policy,profile=profile,**counts(report['records'])))
        copy_reports(path,out,f'retention_{policy}_{profile}',raw)
        if policy=='B':
            for comparator in ('zero10100','A'):
                old=read(OUTPUT/'retention'/comparator/profile/'report.json')
                retention_pairs.extend(dict(profile=profile,comparator=comparator,**r) for r in outcome_pairs(old['records'],report['records'],'stairs'))
        adapters[f'retention_{policy}_{profile}']=actor_adapter(PARENT if policy=='zero10100' else checkpoints[policy][0],path)
    for name,rows in [('stairs_summary',stairs),('stairs_groups',stair_groups),('stairs_episodes',episodes),
        ('turn_summary',turns),('turn_groups',turn_groups),('turn_windows',turn_windows),('turn_pairs',turn_pairs),
        ('outcome_pairs',pairs),('retention_summary',retention_summary),('retention_pairs',retention_pairs)]:write_csv(out/f'{name}.csv',rows)
    source_files=[Path(__file__),ROOT/'scripts/tools/reference_stair_allocation_analyze.py',ROOT/'scripts/tools/reference_turn_analyze.py',
        ROOT/'scripts/tools/reference_low_stairs_heading_analyze.py']
    assert sha(PARENT)==DIGEST
    summary=dict(status='completed',status_scope='evidence checks and full matrix, not deployment approval',new_optimizer_updates=0,
        new_evaluation_episodes=5600,previous_candidate_episodes_reused=480,baseline_core_episodes_reused=2560,
        candidate_core_and_all_retention_episodes=6080,unique_analyzed_episodes=8640,
        behavior_early_stop=False,not_run=[],promoted=False,stairs=stairs,turns=turns,turn_pairs=turn_pairs,
        retention=retention,retention_summary=retention_summary,pair_checks=checks,adapters=adapters,
        descriptive_turn_regressions=[r for r in turn_pairs if r['descriptive_regression']],
        sources=[dict(path=str(p),sha256=sha(p)) for p in source_files],
        limitations=['Single seed and reused development starts; not a generalization or deployment guarantee.',
            'Heading feedback and no-feedback results must remain separate.',
            'Turning tracking uses paired complete windows; all episode failures and lost windows are reported.',
            'Frozen reference reward scores are not historical training rewards.'])
    assert len(episodes)==5760 and sum(r['starts'] for r in turns)==1920 and sum(r['starts'] for r in retention_summary)==960
    save(out/'summary.json',summary)
    raw.update(p for p in OUTPUT.iterdir() if p.is_file());raw.update(source_files)
    save(out/'raw_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(raw)])
    plot_summary(out,stairs,turn_pairs,retention)
    print('ANALYZED: 5600 new + 480 prior candidate + 2560 historical baseline episodes; no automatic promotion.',flush=True)


if __name__=='__main__':main()
