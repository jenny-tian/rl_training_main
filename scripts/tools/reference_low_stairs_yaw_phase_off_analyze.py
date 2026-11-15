"""纯转向相位关闭对照：固定验收门槛、同起点行为与足端机制分开报告。"""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_low_stairs_yaw_phase_off_evaluate import (
    ROOT,OUTPUT,DOCS,EVALUATION,ZERO_OUTPUT,ZERO_RUNS,PARENT_RUNS,GUARDED_RUNS,MODEL,
    directory,checked_training,verify_run,protocol_equal,sources,read,save,sha,same_initial,PROTOCOLS)
from reference_core500_analyze import write_csv,paired_turn_rows
from reference_turn_analyze import analyze as analyze_turn,WINDOWS
from reference_turn_diagnose import CASES
from reference_isaac_stairs_analyze import analyze as analyze_stairs
from reference_low_stairs_analyze import stairs_heights,check_log
from reference_low_stairs_equal100_analyze import validate_trace
from reference_low_stairs_zero_preservation_analyze import paired_records,heldout_changes
from reference_low_stairs_clearance_analyze import screen
from reference_low_stairs_guarded_analyze import adapter_check
from reference_low_stairs_motion_analyze import (
    npz,reconstruct_phase,turn_rows,recovery_rows,down_rows,
    OUTPUT as MOTION_OUTPUT,DOCS as MOTION_DOCS,DT)

POLICIES=('parent','guarded','zero','phaseoff')
ROOTS=dict(parent=PARENT_RUNS,guarded=GUARDED_RUNS,zero=ZERO_RUNS,phaseoff=EVALUATION)
OUTCOMES=('success','stuck','orientation_failure','lateral_exit','timeout')


def numeric_csv(path):
    with path.open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            if value=='':row[key]=None
            elif value in ('True','False'):row[key]=value=='True'
            else:
                try:row[key]=float(value)
                except ValueError:pass
    return rows


def pair_motion(rows,baseline,extra=()):
    keys=('profile','case','window',*extra)
    metadata=set(keys)|{'policy','repeat','env_id','complete','outcome'}
    metrics=[k for k in rows[0] if k not in metadata]
    out=[]
    for identity in sorted({tuple(r[k] for k in keys) for r in rows}):
        maps={p:{r['repeat']:r for r in rows if r['policy']==p and tuple(r[k] for k in keys)==identity}
              for p in (baseline,'phaseoff')}
        ids=sorted(set(maps[baseline])&set(maps['phaseoff']))
        row=dict(zip(keys,identity),baseline_policy=baseline,n=len(ids),starts=20)
        for p,m in maps.items():
            prefix='baseline' if p==baseline else 'candidate'
            for k in metrics:
                values=[m[i][k] for i in ids if m[i][k] is not None]
                row[f'{prefix}_{k}']=float(np.mean(values)) if values else None
        out.append(row)
    return out


def approved_screen(tables):
    def parent_rows(name):
        return [dict(r,policy='baseline' if r['policy']=='parent' else 'candidate') for r in tables[name]
                if r['policy'] in ('parent','phaseoff')]
    result=screen(parent_rows('stairs_heights'),parent_rows('stairs_groups'),
        [r for r in tables['turn_pairs'] if r['baseline_policy']=='parent'],
        [dict(r,new_failures=r['new_vs_parent']) for r in tables['turn_failures'] if r['policy']=='phaseoff'])
    checks=[]
    for speed,target in ((.32,15),(.4,20),(.6,14)):
        r=next(r for r in tables['stairs_groups'] if (r['policy'],r['profile'],r['direction'],r['height'],r['speed'])==
               ('phaseoff','nominal','up',.05,speed))
        checks.append(dict(name=f'zero_up_5cm_{speed}_preserved',actual=r['success'],target=target,passed=r['success']>=target))
    for r in tables['stairs_heights']:
        if r['policy']!='phaseoff' or r['profile']!='nominal':continue
        target=49 if r['direction']=='up' and r['height']==.05 else 60
        checks.append(dict(name=f'zero_{r["direction"]}_{r["height"]}_preserved',actual=r['success'],target=target,passed=r['success']>=target))
        old=next(z for z in tables['stairs_heights'] if z['policy']=='zero' and all(z[k]==r[k] for k in ('profile','direction','height')))
        checks.append(dict(name=f'zero_{r["direction"]}_{r["height"]}_orientation',actual=r['orientation_failure'],
                           target=old['orientation_failure'],passed=r['orientation_failure']<=old['orientation_failure']))
    recovery=next(r for r in tables['turn_records'] if r['policy']=='phaseoff' and r['profile']=='training_randomized' and r['env_id']==57)
    recovered=not (recovery['outcome']=='orientation_failure' and recovery['duration_s']<=2)
    checks.append(dict(name='env57_survives_first2',actual=int(recovered),target=1,passed=recovered))
    # 命令窗缺失与低配对数另列，不靠幸存均值掩盖失败。
    pairs=[r for r in tables['turn_pairs'] if r['baseline_policy']=='parent']
    checks.append(dict(name='all_96_parent_windows_at_least10_pairs',actual=sum(r['n']>=10 for r in pairs),target=96,
                       passed=len(pairs)==96 and all(r['n']>=10 for r in pairs)))
    review=[]
    for r in tables['stairs_groups']:
        if r['policy']!='phaseoff' or r['profile']!='training_randomized':continue
        old=next(z for z in tables['stairs_groups'] if z['policy']=='zero' and all(z[k]==r[k] for k in ('profile','direction','height','speed')))
        if r['success']<old['success'] or r['orientation_failure']>old['orientation_failure']:
            review.append(dict(direction=r['direction'],height=r['height'],speed=r['speed'],
                zero_success=old['success'],candidate_success=r['success'],zero_orientation=old['orientation_failure'],
                candidate_orientation=r['orientation_failure']))
    passed=result['engineering_checks_passed'] and all(r['passed'] for r in checks)
    return dict(status='failed' if not passed else ('requires_review' if review or result['random_decreases_require_review'] else 'passed'),
                original_parent_screen=result,additional_checks=checks,random_zero_decreases_require_review=review,
                env57_record=recovery,automatic_training_extension=False)


def training_metrics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    report=read(directory(1)/'report.json');events=EventAccumulator(str(directory(1)/'training'),size_guidance={'scalars':0});events.Reload()
    tags=('Train/mean_reward','Train/mean_episode_length','Episode_Termination/bad_orientation_2',
          'Episode_Termination/time_out','Metrics/base_velocity/error_vel_xy','Metrics/base_velocity/error_vel_yaw',
          'Episode_Reward/phase_foot_trajectory_exp','Episode_Reward/stair_clearance',
          'Episode_Reward/stair_progress','Episode_Reward/stair_landing','Loss/capability_preservation')
    metrics={}
    for tag in tags:
        values=events.Scalars(tag);assert len(values)==100 and [v.step for v in values]==list(range(10000,10100))
        a=np.array([v.value for v in values]);assert np.isfinite(a).all()
        if tag=='Loss/capability_preservation':assert not a.any()
        metrics[tag]=dict(first20=float(a[:20].mean()),last20=float(a[-20:].mean()),last=float(a[-1]))
    coverage=np.array(report['coverage_by_column_level']);clearance=np.array(report['clearance_by_column_level'])
    assert coverage.shape==clearance.shape==(20,10) and coverage.sum()==9830400 and np.all(clearance<=coverage)
    assert min(report['standing_fixed_left_fixed_right_frames'][1:])>0
    return dict(updates=100,interactions=9830400,adam_steps=202000,metrics=metrics,report=report,
                execution=read(directory(1)/'execution.json'),integration=read(directory(0)/'report.json'))


def motion(tables,checks):
    historic=read(MOTION_DOCS/'artifact_manifest.json');hashes={r['path']:r['sha256'] for r in historic}
    for name in ('turn_episodes','foot_episodes','joint_episodes','recovery57'):
        path=MOTION_DOCS/f'{name}.csv';assert sha(path)==hashes[str(path)]
        tables['motion_'+name]=numeric_csv(path)
    for profile in ('nominal','training_randomized'):
        path=EVALUATION/f'turn_{profile}'
        m,tr,ini,physics=(npz(path/f'{name}.npz') for name in ('motion','trace','initial','physics_initial'))
        names,cfg,report=(read(path/f'{name}.json') for name in ('names','config','report'))
        old=npz(MOTION_OUTPUT/f'parent_{profile}'/'physics_initial.npz')
        assert old.keys()==physics.keys() and all(np.array_equal(old[k],physics[k]) for k in physics)
        a,b=npz(path/'preparation.npz'),npz(MOTION_OUTPUT/f'parent_{profile}'/'preparation.npz')
        assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
        phase=reconstruct_phase(m,ini,physics,names,cfg)
        rows,legs,joints,check=turn_rows('phaseoff',profile,m,tr,report,physics,names,phase)
        tables['motion_turn_episodes']+=rows;tables['motion_foot_episodes']+=legs;tables['motion_joint_episodes']+=joints
        checks.append(dict(run=str(path),initial_physics_and_preparation_exact=True,**check))
        if profile=='training_randomized':tables['motion_recovery57']+=recovery_rows('phaseoff',m,tr,ini,physics,names)
        del m,tr,phase
    for r in tables['motion_turn_episodes']:
        hypothetical=r.pop('phase_rate_training_scale')
        r['hypothetical_original_phase_at_train_scale']=hypothetical
        r['candidate_formula_phase_at_train_scale']=0. if r['policy']=='phaseoff' else hypothetical
    for name,extra in (('turn',()),('foot',('leg',)),('joint',('joint',))):
        tables['motion_'+name+'_pairs']=[]
        for baseline in ('parent','zero'):
            tables['motion_'+name+'_pairs']+=pair_motion(tables['motion_'+name+'_episodes'],baseline,extra)


def plots(tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=tables['turn_pairs'];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,profile in zip(axes,('nominal','training_randomized')):
        selected=[r for r in pairs if r['profile']==profile and r['case']=='fast_left' and r['baseline_policy']=='zero']
        selected.sort(key=lambda r:list(WINDOWS).index(r['window']));x=np.arange(len(selected))
        for prefix,offset,label in (('baseline',-.18,'zero'),('candidate',.18,'phase off')):
            ax.bar(x+offset,[r[prefix+'_mae'] for r in selected],.36,label=label)
        ax.set_xticks(x,[f'{r["window"]}\nn={r["n"]}' for r in selected]);ax.set(title=profile,ylabel='Paired yaw MAE (rad/s)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fast left +0.8 rad/s, fixed complete-window pairs');fig.savefig(DOCS/'turn_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for policy in ('parent','zero','phaseoff'):
        rows=[r for r in tables['motion_recovery57'] if r['policy']==policy]
        axes[0].plot([r['time_s'] for r in rows],[r['tilt_deg'] for r in rows],label=policy)
        axes[1].plot([r['time_s'] for r in rows],[r['support_count'] for r in rows],label=policy)
    for ax,title in zip(axes,('Env 57 tilt (deg)','Support feet, force-z > 1 N')):
        ax.set(title=title,xlabel='Time (s)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(DOCS/'recovery57.png',dpi=160);plt.close(fig)


def main():
    checkpoint,digest=checked_training();complete=read(EVALUATION/'complete.json')
    assert complete['status']=='passed' and complete['formal_episodes']==1600 and complete['heading_episodes']==0
    assert complete['checkpoint_sha256']==digest
    tables={k:[] for k in ('turn_groups','turn_windows','turn_pairs','turn_failures','turn_records',
                           'stairs_heights','stairs_groups','stairs_pairs','stairs_records','down_episodes')}
    checks=[];runs=[]
    for profile in ('nominal','training_randomized'):
        reports={};windows={}
        for policy in POLICIES:
            path=ROOTS[policy]/f'turn_{profile}'
            if policy=='phaseoff':verify_run(path,'turn',profile,digest,MODEL)
            report,data,rows,groups=analyze_turn(path);validate_trace(report,data)
            assert same_initial(path,PARENT_RUNS/f'turn_{profile}')
            protocol_equal(read(path/'config.json'),read(PARENT_RUNS/f'turn_{profile}/config.json'))
            tables['turn_groups']+=[dict(policy=policy,**r) for r in groups]
            tables['turn_windows']+=[dict(policy=policy,**r) for r in rows]
            tables['turn_records']+=[dict(policy=policy,profile=profile,**r) for r in report['records']]
            reports[policy]=report;windows[policy]=rows;checks.append(check_log(path));runs.append((policy,path));del data
        for old in POLICIES[:-1]:
            paired_records(reports[old],reports['phaseoff'],'turn')
            tables['turn_pairs']+=[dict(baseline_policy=old,**r) for r in paired_turn_rows(windows[old],windows['phaseoff'],profile)]
        for policy,report in reports.items():
            for case in CASES:
                records=[r for r in report['records'] if r['case']==case]
                row=dict(policy=policy,profile=profile,case=case,starts=len(records),
                    initial_failures=sum(r['outcome']=='orientation_failure' and r['duration_s']<=2 for r in records),
                    after_command_failures=sum(r['outcome']=='orientation_failure' and r['duration_s']>2 for r in records))
                for old in POLICIES[:-1]:
                    a={r['env_id']:r for r in reports[old]['records']}
                    row[f'new_vs_{old}']=sum(r['outcome']=='orientation_failure' and a[r['env_id']]['outcome']!='orientation_failure' for r in records)
                    row[f'recovered_vs_{old}']=sum(r['outcome']!='orientation_failure' and a[r['env_id']]['outcome']=='orientation_failure' for r in records)
                tables['turn_failures'].append(row)
        for direction in ('up','down'):
            reports={}
            for policy in POLICIES:
                path=ROOTS[policy]/f'{direction}_{profile}'
                if policy=='phaseoff':verify_run(path,'stairs',profile,digest,MODEL,direction)
                report,cfg,data,*_=analyze_stairs(path);validate_trace(report,data)
                assert same_initial(path,PARENT_RUNS/f'{direction}_{profile}')
                protocol_equal(cfg,read(PARENT_RUNS/f'{direction}_{profile}/config.json'))
                tables['stairs_heights']+=[dict(policy=policy,**r) for r in stairs_heights(report)]
                tables['stairs_records']+=[dict(policy=policy,profile=profile,direction=direction,**r) for r in report['records']]
                tables['stairs_groups']+=[dict(policy=policy,profile=profile,direction=direction,height=g['height'],speed=g['speed'],
                    starts=g['episodes'],**{k:g['outcomes'].get(k,0) for k in OUTCOMES}) for g in report['groups']]
                reports[policy]=report;checks.append(check_log(path));runs.append((policy,path));del data
                if direction=='down':tables['down_episodes']+=down_rows(policy,ROOTS[policy],profile)[0]
            for old in POLICIES[:-1]:
                a,b=paired_records(reports[old],reports['phaseoff'],'stairs')
                for height in (0.,.02,.03,.05):
                    for speed in (.32,.4,.6):
                        ids=[i for i,r in a.items() if (r['height'],r['speed'])==(height,speed)]
                        tables['stairs_pairs'].append(dict(profile=profile,direction=direction,baseline_policy=old,
                            height=height,speed=speed,starts=len(ids),
                            gained_success=sum(a[i]['outcome']!='success' and b[i]['outcome']=='success' for i in ids),
                            lost_success=sum(a[i]['outcome']=='success' and b[i]['outcome']!='success' for i in ids),
                            new_orientation_failure=sum(a[i]['outcome']!='orientation_failure' and b[i]['outcome']=='orientation_failure' for i in ids)))
    screening=approved_screen(tables);motion(tables,checks);plots(tables)
    for name,rows in tables.items():write_csv(DOCS/f'{name}.csv',rows)
    training=training_metrics();save(DOCS/'training_summary.json',training)
    adapter=adapter_check(checkpoint,EVALUATION);heldout=heldout_changes(checkpoint)
    historical={r['path']:r['sha256'] for r in read(ROOT/'docs/lite3-low-stairs-zero-preservation-20261115/raw_manifest.json')}
    raw={};reused=0
    for policy,path in runs:
        if policy=='phaseoff':continue
        for p in path.iterdir():
            if p.is_file():
                assert sha(p)==historical[str(p)];reused+=1
                raw[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    for p in OUTPUT.rglob('*'):
        if not p.is_file():continue
        raw[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        if p.suffix=='.json':
            target=DOCS/'runs'/p.relative_to(OUTPUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for p in (ROOT/'scripts/tools/reference_low_stairs_yaw_phase_off_evaluate.py',Path(__file__)):
        raw[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    tests=[]
    for name in ('train_tests.xml','evaluation_tests.xml','analysis_tests.xml'):tests+=ET.parse(DOCS/name).getroot().findall('testsuite')
    assert all(int(s.get(k))==0 for s in tests for k in ('errors','failures','skipped'))
    for phase in (0,1):checks.append(check_log(directory(phase)))
    save(DOCS/'summary.json',dict(status='passed',scope='来源与完整性；行为结果见 screening',updates=100,formal_episodes=1600,
        checkpoint=str(checkpoint),checkpoint_sha256=digest,screening=screening,heldout=heldout,
        **{k:tables[k] for k in ('turn_pairs','turn_failures','stairs_heights','stairs_groups','stairs_pairs','motion_turn_pairs','motion_foot_pairs')}))
    save(DOCS/'verification.json',dict(status='passed',adapter=adapter,checks=checks,reused_files_verified=reused,
        tests_passed=sum(int(s.get('tests')) for s in tests),implementation_sources=sources(),
        evaluation_elapsed_s=sum(read(p/'execution.json')['elapsed_s'] for policy,p in runs if policy=='phaseoff')))
    save(DOCS/'raw_manifest.json',list(raw.values()))
    print('YAW PHASE OFF ANALYSIS COMPLETE:',screening['status'],flush=True)


if __name__=='__main__':main()
