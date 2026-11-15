"""分析系数归零单项对照；能力筛查与运行完整性分别报告。"""
from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT, MODELS, load_actor, save, sha
from reference_low_stairs_zero_preservation_train import OUTPUT, DOCS, directory, BANK, sources
from reference_low_stairs_zero_preservation_evaluate import checked_training, verify_run, EVALUATION, MODEL, PARENT_RUNS, GUARDED_RUNS, read
from reference_low_stairs_equal100_evaluate import OUTPUT as CLEARANCE, EXPECTED as CLEARANCE_SHA, GUARDED_SHA
from reference_low_stairs_evaluate import EXPECTED_SHA
from reference_low_stairs_guarded_analyze import adapter_check
from reference_low_stairs_guarded_evaluate import turn_stop_checks
from reference_low_stairs_clearance_analyze import screen
from reference_low_stairs_analyze import stairs_heights, check_log
from reference_turn_analyze import analyze as analyze_turn, WINDOWS
from reference_turn_diagnose import CASES
from reference_isaac_stairs_analyze import analyze as analyze_stairs
from reference_core500_analyze import write_csv, paired_turn_rows
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_low_stairs_equal100_analyze import (historical_heading, heading_error, validate_trace,
    stair_diagnostics, group_diagnostics, DIAGNOSIS, DT)

POLICIES = ('parent10000','clearance10100','guarded10100','zero10100')
OUTCOMES = ('success','stuck','orientation_failure','lateral_exit','timeout')


def paired_records(before, after, kind):
    """同一环境编号还必须对应同一条件和重复编号。"""
    a,b=({r['env_id']:r for r in report['records']} for report in (before,after))
    assert len(a)==len(before['records']) and len(b)==len(after['records'])
    assert a.keys()==b.keys()
    keys=('case','repeat') if kind=='turn' else ('height','speed','repeat','column')
    assert all(a[i][key]==b[i][key] for i in a for key in keys)
    return a,b


def common_windows(windows,profile):
    rows=[]
    for case in CASES:
        for window in WINDOWS:
            maps={p:{r['repeat']:r for r in windows[p] if r['case']==case and r['window']==window} for p in POLICIES}
            ids=sorted(set.intersection(*(set(m) for m in maps.values())))
            ids=[i for i in ids if all(m[i]['complete_window'] for m in maps.values())]
            row=dict(profile=profile,case=case,window=window,starts=20,n=len(ids))
            for p,m in maps.items():
                for metric in ('yaw_command','yaw_actual','yaw_mae'):
                    row[f'{p}_{metric}']=float(np.mean([m[i][metric] for i in ids])) if ids else None
            rows.append(row)
    return rows


def matched_contacts(series,profile,heading):
    rows=[]
    # 主对照仅改变动作保持系数；同一回合双方首次倾斜/终止前，共同到阶区。
    before,after=series['guarded10100'],series['zero10100']
    for i in sorted(set(before)&set(after)):
        a,b=before[i],after[i];end=min(a['cutoff'],b['cutoff'])
        mask=(a['root'][:end,0]>.85)&(b['root'][:end,0]>.85)
        row=dict(profile=profile,heading=heading,env_id=i,speed=a['speed'],exposure_s=float(mask.sum()*DT))
        for policy,values in (('guarded10100',a),('zero10100',b)):
            for feature in ('collision','wall'):
                row[f'{policy}_{feature}_foot_seconds']=float(values[feature][:end][mask].sum()*DT)
        rows.append(row)
    return rows


def training_metrics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run=directory(1);report=read(run/'report.json')
    event=EventAccumulator(str(run/'training'),size_guidance={'scalars':0});event.Reload()
    metrics={}
    for tag in ('Train/mean_reward','Train/mean_episode_length','Episode_Termination/bad_orientation_2',
                'Episode_Termination/time_out','Metrics/base_velocity/error_vel_xy','Metrics/base_velocity/error_vel_yaw',
                'Episode_Reward/stair_clearance','Episode_Reward/stair_progress','Episode_Reward/stair_landing',
                'Loss/capability_preservation'):
        values=event.Scalars(tag)
        assert len(values)==100 and [v.step for v in values]==list(range(10000,10100))
        a=np.array([v.value for v in values]);assert np.isfinite(a).all()
        if tag=='Loss/capability_preservation':assert not a.any()
        metrics[tag]=dict(first20=float(a[:20].mean()),last20=float(a[-20:].mean()),last=float(a[-1]))
    coverage=np.array(report['coverage_by_column_level']);clearance=np.array(report['clearance_by_column_level'])
    assert coverage.shape==clearance.shape==(20,10) and coverage.sum()==9830400 and np.all(clearance<=coverage)
    assert report['standing_fixed_left_fixed_right_frames'][1]>0 and report['standing_fixed_left_fixed_right_frames'][2]>0
    return dict(updates=100,interactions=9830400,adam_steps=202000,auxiliary_coefficient=0.,
        metrics=metrics,report=report,execution=read(run/'execution.json'),integration=read(directory(0)/'report.json'))


def heldout_changes(checkpoint):
    import torch
    from reference_low_stairs_preservation import GROUPS, ACTION_SCALES
    path=ROOT/'docs/lite3-low-stairs-regression-preparation-20261115/diagnostic_anchor_bank.npz'
    assert sha(path)==read(path.with_name('preservation_validation.json'))['anchor_sha256']
    MODELS['zero_preservation_action_check']=checkpoint
    actor=load_actor('zero_preservation_action_check')
    with np.load(path) as archive,torch.no_grad():
        obs=torch.tensor(archive['observation']);target=torch.tensor(archive['parent_action'])
        delta=(actor(obs)-target)*torch.tensor(ACTION_SCALES);groups=archive['group']
        rows=[dict(group=name,samples=int((groups==g).sum()),
            target_rmse_rad=float(delta[groups==g].square().mean().sqrt()),
            absolute_p95_rad=float(delta[groups==g].abs().flatten().quantile(.95))) for g,name in enumerate(GROUPS)]
    return dict(bank_sha256=sha(path),used_for_training=False,groups=rows,
        note='开发诊断样本已被重复观察；不能视为未见过的泛化测试。')


def main():
    checkpoint,digest=checked_training()
    complete=read(EVALUATION/'complete.json')
    assert complete['status']=='passed' and complete['checkpoint_sha256']==digest
    assert complete['formal_episodes']==1600 and complete['heading_episodes']==240
    roots=dict(parent10000=PARENT_RUNS,clearance10100=CLEARANCE,guarded10100=GUARDED_RUNS,zero10100=EVALUATION)
    hashes=dict(parent10000=EXPECTED_SHA,clearance10100=CLEARANCE_SHA,guarded10100=GUARDED_SHA,zero10100=digest)
    labels=dict(parent10000='lowstairs10000',clearance10100='lowstairs_clearance10100',guarded10100='lowstairs_guarded10100',zero10100=MODEL)
    tables={k:[] for k in ('turn_groups','turn_windows','turn_pairs','turn_common_four','turn_failures',
        'stairs_heights','stairs_groups','stairs_pairs','diagnostics','matched_contacts')}
    checks,runs,gates=[],[],[]
    for profile in ('nominal','training_randomized'):
        windows,reports={},{}
        for policy in POLICIES:
            path=roots[policy]/f'turn_{profile}'
            verify_run(path,'turn',profile,hashes[policy],labels[policy])
            report,data,rows,groups=analyze_turn(path)
            validate_trace(report,data)
            assert same_initial(path,PARENT_RUNS/f'turn_{profile}')
            assert same_configuration(read(path/'config.json'),read(PARENT_RUNS/f'turn_{profile}/config.json'))
            tables['turn_windows'].extend(dict(policy=policy,**r) for r in rows)
            tables['turn_groups'].extend(dict(policy=policy,**r) for r in groups)
            windows[policy],reports[policy]=rows,report
            checks.append(check_log(path));runs.append((policy,path));del data
        tables['turn_common_four'].extend(common_windows(windows,profile))
        for old in POLICIES[:-1]:
            paired_records(reports[old],reports['zero10100'],'turn')
            pairs=paired_turn_rows(windows[old],windows['zero10100'],profile)
            tables['turn_pairs'].extend(dict(baseline_policy=old,candidate_policy='zero10100',**r) for r in pairs)
            if old=='parent10000':gates.extend(turn_stop_checks(reports[old],reports['zero10100'],pairs,profile))
        parent={r['env_id']:r for r in reports['parent10000']['records']}
        guarded={r['env_id']:r for r in reports['guarded10100']['records']}
        for policy,report in reports.items():
            for case in CASES:
                records=[r for r in report['records'] if r['case']==case]
                row=dict(policy=policy,profile=profile,case=case,starts=len(records),
                    initial_failures=sum(r['outcome']=='orientation_failure' and r['duration_s']<=2 for r in records),
                    after_command_failures=sum(r['outcome']=='orientation_failure' and r['duration_s']>2 for r in records))
                for name,old in (('parent',parent),('guarded',guarded)):
                    row[f'new_vs_{name}']=sum(r['outcome']=='orientation_failure' and old[r['env_id']]['outcome']!='orientation_failure' for r in records)
                    row[f'recovered_vs_{name}']=sum(r['outcome']!='orientation_failure' and old[r['env_id']]['outcome']=='orientation_failure' for r in records)
                tables['turn_failures'].append(row)
    jobs=[(p,d,False) for p in ('nominal','training_randomized') for d in ('up','down')]+[('nominal','up',True)]
    for profile,direction,heading in jobs:
        series,reports={},{}
        for policy in POLICIES:
            if direction=='down' and policy=='clearance10100':continue
            historical=heading and policy=='parent10000'
            path=DIAGNOSIS/'stairs_parent_heading' if historical else roots[policy]/('heading_up_nominal' if heading else f'{direction}_{profile}')
            if historical:report,cfg,data=historical_heading(path,hashes[policy])
            else:
                verify_run(path,'stairs',profile,hashes[policy],labels[policy],direction,heading)
                report,cfg,data,*_=analyze_stairs(path)
            validate_trace(report,data)
            assert same_initial(path,PARENT_RUNS/f'{direction}_{profile}')
            assert same_configuration(cfg,read(PARENT_RUNS/f'{direction}_{profile}/config.json'))
            check=check_log(path)
            if heading:check['feedback_max_error']=heading_error(path,data)
            if historical:check['source_note']='沿用已提交旧报告与完整轨迹；旧探针源码未单独留档，详见相同更新数报告。'
            checks.append(check);runs.append((policy,path));reports[policy]=report
            tables['stairs_heights'].extend(dict(policy=policy,heading=heading,**r) for r in stairs_heights(report))
            tables['stairs_groups'].extend(dict(policy=policy,profile=profile,direction=direction,heading=heading,
                height=g['height'],speed=g['speed'],starts=g['episodes'],**{k:g['outcomes'].get(k,0) for k in OUTCOMES}) for g in report['groups'])
            if direction=='up':
                rows,series[policy]=stair_diagnostics(report,cfg,data,policy,heading)
                tables['diagnostics'].extend(rows)
            del data
        if direction=='up':tables['matched_contacts'].extend(matched_contacts(series,profile,heading))
        for old in POLICIES[:-1]:
            if old not in reports:continue
            a,b=paired_records(reports[old],reports['zero10100'],'stairs')
            for height in (0.,.02,.03,.05):
                for speed in (.32,.4,.6):
                    ids=[i for i,r in a.items() if (r['height'],r['speed'])==(height,speed)]
                    tables['stairs_pairs'].append(dict(profile=profile,direction=direction,heading=heading,
                        baseline_policy=old,candidate_policy='zero10100',height=height,speed=speed,starts=len(ids),
                        gained_success=sum(a[i]['outcome']!='success' and b[i]['outcome']=='success' for i in ids),
                        lost_success=sum(a[i]['outcome']=='success' and b[i]['outcome']!='success' for i in ids),
                        new_orientation_failure=sum(a[i]['outcome']!='orientation_failure' and b[i]['outcome']=='orientation_failure' for i in ids)))
    tables['diagnostic_groups']=group_diagnostics(tables['diagnostics'])
    def screening_rows(name):
        return [dict(r,policy='baseline' if r['policy']=='parent10000' else 'candidate')
                for r in tables[name] if r['policy'] in ('parent10000','zero10100') and not r['heading']]
    screening=screen(screening_rows('stairs_heights'),screening_rows('stairs_groups'),
        [r for r in tables['turn_pairs'] if r['baseline_policy']=='parent10000'],
        [dict(r,new_failures=r['new_vs_parent']) for r in tables['turn_failures'] if r['policy']=='zero10100'])
    for name,rows in tables.items():write_csv(DOCS/f'{name}.csv',rows)
    historical={}
    for folder in ('results-20261114','guarded-trial-20261115','equal100-20261115'):
        for r in read(ROOT/f'docs/lite3-low-stairs-{folder}/raw_manifest.json'):
            if r['path'] in historical:assert historical[r['path']]==r['sha256']
            historical[r['path']]=r['sha256']
    raw={};reused=0
    for policy,path in runs:
        for file in path.iterdir():
            if not file.is_file():continue
            digest_file=sha(file)
            if policy!='zero10100':assert historical[str(file)]==digest_file;reused+=1
            raw[str(file)]=dict(path=str(file),bytes=file.stat().st_size,sha256=digest_file)
        assert all(sha(r['path'])==r['sha256'] for r in read(path/'config.json')['sources'])
    # 新结果的小型 JSON 留档；历史结果已有提交，不重复复制。
    for file in OUTPUT.rglob('*'):
        if not file.is_file():continue
        raw[str(file)]=dict(path=str(file),bytes=file.stat().st_size,sha256=sha(file))
        if file.suffix=='.json':
            target=DOCS/'runs'/file.relative_to(OUTPUT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,target)
    raw[str(BANK)]=dict(path=str(BANK),bytes=BANK.stat().st_size,sha256=sha(BANK))
    train=training_metrics();adapter=adapter_check(checkpoint,EVALUATION);heldout=heldout_changes(checkpoint)
    suites=[]
    for name in ('tests.xml','evaluation_tests.xml','analysis_tests.xml'):
        suites+=ET.parse(DOCS/name).getroot().findall('testsuite')
    assert all(int(s.get(k))==0 for s in suites for k in ('errors','failures','skipped'))
    for phase in (0,1):checks.append(check_log(directory(phase)))
    assert sha(checkpoint)==digest
    save(DOCS/'summary.json',dict(status='passed',status_scope='证据核验通过，能力结论见 screening',
        updates=100,checkpoint=str(checkpoint),checkpoint_sha256=digest,formal_episodes=1600,heading_episodes=240,
        screening=screening,original_turn_gate=dict(passed=all(c['passed'] for c in gates),checks=gates),heldout=heldout,
        **{k:v for k,v in tables.items() if k not in ('turn_windows','diagnostics')}))
    save(DOCS/'training_summary.json',train)
    save(DOCS/'verification.json',dict(status='passed',adapter=adapter,runs=checks,
        tests_passed=sum(int(s.get('tests')) for s in suites),historical_files_verified=reused,
        implementation_sources=sources()+[dict(path=str(Path(__file__)),sha256=sha(__file__))],
        evaluation_elapsed_s=sum(read(p/'execution.json')['elapsed_s'] for policy,p in runs if policy=='zero10100'),
        automatic_training_extension=False))
    save(DOCS/'raw_manifest.json',list(raw.values()))
    print('ZERO PRESERVATION ANALYSIS COMPLETE:',screening['status'],flush=True)


if __name__=='__main__':main()
