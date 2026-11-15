"""归档短试验和原协议结果；把航向反馈诊断单独列出。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT, MODELS, load_actor, save, sha
from reference_low_stairs_guarded_bank import OUTPUT, BANK
from reference_low_stairs_guarded_train import directory
from reference_low_stairs_guarded_evaluate import checked_training, read
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_core500_analyze import write_csv, paired_turn_rows
from reference_turn_analyze import analyze as analyze_turn
from reference_turn_diagnose import CASES
from reference_isaac_stairs_analyze import analyze as analyze_stairs
from reference_low_stairs_analyze import stairs_heights, check_log, read_agent
from reference_low_stairs_clearance_analyze import screen

OUT = ROOT / 'docs/lite3-low-stairs-guarded-trial-20261115'
OUTCOMES = ('success','stuck','orientation_failure','lateral_exit','timeout')


def adapter_check(checkpoint, evaluation):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    torch.set_num_threads(1)
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    with np.load(evaluation/'turn_nominal/initial.npz') as values:
        inputs = torch.tensor(values['observation'])
    obs = TensorDict({'policy':inputs},batch_size=[len(inputs)])
    model = MLPModel(obs,{'actor':['policy']},'actor',12,**read_agent()).eval()
    model.load_state_dict(state['actor_state_dict'],strict=True)
    MODELS['guarded_adapter_check'] = checkpoint
    actor = load_actor('guarded_adapter_check')
    with torch.no_grad():
        error = float((model(obs)-actor(inputs)).abs().max())
    assert error == 0.
    return dict(max_output_error=error,samples=len(inputs),status='passed')


def training_summary(updates):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    reports, scalars = [], {}
    for phase in range(1,updates//100+1):
        run = directory(phase); report = read(run/'report.json')
        event = EventAccumulator(str(run/'training'),size_guidance={'scalars':0}); event.Reload()
        for tag in ('Train/mean_reward','Train/mean_episode_length','Episode_Termination/bad_orientation_2',
                    'Episode_Termination/time_out','Metrics/base_velocity/error_vel_xy','Metrics/base_velocity/error_vel_yaw',
                    'Episode_Reward/stair_clearance','Episode_Reward/stair_progress','Episode_Reward/stair_landing',
                    'Loss/capability_preservation'):
            values = event.Scalars(tag)
            assert len(values)==100 and [v.step for v in values]==list(range(10000+(phase-1)*100,10000+phase*100))
            scalars.setdefault(tag,[]).extend(v.value for v in values)
        coverage = np.asarray(report['coverage_by_column_level'])
        clearance = np.asarray(report['clearance_by_column_level'])
        assert coverage.sum()==report['interactions'] and np.all(clearance<=coverage)
        assert report['standing_fixed_left_fixed_right_frames'][1]>0 and report['standing_fixed_left_fixed_right_frames'][2]>0
        reports.append(dict(report=report,execution=read(run/'execution.json')))
    metrics={}
    for tag,values in scalars.items():
        a=np.asarray(values); assert np.isfinite(a).all()
        metrics[tag]=dict(first20=float(a[:20].mean()),last20=float(a[-20:].mean()),last=float(a[-1]))
    return dict(updates=updates,interactions=updates*24*4096,adam_steps=200000+updates*20,
                metrics=metrics,segments=reports)


def main(updates):
    checkpoint,digest=checked_training(updates)
    evaluation=OUTPUT/f'evaluation_{10000+updates}'
    complete=read(evaluation/'complete.json')
    assert complete['formal_episodes']==1600 and complete['checkpoint_sha256']==digest
    tables={k:[] for k in ('stairs_heights','stairs_groups','turn_groups','turn_windows','turn_paired','turn_failures')}
    logs=[]
    for entry in complete['matrix']:
        old,new=Path(entry['baseline']),Path(entry['candidate'])
        assert same_initial(old,new) and same_configuration(read(old/'config.json'),read(new/'config.json'))
        reports,windows={},{}
        for policy,path in (('baseline',old),('candidate',new)):
            logs.append(check_log(path))
            if entry['kind']=='turn':
                report,data,rows,groups=analyze_turn(path)
                tables['turn_groups'].extend(dict(policy=policy,**r) for r in groups)
                tables['turn_windows'].extend(dict(policy=policy,**r) for r in rows)
                windows[policy]=rows
            else:
                report,_,data,_,_,_=analyze_stairs(path)
                tables['stairs_heights'].extend(dict(policy=policy,**r) for r in stairs_heights(report))
                tables['stairs_groups'].extend(dict(policy=policy,profile=entry['profile'],direction=entry['direction'],
                    **{k:v for k,v in row.items() if k!='outcomes'},
                    **{k:row['outcomes'].get(k,0) for k in OUTCOMES}) for row in report['groups'])
            assert all(np.isfinite(v).all() for v in data.values())
            reports[policy]=report
        if entry['kind']=='turn':
            tables['turn_paired'].extend(paired_turn_rows(windows['baseline'],windows['candidate'],entry['profile']))
            a,b=({r['env_id']:r for r in reports[p]['records']} for p in ('baseline','candidate'))
            for case in CASES:
                ids=[i for i,r in a.items() if r['case']==case]
                row=dict(profile=entry['profile'],case=case,starts=len(ids))
                for policy,data in (('baseline',a),('candidate',b)):
                    for suffix,predicate in (
                        ('failures',lambda r:True),
                        ('failures_before_command',lambda r:r['duration_s']<=2.),
                        ('failures_after_command',lambda r:r['duration_s']>2.)):
                        row[f'{policy}_{suffix}']=sum(data[i]['outcome']=='orientation_failure' and predicate(data[i]) for i in ids)
                row['new_failures']=sum(a[i]['outcome']!='orientation_failure' and b[i]['outcome']=='orientation_failure' for i in ids)
                row['recovered']=sum(a[i]['outcome']=='orientation_failure' and b[i]['outcome']!='orientation_failure' for i in ids)
                tables['turn_failures'].append(row)
    for name,rows in tables.items(): write_csv(OUT/f'{name}.csv',rows)
    heading_path=Path(complete['heading_diagnostic'])
    heading_report,_,heading_data,_,_,_=analyze_stairs(heading_path)
    assert all(np.isfinite(v).all() for v in heading_data.values())
    logs.append(check_log(heading_path))
    heading=dict(heights=stairs_heights(heading_report),groups=heading_report['groups'],
                 intervention=read(heading_path/'heading_intervention.json'),episodes=240,
                 initial_equal=same_initial(heading_path,evaluation/'up_nominal'),included_in_formal_acceptance=False)
    assert heading['initial_equal']
    screening=screen(tables['stairs_heights'],tables['stairs_groups'],tables['turn_paired'],tables['turn_failures'])
    train=training_summary(updates)
    adapter=adapter_check(checkpoint,evaluation)
    suites=ET.parse(OUT/'tests.xml').getroot().findall('testsuite')
    assert all(int(s.get(k))==0 for s in suites for k in ('errors','failures','skipped'))
    save(OUT/'summary.json',dict(status='passed',scope='证据核验通过，能力验收见 screening',updates=updates,
        checkpoint=str(checkpoint),checkpoint_sha256=digest,formal_episodes=1600,heading_episodes=240,
        screening=screening,heading=heading,gate100=read(OUTPUT/'gate100.json'),**tables))
    save(OUT/'training_summary.json',train)
    save(OUT/'verification.json',dict(status='passed',adapter=adapter,logs=logs,
        tests_passed=sum(int(s.get('tests')) for s in suites),parameters_updated_by_analysis=False,
        implementation_sources=[dict(path=str(p),sha256=sha(p)) for p in sorted((ROOT/'scripts/tools').glob('reference_low_stairs_guarded*.py'))]))
    # 小型 JSON 证据进入版本库；原始轨迹、检查点和事件日志按哈希保留在本地。
    for path in sorted(OUTPUT.rglob('*.json')):
        target=OUT/'runs'/path.relative_to(OUTPUT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    shutil.copy2(BANK,OUT/'training_bank.npz')
    raw=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUTPUT.rglob('*')) if p.is_file()]
    save(OUT/'raw_manifest.json',raw)
    print('GUARDED ANALYSIS COMPLETE:',screening['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,choices=(100,200),required=True)
    main(p.parse_args().updates)
