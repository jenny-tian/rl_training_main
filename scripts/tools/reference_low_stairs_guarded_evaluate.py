"""中途停止检查及最终原协议矩阵；不启动或延长训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_guarded_bank import OUTPUT
from reference_low_stairs_guarded_train import directory, sources, VARIANT
from reference_low_stairs_guarded_ppo import KEY
from reference_low_stairs_clearance_train import check_algorithm, source_preflight
from reference_low_stairs_evaluate import OUTPUT as BASELINE, EXPECTED_SHA, verify_run as verify_baseline
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_turn_analyze import analyze as analyze_turn
from reference_core500_analyze import paired_turn_rows


def read(path):
    return json.loads(Path(path).read_text())


def checked_training(updates):
    import torch
    source_preflight()
    run = directory(updates // 100)
    report = read(run / 'report.json')
    assert report['status'] == 'passed' and report['phase_updates'] == updates
    assert report['optimizer_updates'] == 100 and report['environment_steps'] == 2400
    assert report['max_updates'] == 200 and report['variant'] == VARIANT
    assert report['original_parent_sha256'] == EXPECTED_SHA
    assert report['implementation_sources'] == sources()
    assert read(run / 'execution.json')['status'] == 'completed'
    checkpoint, digest = Path(report['output_checkpoint']), report['output_sha256']
    assert sha(checkpoint) == digest
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_algorithm(state, updates)
    assert state['iter'] == 9999+updates and state['infos']['phase_updates'] == updates
    assert state['infos'][KEY]['calls'] == updates*20
    assert state['infos'][KEY]['bank_sha256'] == report['bank_sha256']
    return checkpoint, digest


def turn_stop_checks(old_report, new_report, pairs, profile):
    checks = []
    def count(case, before=False):
        def value(report):
            return sum(r['outcome']=='orientation_failure' and (case is None or r['case']==case)
                       and (not before or r['duration_s'] <= 2.) for r in report['records'])
        a,b = value(old_report),value(new_report)
        checks.append(dict(profile=profile, case=case, metric='initial_failures' if before else 'orientation_failures',
                           baseline=a, candidate=b, passed=b<=a))
    count(None, True); count('standing')
    for case in ('low_left','low_right','turn_left','turn_right','fast_left','fast_right'):
        count(case)
        row = next((p for p in pairs if p['case']==case and p['window']=='steady'), None)
        complete = row is not None and row['n'] >= 10
        worsening = complete and row['candidate_mae'] > row['baseline_mae']*1.1 and row['mae_delta'] > .01
        checks.append(dict(profile=profile, case=case, metric='paired_steady_yaw_mae',
            pairs=0 if row is None else row['n'], baseline=None if row is None else row['baseline_mae'],
            candidate=None if row is None else row['candidate_mae'], passed=bool(complete and not worsening)))
    return checks


def evaluate(updates, mode):
    checkpoint, digest = checked_training(updates)
    model = f'lowstairs_guarded{10000+updates}'
    root = OUTPUT / f'evaluation_{10000+updates}'
    root.mkdir(parents=True, exist_ok=True)
    matrix = []
    def run(kind, profile, direction=None):
        name = f'turn_{profile}' if kind == 'turn' else f'{direction}_{profile}'
        dest, baseline = root/name, BASELINE/name
        script = 'reference_turn_diagnose.py' if kind == 'turn' else 'reference_isaac_stairs_diagnose.py'
        flags = ['--model',model,'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,
                 '--profile',profile,'--repeats','20','--gpu','1','--minimum-free-mib','10240','--output',str(dest)]
        if kind == 'stairs': flags += ['--direction',direction,'--height-set','low_stairs','--action-mode','mean']
        if not dest.exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/tools'/script),*flags],cwd=ROOT,check=True)
        report, execution, config = (read(dest/f'{n}.json') for n in ('report','execution','config'))
        assert execution['status']=='completed' and execution['physical_gpu']==1
        assert report['status']=='passed' and report['actor_unchanged'] and not report['probe']
        assert report.get('model_sha256',report.get('checkpoint_sha256'))==digest
        assert report['model']==model and report['profile']==profile
        assert report['episodes']==(320 if kind=='turn' else 240)
        if kind == 'stairs':
            assert report['direction']==direction and report['height_set']=='low_stairs'
        verify_baseline(baseline,kind,profile,direction)
        assert same_initial(dest,baseline),name
        assert same_configuration(config,read(baseline/'config.json')),name
        matrix.append(dict(name=name,kind=kind,profile=profile,direction=direction,candidate=str(dest),
                           baseline=str(baseline),episodes=report['episodes'],initial_equal=True))
        return dest,baseline

    pairs,checks = [],[]
    for profile in ('nominal','training_randomized'):
        new,old = run('turn',profile)
        old_report,_,old_rows,_ = analyze_turn(old)
        new_report,_,new_rows,_ = analyze_turn(new)
        values = paired_turn_rows(old_rows,new_rows,profile)
        pairs.extend(values); checks.extend(turn_stop_checks(old_report,new_report,values,profile))
    if mode=='gate':
        assert updates==100
        result = dict(status='passed' if all(c['passed'] for c in checks) else 'stopped_for_regression',
            phase_updates=100, checkpoint=str(checkpoint),checkpoint_sha256=digest,
            checks=checks,paired_turns=pairs,matrix=matrix,automatic_training_extension=False)
        save(OUTPUT/'gate100.json',result)
        print(json.dumps(dict(status=result['status'],failed=[c for c in checks if not c['passed']]),indent=2))
        return
    for profile in ('nominal','training_randomized'):
        for direction in ('up','down'):
            run('stairs',profile,direction)
    heading = root/'heading_up_nominal'
    if not heading.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/tools/reference_low_stairs_guarded_heading.py'),
            '--model',model,'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--output',str(heading)],check=True,cwd=ROOT)
    assert read(heading/'execution.json')['status']=='completed'
    assert read(heading/'heading_intervention.json')['checkpoint_sha256']==digest
    assert same_initial(heading,BASELINE/'up_nominal')
    assert read(heading/'report.json')['episodes']==240
    assert sum(row['episodes'] for row in matrix)==1600
    save(root/'complete.json',dict(status='passed',status_scope='运行完成，能力验收另行分析',
        updates=updates,checkpoint=str(checkpoint),checkpoint_sha256=digest,matrix=matrix,
        formal_episodes=1600,heading_diagnostic=str(heading),heading_episodes=240,
        paired_turns=pairs,turn_gate_checks=checks,automatic_training_extension=False))
    print('GUARDED EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,choices=(100,200),required=True)
    p.add_argument('--mode',choices=('gate','final'),required=True)
    args=p.parse_args(); evaluate(args.updates,args.mode)
