"""Complete the approved frozen matrix; behavioral regressions are reported, not gates."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import traceback

from reference_stair_allocation_evaluate import (
    EVALUATION as PREVIOUS, jobs, baseline_path, checked_training, read, record_regressions,
)
from reference_stair_allocation import PARENT, DIGEST
from reference_stairs_eval import ROOT, save, sha
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_low_stairs_zero_preservation_evaluate import verify_run

OUTPUT = ROOT/'logs/reference_six/stair_allocation_completion_20261116'
DOCS = ROOT/'docs/lite3-stair-allocation-completion-20261116'
LIMIT = 5600


def schedule(previous):
    assert previous['status'] == 'stopped_regression'
    done = {(r['variant'],r['name']) for r in previous['matrix']}
    assert done == {('A','heading_up_nominal'),('B','heading_up_nominal')}
    assert sum(r['episodes'] for r in previous['matrix']) == 480
    result = []
    for name,kind,profile,direction,heading in jobs():
        for variant in ('A','B'):
            if (variant,name) not in done:
                result.append(dict(variant=variant,name=name,kind=kind,profile=profile,
                    direction=direction,heading=heading,episodes=320 if kind=='turn' else 240))
    assert len(result)==18 and sum(r['episodes'] for r in result)==4640
    assert sum(r['episodes'] for r in result)+6*160==LIMIT
    return result


def execute_once(script, flags, destination):
    """A failed/incomplete previous launch is never silently retried."""
    if destination.exists():
        execution = destination/'execution.json'
        if not execution.is_file() or read(execution)['status']!='completed':
            raise RuntimeError(f'incomplete previous launch; no automatic retry: {destination}')
        return
    subprocess.run([sys.executable,str(ROOT/'scripts/tools'/script),*flags,
        '--output',str(destination)],cwd=ROOT,check=True)


def verify_core(entry, path, digest, model, baseline):
    result = verify_run(path,entry['kind'],entry['profile'],digest,model,entry['direction'],heading=False)
    assert same_initial(path,baseline)
    assert same_configuration(read(path/'config.json'),read(baseline/'config.json'))
    if entry['heading']:
        meta = read(path/'heading_intervention.json')
        assert meta['source_sha256']==sha(ROOT/'scripts/tools/reference_stair_allocation_heading.py')
        assert meta['checkpoint_sha256']==digest and meta['optimizer_updates']==0
        assert meta['gain']==.5 and meta['limit_rad_s']==.8 and not meta['lateral_position_feedback']
    return result


def brief(report):
    return dict(episodes=report['episodes'],outcomes=dict(Counter(r['outcome'] for r in report['records'])))


def verify_retention(path, model, profile, digest, baseline):
    from reference_stair_allocation_retention import protocol
    r,e,c=(read(path/f'{name}.json') for name in ('report','execution','config'))
    assert e['status']=='completed' and e['exit_code']==0 and e['physical_gpu']==1
    assert r['status']=='passed' and r['model']==model and r['profile']==profile
    assert r['model_sha256']==c['model_sha256']==digest
    assert r['episodes']==len(r['records'])==160 and r['optimizer_updates']==0
    assert r['actor_unchanged'] and r['first_episodes_only'] and r['pre_reset_capture']
    assert c['specs']==protocol() and c['source_sha256']==sha(ROOT/'scripts/tools/reference_stair_allocation_retention.py')
    assert same_initial(path,baseline) and same_configuration(c,read(baseline/'config.json'))
    return r


def preflight():
    previous = read(PREVIOUS/'complete.json'); remaining = schedule(previous)
    checkpoints = {v:checked_training(v) for v in ('A','B')}
    assert sha(PARENT)==DIGEST
    for name,kind,profile,direction,heading in jobs():
        path=baseline_path(name); r=read(path/'report.json'); e=read(path/'execution.json')
        assert e['status']=='completed' and e['exit_code']==0 and r['actor_unchanged']
        assert r.get('model_sha256',r.get('checkpoint_sha256'))==DIGEST
        assert r['episodes']==(320 if kind=='turn' else 240)
    for row in previous['matrix']:
        v=row['variant']; path=Path(row['path'])
        verify_core(row,path,checkpoints[v][1],f'allocation_{v.lower()}10200',Path(row['baseline']))
    sources = [dict(path=str(ROOT/'scripts/tools'/name),sha256=sha(ROOT/'scripts/tools'/name)) for name in (
        'reference_stair_allocation_complete.py','reference_stair_allocation_evaluate.py',
        'reference_stair_allocation_heading.py','reference_stair_allocation_retention.py',
        'reference_turn_diagnose.py','reference_isaac_stairs_diagnose.py','source_block_runtime.py')]
    plan=dict(status='prepared',training=False,new_optimizer_updates=0,physical_gpu=1,
        new_episode_limit=LIMIT,previous_episodes_reused=480,combined_candidate_and_retention_episodes=6080,
        behavior_early_stop=False,automatic_promotion=False,automatic_retry=False,
        stop_on='runtime/numerical/provenance/pairing/budget error only',
        predecessor=dict(path=str(PREVIOUS/'complete.json'),sha256=sha(PREVIOUS/'complete.json')),
        checkpoints={v:dict(path=str(p),sha256=h) for v,(p,h) in checkpoints.items()},
        remaining_core_jobs=remaining,retention_jobs=6,retention_episodes=960,sources=sources)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'plan.json').exists(): assert read(OUTPUT/'plan.json')==plan
    else: save(OUTPUT/'plan.json',plan)
    return previous,checkpoints,plan


def run():
    if (OUTPUT/'complete.json').exists(): raise RuntimeError('completion phase already ended; no automatic extension')
    previous,checkpoints,plan=preflight()
    matrix=[dict(r,reused_previous_phase=True) for r in previous['matrix']]
    retention=[]; new_count=0
    try:
        for entry in plan['remaining_core_jobs']:
            v=entry['variant']; checkpoint,digest=checkpoints[v]; model=f'allocation_{v.lower()}10200'
            dest=OUTPUT/v/entry['name']; baseline=baseline_path(entry['name'])
            flags=['--model',model,'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--profile',entry['profile']]
            if entry['heading']:
                script='reference_stair_allocation_heading.py'; flags+=['--direction',entry['direction']]
            else:
                script='reference_turn_diagnose.py' if entry['kind']=='turn' else 'reference_isaac_stairs_diagnose.py'
                flags+=['--repeats','20','--gpu','1','--minimum-free-mib','10240']
                if entry['kind']=='stairs':flags+=['--direction',entry['direction'],'--height-set','low_stairs','--action-mode','mean']
            print('EVALUATE',v,entry['name'],flush=True)
            execute_once(script,flags,dest)
            r=verify_core(entry,dest,digest,model,baseline)
            matrix.append(dict(entry,path=str(dest),baseline=str(baseline),reused_previous_phase=False,
                initial_exact=True,configuration_exact=True))
            new_count+=r['episodes']; assert new_count<=LIMIT
            save(OUTPUT/'matrix.json',matrix)
            save(OUTPUT/'progress.json',dict(new_episodes=new_count,limit=LIMIT,last=dict(entry,**brief(r))))
            print('RESULT',v,entry['name'],brief(r),flush=True)
        for profile in ('nominal','training_randomized'):
            baseline=OUTPUT/'retention/zero10100'/profile
            for model,(checkpoint,digest) in [('zero10100',(PARENT,DIGEST)),*checkpoints.items()]:
                dest=OUTPUT/'retention'/model/profile; label=f'allocation_retention_{model.lower()}'
                flags=['--model',label,'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--profile',profile]
                print('RETENTION',model,profile,flush=True)
                execute_once('reference_stair_allocation_retention.py',flags,dest)
                r=verify_retention(dest,label,profile,digest,baseline)
                retention.append(dict(model=model,profile=profile,path=str(dest),episodes=160))
                new_count+=160; assert new_count<=LIMIT
                save(OUTPUT/'retention_matrix.json',retention)
                save(OUTPUT/'progress.json',dict(new_episodes=new_count,limit=LIMIT,last=dict(model=model,profile=profile,**brief(r))))
                print('RESULT retention',model,profile,brief(r),flush=True)
        assert new_count==LIMIT and len(matrix)==20 and len(retention)==6
        assert all(sha(p)==h for p,h in checkpoints.values()) and sha(PARENT)==DIGEST
        assert all(sha(r['path'])==r['sha256'] for r in plan['sources'])
        assert sha(PREVIOUS/'complete.json')==plan['predecessor']['sha256']
        save(OUTPUT/'complete.json',dict(status='completed',status_scope='execution and identity checks, not behavior acceptance',
            new_evaluation_episodes=new_count,new_optimizer_updates=0,previous_episodes_reused=480,
            core_episodes=5120,retention_episodes=960,total_candidate_and_retention_episodes=6080,
            matrix=matrix,retention=dict(matrix=retention),promoted=False,not_run=[]))
        print('COMPLETED: 5600 new frozen episodes; no training or automatic promotion.',flush=True)
    except Exception:
        save(OUTPUT/'complete.json',dict(status='failed',new_evaluation_episodes=new_count,
            matrix=matrix,retention=dict(matrix=retention),promoted=False,error=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare-only',action='store_true')
    if parser.parse_args().prepare_only:
        _,_,plan=preflight(); print('PREPARED',plan['new_episode_limit'],'new episodes; zero training updates.')
    else:run()
