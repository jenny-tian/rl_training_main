"""Bounded grid selection with locked codes before independent-start verification."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

from docs.analysis.audit_sac_huber_coverage import read
from rl_training.online_adaptation.episode_search import EpisodeSearch
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def select_grid(records):
    eligible=[r for r in records if r['eligible']]
    eligible.sort(key=lambda r:(-int(r['completion']),-r['normalized_progress'],r['safety_interventions'],r['index']))
    return [r['index'] for r in eligible[:2]] if len(eligible)>=2 else []


def command(p, terrain, device, stage, code_index, seed, trials):
    root=Path(p['output'])
    options=root/('base_options.json' if code_index is None else f'grid-{code_index:02d}_options.json')
    return [sys.executable,'-u','scripts/reinforcement_learning/online_adaptation/run_adaptation_session.py',
        '--session-dir',str(root/f'{stage}-{terrain}'),'--algorithm','sac','--base-checkpoint',p['base_checkpoint'],
        '--terrain-seed',str(terrain),'--seed',str(seed),'--budget-steps',str(trials*p['trial_steps']),
        '--target-warmup-steps','0','--trial-steps',str(p['trial_steps']),'--selection-budget','0','--probe-milestones',
        '--evaluation-steps',str(trials*p['trial_steps']),'--evaluation-trials',str(trials),
        '--retention-trials','1','--retention-trial-steps','1','--worker-options',str(options),
        '--device',device,'--phase','evaluation']


def run_stage(path, state, stage, cmd, root):
    prior=state['stages'].get(stage)
    if prior:
        if prior['command']!=cmd:raise ValueError('stage command changed')
        if prior['status'] in ('complete','failed'):return prior['status']=='complete'
    state['stages'][stage]={'command':cmd,'status':'running'}
    write_json_atomic(path,state)
    with (root/f'{path.stem}-{stage}.log').open('a') as stream:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    state['stages'][stage].update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
    write_json_atomic(path,state)
    return result.returncode==0


def grid_record(p, terrain, index, stage_ok):
    root=Path(p['output'])/f'grid-{index:02d}-{terrain}'
    record={'index':index,'code':p['codes'][index],'eligible':False,'input_hashes':{}}
    summary=root/'session_summary.json'
    if not summary.exists():return record
    ledger=read(summary);record['input_hashes'][str(summary)]=sha256_file(summary)
    trials=list(ledger['trials'].values())
    if not stage_ok or len(trials)!=1 or trials[0]['status']!='complete':return record
    path=Path(trials[0]['output']);s=read(path/'episode_summary.json')
    records=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
    for file in (path/'episode_summary.json',path/'metrics.jsonl',root/'ledger/manifest.json'):
        record['input_hashes'][str(file)]=sha256_file(file)
    if not all(math.isfinite(s[k]) for k in ('max_x','initial_x')):
        record['invalid_reason']='nonfinite_progress'
        return record
    record.update(eligible=s['task_stop_reason'] in EpisodeSearch.TERMINALS,
        completion=s['task_stop_reason']=='completion',stop_reason=s['task_stop_reason'],
        normalized_progress=max(0.,min(1.,(s['max_x']-s['initial_x'])/p['track_distance_m'])),
        safety_interventions=sum(bool(r['safety_reasons']) or bool(r['paused']) for r in records),
        charged_steps=trials[0]['charged_steps'])
    return record


def check_cost_and_source(p,terrain):
    root=Path(p['output']);cost=0;hashes=set()
    for directory in root.glob(f'*-{terrain}'):
        state_path=directory/'session_summary.json'
        ledger_path=directory/'ledger/manifest.json'
        if ledger_path.exists():hashes.add(read(ledger_path)['worker_configuration']['source_sha256'])
        if not state_path.exists():continue
        s=read(state_path);cost+=sum(s['spent'].values())
        if any(v for k,v in s['spent'].items() if k!='evaluation'):
            raise ValueError('unexpected phase costs')
    if cost>p['per_terrain_max_steps'] or len(hashes)>1:raise ValueError('cost/source contract violated')
    return cost


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--terrain',type=int,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();p=read(args.protocol);root=Path(p['output'])
    assert p['schema']=='lite3_c0_grid_headroom_v1' and args.terrain in p['terrains']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    assert sha256_file(p['prior_engineering_report'])==p['prior_engineering_sha256']
    assert read(p['prior_engineering_report'])['engineering_pass']
    assert read(root/'base_options.json')==p['worker_options']
    for i,code in enumerate(p['codes']):
        assert read(root/f'grid-{i:02d}_options.json')==dict(p['worker_options'],**{'fixed-code':code})
    with (root/f'execution-{args.terrain}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/f'execution-{args.terrain}.json'
        if path.exists():
            if not args.resume:raise ValueError('inspect existing process and use explicit resume')
            state=read(path);assert state['protocol_sha256']==sha256_file(args.protocol)
        else:state={'status':'running','protocol_sha256':sha256_file(args.protocol),'stages':{}}
        for i in range(len(p['codes'])):
            stage=f'grid-{i:02d}'
            run_stage(path,state,stage,command(p,args.terrain,args.device,stage,i,p['grid_seed'],p['grid_trials']),root)
            check_cost_and_source(p,args.terrain)
        records=[grid_record(p,args.terrain,i,state['stages'][f'grid-{i:02d}']['status']=='complete') for i in range(len(p['codes']))]
        selection={'protocol_sha256':state['protocol_sha256'],'grid_records':records,'selected_indices':select_grid(records)}
        selection_path=root/f'selection-{args.terrain}.json'
        if selection_path.exists():
            if read(selection_path)!=selection:raise ValueError('locked selection or inputs changed')
        else:write_json_atomic(selection_path,selection)
        state['selection_sha256']=sha256_file(selection_path);write_json_atomic(path,state)
        selected=selection['selected_indices']
        if len(selected)==2:
            for role in p['verification_order']:
                index=None if role=='base' else selected[int(role[-1])-1]
                stage=f'verify-{role}'
                run_stage(path,state,stage,command(p,args.terrain,args.device,stage,index,p['verification_seed'],p['verification_trials']),root)
                check_cost_and_source(p,args.terrain)
        state['actual_control_transitions']=check_cost_and_source(p,args.terrain)
        state['status']='complete' if len(selected)==2 and all(s['status']=='complete' for s in state['stages'].values()) else 'failed'
        write_json_atomic(path,state)
        if state['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
