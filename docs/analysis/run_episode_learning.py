"""Learn bounded episode codes from new charged rollouts; freeze the final mean."""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path

from docs.analysis.audit_sac_huber_coverage import read
from docs.analysis.run_c0_headroom import run_stage
from rl_training.online_adaptation import managed_trial
from rl_training.online_adaptation.episode_search import EpisodeSearch
from rl_training.online_adaptation.interaction_budget import InteractionBudget, digest
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def runtime_hash():
    package=Path(managed_trial.__file__).resolve().parent;root=package.parents[3]
    paths=sorted(package.glob('*.py'))+sorted((root/'scripts/reinforcement_learning/online_adaptation').glob('*.py'))
    paths += [package.parent/name for name in ('lite3_joint_conventions.py','lite3_stability_contract.py')]
    return digest({str(p.relative_to(root)):sha256_file(p) for p in paths})


def write_locked(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError(f'locked data changed: {path}')
    else:write_json_atomic(path,value)


def session_state(directory):
    if (directory/'session_summary.json').exists():return read(directory/'session_summary.json')
    if (directory/'ledger/manifest.json').exists():
        with InteractionBudget(directory/'ledger') as budget:return copy.deepcopy(budget.state)
    return None


def costs(group):
    result={'training':0,'evaluation':0,'retention':0,'uncertain_steps':0}
    for directory in group.iterdir():
        if not directory.is_dir():continue
        state=session_state(directory)
        if state is None:continue
        if directory.name.startswith('g'):
            if any(v for k,v in state['spent'].items() if k!='evaluation'):raise ValueError('unexpected training child phase')
            result['training']+=sum(state['spent'].values())
        elif directory.name in ('candidate','base'):
            if any(v for k,v in state['spent'].items() if k not in ('evaluation','retention')):raise ValueError('unexpected frozen phase')
            for key in ('evaluation','retention'):result[key]+=state['spent'][key]
        else:raise ValueError('unexpected child session')
        result['uncertain_steps']+=state['uncertain_steps']
    return result


def outcome(directory,attempt_id):
    state=session_state(directory)
    if state is None:raise ValueError('missing physical accounting')
    result={'id':attempt_id,'terminal_reason':'interrupted','normalized_progress':0.,'safety_interventions':0,
            'charged_steps':sum(state['spent'].values()),'uncertain_steps':state['uncertain_steps']}
    trials=list(state['trials'].values())
    if len(trials)!=1:raise ValueError('one physical attempt required per search code')
    result['reset_seed']=trials[0]['reset_seed']
    if trials[0]['status']=='complete':
        path=Path(trials[0]['output']);s=read(path/'episode_summary.json')
        progress=(s['max_x']-s['initial_x'])/8.
        if not math.isfinite(progress):raise ValueError('nonfinite physical progress')
        records=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        result.update(terminal_reason=s['task_stop_reason'],normalized_progress=max(0.,min(1.,progress)),
            safety_interventions=sum(bool(r['safety_reasons']) or bool(r['paused']) for r in records),
            summary_sha256=sha256_file(path/'episode_summary.json'),metrics_sha256=sha256_file(path/'metrics.jsonl'))
    return result


def command(p,group,terrain,stage,worker_seed,code,*,phase='evaluation',training=False):
    import sys
    options=dict(p['worker_options'])
    if code is not None:options['fixed-code']=code
    options_path=group/f'{stage}_options.json';write_locked(options_path,options)
    trials=1 if training else p['evaluation_trials']
    return [sys.executable,'-u','scripts/reinforcement_learning/online_adaptation/run_adaptation_session.py',
        '--session-dir',str(group/stage),'--algorithm','sac','--base-checkpoint',p['base_checkpoint'],
        '--terrain-seed',str(terrain),'--seed',str(worker_seed),'--budget-steps',str(p['trial_steps']),
        '--target-warmup-steps','0','--trial-steps',str(p['trial_steps']),'--selection-budget','0','--probe-milestones',
        '--evaluation-steps',str(trials*p['trial_steps']),'--evaluation-trials',str(trials),
        '--retention-trials',str(1 if training else p['retention_trials']),
        '--retention-trial-steps',str(1 if training else p['retention_steps']),
        '--worker-options',str(options_path),'--device',p['_device'],'--phase',phase]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--terrain',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--device',required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    p=read(args.protocol);assert p['schema']=='lite3_c0_episode_learning_v1'
    assert args.terrain in p['terrains'] and args.seed in p['learner_seeds']
    assert p['population']==4 and p['code_dim']==2 and p['initial_mean']==[0.,0.]
    assert p['initial_std']==[.3,.3] and p['std_floor']==.1
    assert p['generations']*p['population']*p['trial_steps']<=p['training_max_steps_per_session']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    assert sha256_file(p['prior_confirmation_report'])==p['prior_confirmation_sha256']
    assert any(t['confirmation_pass'] for t in read(p['prior_confirmation_report'])['terrains'].values())
    root=Path(p['output'])
    if args.seed!=p['learner_seeds'][0]:
        gate=read(root/f'audit-seed{p["learner_seeds"][0]}/report.json')
        if gate['status']!='complete' or not gate['replication_gate']:raise ValueError('second learning seed not admitted')
    group=root/f'seed-{args.seed}-terrain-{args.terrain}';group.mkdir(exist_ok=True)
    p['_device']=args.device
    with (group/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=group/'execution.json'
        if path.exists():
            if not args.resume:raise ValueError('inspect process before explicit resume')
            state=read(path);assert state['protocol_sha256']==sha256_file(args.protocol)
            search=EpisodeSearch.from_state_dict(state['search_state'])
        else:
            search=EpisodeSearch(args.seed,p['code_dim'])
            state={'status':'running','protocol_sha256':sha256_file(args.protocol),'source_sha256':runtime_hash(),
                   'seed':args.seed,'terrain':args.terrain,'stages':{},'search_state':search.state_dict()}
        def execute(stage,cmd):
            if runtime_hash()!=state['source_sha256']:raise ValueError('runtime source changed')
            ok=run_stage(path,state,stage,cmd,group)
            state['costs']=costs(group)
            total=sum(state['costs'][k] for k in ('training','evaluation','retention'))
            assert state['costs']['training']<=p['training_max_steps_per_session'] and total<=p['max_steps_per_terrain_seed']
            if not ok:
                state['status']='failed';write_json_atomic(path,state);raise SystemExit(1)
            write_json_atomic(path,state)
        while search.generation<p['generations']:
            generation=search.generation;pending=search.ask()
            state['search_state']=search.state_dict();write_json_atomic(path,state)
            results=[]
            worker_seed=args.seed+100000*(generation+1)
            for i,attempt in enumerate(pending):
                stage=f'g{generation:02d}-c{i:02d}'
                execute(stage,command(p,group,args.terrain,stage,worker_seed,attempt['code'],training=True))
                results.append(outcome(group/stage,attempt['id']))
            assert len({r['reset_seed'] for r in results})==1
            search.tell(results);state['search_state']=search.state_dict();write_json_atomic(path,state)
        final={'protocol_sha256':state['protocol_sha256'],'source_sha256':state['source_sha256'],
               'search_state_sha256':digest(search.state_dict()),'code':search.mean.tolist(),'generation':search.generation}
        write_locked(group/'final_code.json',final)
        state['final_code_sha256']=sha256_file(group/'final_code.json');write_json_atomic(path,state)
        roles=('candidate','base') if args.terrain==101 else ('base','candidate')
        for role in roles:
            for phase in ('evaluation','retention'):
                execute(f'{role}:{phase}',command(p,group,args.terrain,role,args.seed,final['code'] if role=='candidate' else None,phase=phase))
        state['status']='complete';write_json_atomic(path,state)


if __name__=='__main__':main()
