"""Audit all registered new-terrain pairs without selecting favorable terrain or starts."""
import argparse
from pathlib import Path
from docs.analysis.audit_c0_headroom import audit_child,check_pairs
from docs.analysis.audit_full_policy_evaluation import outcomes
from docs.analysis.audit_learning_replication import retention_pass
from docs.analysis.audit_sac_huber_coverage import read
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def qualify(terrains):
    return bool(len(terrains)==4 and all(
        len(runs)==2 and sum(r['candidate'] for r in runs)>=9
        and sum(r['candidate'] for r in runs)>=sum(r['base'] for r in runs)-1
        and all(r['complete'] and r['safe'] and r['retention'] and r['candidate']>=4 for r in runs)
        for runs in terrains.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    p=read(args.protocol);root=Path(p['output']);lock=read(root/'transfer_lock.json')
    assert lock['protocol_sha256']==sha256_file(args.protocol)
    assert lock['replication_audit_sha256']==sha256_file(p['replication_audit'])
    assert lock['prior_program_audit_sha256']==sha256_file(p['prior_program_audit'])
    assert read(p['replication_audit'])['independent_route_replication_passed']
    assert p['terrain_seeds']==read(p['seed_registration'])['terrain_seeds']
    executions=[read(root/f'execution-{i}.json') for i in (0,1)]
    assert all(e['status']!='running' and e['protocol_sha256']==sha256_file(args.protocol) for e in executions)
    report={'status':'running','protocol_sha256':sha256_file(args.protocol),'terrains':{},'children':{},
            'controls':0,'uncertain_controls':0,'goal_complete':False}
    sources=set();objectives=set()
    for index,terrain in enumerate(p['terrain_seeds']):
        runs=[]
        for run in p['policies']:
            phases={};results={}
            for role in ('candidate','base'):
                name=f"{role}-{run['learning_seed']}-{terrain}";directory=root/name
                checkpoint=run['checkpoint'] if role=='candidate' else {'path':p['base_checkpoint'],'sha256':p['base_sha256']}
                assert sha256_file(checkpoint['path'])==checkpoint['sha256']
                child=audit_child(directory,{'base_sha256':checkpoint['sha256']},terrain,run['evaluation_seed'],None,7,
                                  expected_phases={'evaluation':5,'retention':2})
                manifest=read(directory/'ledger/manifest.json')
                assert Path(manifest['base']['path']).resolve()==Path(checkpoint['path']).resolve()
                arguments=manifest['worker_configuration']['arguments']
                assert arguments['route_observations']==('sim-localization-v1' if role=='candidate' else 'none')
                assert arguments['stochastic_control'] is False
                objectives.add(manifest['task_objective_sha256']);sources.add(child['source_sha256'])
                for trial in read(directory/'session_summary.json')['trials'].values():
                    if trial['status']!='complete':continue
                    summary=read(Path(trial['output'])/'episode_summary.json')
                    assert summary['mode']=='base' and not summary['parameter_changed']
                    assert summary['critic_transitions_committed']==summary['warmup_critic_updates']==0
                for row in child['trials']:
                    if row['status']=='complete':assert row['requested_residual_rms']==row['executed_residual_rms']==0
                report['controls']+=child['spent'];report['uncertain_controls']+=child['uncertain_steps']
                report['children'][name]=child;results[role]=child
                phases[role]={phase:[r for r in child['trials'] if r['phase']==phase] for phase in ('evaluation','retention')}
            for phase in ('evaluation','retention'):check_pairs(phases['candidate'][phase],phases['base'][phase])
            complete=executions[index%2]['status']=='complete' and all(r['all_expected_trials_complete'] and r['uncertain_steps']==0 for r in results.values())
            complete=complete and all(t['valid_complete_episode'] for arm in phases.values() for t in arm['evaluation'])
            flats=[t for arm in phases.values() for t in arm['retention']]
            eligible=complete and all(t['flat_retention']['sample_count']==1000 and t['flat_retention']['steady_sample_count']==750 for t in flats)
            summary={role:outcomes(phases[role]['evaluation']) for role in ('candidate','base')}
            runs.append({'learning_seed':run['learning_seed'],'candidate':summary['candidate']['completions'],
                'base':summary['base']['completions'],'complete':complete,'safe':all(v['safety_violations']==0 for v in summary.values()),
                'retention':bool(eligible and retention_pass(phases['candidate']['retention'],phases['base']['retention'])), 'outcomes':summary})
        report['terrains'][str(terrain)]=runs
    assert len(sources)==len(objectives)==1 and report['controls']<=p['max_controls']==192000
    report.update(status='complete',new_development_transfer_passed=qualify(report['terrains']),
                  full_policy_program_controls=lock['prior_program_controls']+report['controls'])
    write_json_atomic(Path(args.output),report)
    print({k:v for k,v in report.items() if k not in ('children','terrains')})


if __name__=='__main__':main()
