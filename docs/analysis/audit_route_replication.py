"""Aggregate two independently trained route policies; new-terrain qualification stays separate."""
import argparse
from pathlib import Path
from docs.analysis.audit_sac_huber_coverage import read
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def qualify(terrains):
    return bool(terrains and all(
        len(runs)==2 and sum(r['candidate'] for r in runs)>=9
        and all(r['complete'] and r['retention'] and r['safe'] and r['candidate']>=4
                and r['candidate']>r['base'] for r in runs)
        for runs in terrains.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',required=True);parser.add_argument('--second',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    protocols=[read(args.first),read(args.second)]
    first,second=protocols
    for key in ('base_checkpoint','base_sha256','task','num_envs','rollout_steps','iterations','initial_std',
                'ppo','symmetry','route_observations','route_training','evaluation_terrains','evaluation_trials',
                'evaluation_trial_steps','retention_trials','retention_steps'):
        assert first[key]==second[key], f'Configuration differs: {key}'
    assert first.get('flat_anchor_weight')==second.get('flat_anchor_weight')
    assert first['seed']!=second['seed'] and first['evaluation_seed']!=second['evaluation_seed']
    assert second['replication_of_protocol_sha256']==sha256_file(args.first)
    report={'status':'running','terrains':{},'runs':[], 'goal_complete':False,
            'remaining':'New-development terrain confirmation and complete gait/qualification audit.'}
    sources=set();checkpoints=set()
    for protocol,path in zip(protocols,(args.first,args.second)):
        root=Path(protocol['output']);train=read(root/'train/report.json');audit=read(root/'evaluation_audit.json');lock=read(root/'evaluation_lock.json')
        assert train['status']==audit['status']=='complete'
        assert train['protocol_sha256']==audit['protocol_sha256']==sha256_file(path)
        assert lock['training_report_sha256']==sha256_file(root/'train/report.json')
        assert train['ppo_updates']==protocol['iterations'] and train['ledger']['confirmed']==protocol['training_control_transitions']
        assert train['ledger']['uncertain_transitions']==audit['uncertain_steps']==0
        assert train['base_sha256']==protocol['base_sha256']
        assert train['mean_actor_changed'] and train['warmstart']['critic_loaded'] is False and train['warmstart']['optimizer_loaded'] is False
        assert sha256_file(train['final_checkpoint']['path'])==lock['candidate']['sha256']==train['final_checkpoint']['sha256']
        sources.add(train['source_sha256']);checkpoints.add(train['final_checkpoint']['sha256'])
        report['runs'].append({'seed':protocol['seed'],'evaluation_seed':protocol['evaluation_seed'],
            'protocol_sha256':sha256_file(path),'evaluation_audit_sha256':sha256_file(root/'evaluation_audit.json'),
            'checkpoint':train['final_checkpoint'],'training_controls':train['ledger']['confirmed'],'evaluation_controls':audit['evaluation_controls']})
        for terrain in protocol['evaluation_terrains']:
            t=audit['terrains'][str(terrain)]
            report['terrains'].setdefault(str(terrain),[]).append({'seed':protocol['seed'],
                'candidate':t['outcomes']['candidate']['completions'],'base':t['outcomes']['base']['completions'],
                'complete':t['complete'],'retention':t['retention_pass'],
                'safe':all(v['safety_violations']==0 for v in t['outcomes'].values())})
    assert len(sources)==1 and len(checkpoints)==2
    report.update(status='complete',independent_route_replication_passed=qualify(report['terrains']))
    write_json_atomic(Path(args.output),report)
    print({k:v for k,v in report.items() if k!='runs'})


if __name__=='__main__':main()
