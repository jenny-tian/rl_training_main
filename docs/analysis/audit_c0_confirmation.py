"""Independent fixed-code confirmation; never pool old successes to rescue failures."""
import argparse
from pathlib import Path

from docs.analysis.audit_c0_headroom import audit_child, check_pairs, headroom
from docs.analysis.audit_learning_replication import retention_pass
from docs.analysis.audit_sac_huber_coverage import read
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def assess(candidate,base,code):
    phases={phase:{role:[t for t in child['trials'] if t['phase']==phase]
                   for role,child in (('candidate',candidate),('base',base))}
            for phase in ('evaluation','retention')}
    task=headroom(phases['evaluation']['candidate'],phases['evaluation']['base'],code)
    flat=phases['retention']
    valid_flat=all(len(rows)==2 and all(t['status']=='complete' for t in rows) for rows in flat.values())
    retained=valid_flat and retention_pass(flat['candidate'],flat['base'])
    complete=all(c['all_expected_trials_complete'] for c in (candidate,base))
    return {'task_confirmation':task,'retention_pass':bool(retained),
            'confirmation_pass':bool(complete and task['passed'] and retained),
            'stable_gait_qualified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();p=read(args.protocol);root,out=Path(p['output']),Path(args.output)
    assert p['schema']=='lite3_c0_independent_confirmation_v1'
    assert sha256_file(p['prior_headroom_report'])==p['prior_headroom_sha256']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','protocol_sha256':sha256_file(args.protocol),'terrains':{},
            'actual_control_transitions':0,'uncertain_steps':0,'goal_complete':False,'limits':p['limits']}
    sources=set()
    for terrain in p['terrains']:
        assert read(root/f'execution-{terrain}.json')['status']!='running'
        children={role:audit_child(root/f'{role}-{terrain}',p,terrain,p['learner_seed'],
            p['codes'][str(terrain)] if role=='candidate' else None,5,
            expected_phases={'evaluation':3,'retention':2}) for role in ('candidate','base')}
        for child in children.values():
            sources.add(child['source_sha256']);report['actual_control_transitions']+=child['spent']
            report['uncertain_steps']+=child['uncertain_steps']
        for phase in ('evaluation','retention'):
            check_pairs(*[[t for t in children[role]['trials'] if t['phase']==phase] for role in ('candidate','base')])
        result=assess(children['candidate'],children['base'],p['codes'][str(terrain)])
        result['children']=children;report['terrains'][str(terrain)]=result
        write_json_atomic(out/'report.json',report)
    assert len(sources)==1 and report['actual_control_transitions']<=p['max_actual_control_transitions']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    report.update(status='complete',confirmation_pass=all(t['confirmation_pass'] for t in report['terrains'].values()))
    write_json_atomic(out/'report.json',report)
    print({k:v for k,v in report.items() if k!='terrains'})


if __name__=='__main__':main()
