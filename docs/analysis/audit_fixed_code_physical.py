"""Audit short fixed-code execution windows; never claim task or gait success."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from docs.analysis.audit_sac_huber_coverage import read, trace_arrays, equal_state
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.structured_code import fixed_joint_basis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    p=read(args.protocol)
    assert p['schema']=='lite3_fixed_code_physical_check_v1'
    root,out=Path(p['output']),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','protocol_sha256':sha256_file(args.protocol),'terrains':{},
            'actual_control_transitions':0,'uncertain_steps':0,'goal_complete':False,
            'limits':p['limits']}
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    for terrain in p['terrains']:
        assert read(root/f'execution-{terrain}.json')['status']!='running'
        items,arrays_by_key,initials={},{},{}
        for arm,code in p['arms'].items():
            session=root/f'{arm}-{terrain}'
            state=read(session/'session_summary.json')
            manifest=read(session/'ledger/manifest.json')
            assert manifest['base']['sha256']==p['base_sha256']
            assert manifest['worker_configuration']['arguments']['fixed_code']==code
            report['actual_control_transitions']+=sum(state['spent'].values())
            report['uncertain_steps']+=state['uncertain_steps']
            item={'trials':[],'spent':state['spent']}
            for trial in state['trials'].values():
                entry={'phase':trial['phase'],'status':trial['status'],'reset_seed':trial['reset_seed']}
                item['trials'].append(entry)
                if trial['status']!='complete':continue
                path=Path(trial['output']);summary=read(path/'episode_summary.json')
                trace,arrays,hashes=trace_arrays(path)
                assert trace['status']=='complete'
                assert summary['steps']==trace['steps']==trial['confirmed_steps']
                assert summary['actor_updates']==summary['critic_updates']==summary['update_attempts']==0
                assert summary['replay_transitions_written']==summary['actor_transitions_committed']==0
                assert summary['terrain_valid_fraction']==1.
                before=torch.load(path/'residual_before.pt',map_location='cpu',weights_only=False)
                after=torch.load(path/'residual_after.pt',map_location='cpu',weights_only=False)
                equal_state(before,after)
                requested=arrays['requested_residual_action'].reshape(summary['steps'],12)
                expected=np.zeros(12) if code is None else (.5*torch.tensor([.24,.2,.2]*4)*torch.tanh(fixed_joint_basis()@torch.tensor(code))).numpy()
                error=float(np.max(np.abs(requested-expected)))
                entry.update(steps=summary['steps'],stop_reason=summary['task_stop_reason'],
                    fell=summary['fell'],safety_violation=summary['safety_violation'],
                    request_formula_max_abs_error=error,request_formula_pass=error<=1e-6,
                    executed_residual_rms=float(np.sqrt(np.mean(arrays['executed_residual_action'].astype(float)**2))),
                    control_timeouts=int(arrays['control_timed_out'].sum()),input_hashes=hashes,
                    summary_sha256=sha256_file(path/'episode_summary.json'),
                    frozen_actor_equal=True,zero_updates_and_replay_writes=True)
                arrays_by_key[arm,trial['phase']]=arrays
                initials[arm,trial['phase']]=(trial['reset_seed'],trace['metadata']['initial_root_state'])
            items[arm]=item
        parity={}
        for phase in ('evaluation','retention'):
            if ('base',phase) not in arrays_by_key or ('zero',phase) not in arrays_by_key:continue
            seed,initial=initials['base',phase]
            for arm in p['arms']:
                if (arm,phase) in initials:
                    other_seed,other_initial=initials[arm,phase]
                    assert seed==other_seed
                    np.testing.assert_allclose(initial,other_initial,atol=1e-6,rtol=0)
            a,b=arrays_by_key['base',phase],arrays_by_key['zero',phase]
            parity[phase]={}
            for key in ('root_state','base_action','executed_action','reward','learning_terminated'):
                if a[key].shape!=b[key].shape:
                    parity[phase][key]={'equal_length':False,'passed':False}
                else:
                    error=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float))))
                    parity[phase][key]={'max_abs_error':error,'passed':error<=1e-6}
        full=all(len(v['trials'])==2 and all(t['status']=='complete' and t.get('steps')==64
                 and t.get('request_formula_pass') and not t.get('fell') and not t.get('safety_violation')
                 and t.get('control_timeouts')==0 for t in v['trials']) for v in items.values())
        passed=full and len(parity)==2 and all(v['passed'] for phase in parity.values() for v in phase.values())
        report['terrains'][str(terrain)]={'arms':items,'base_zero_parity':parity,'engineering_pass':passed}
        write_json_atomic(out/'report.json',report)
    assert report['actual_control_transitions']<=p['max_actual_control_transitions']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    report.update(status='complete',engineering_pass=all(t['engineering_pass'] for t in report['terrains'].values()))
    write_json_atomic(out/'report.json',report)
    print({k:v for k,v in report.items() if k!='terrains'})


if __name__=='__main__':main()
