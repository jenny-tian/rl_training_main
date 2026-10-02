"""Audit locked code selection and paired new-start headroom, not stable gait."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from docs.analysis.audit_sac_huber_coverage import read, trace_arrays, equal_state
from docs.analysis.run_c0_headroom import grid_record, select_grid
from rl_training.online_adaptation.episode_search import EpisodeSearch
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.structured_code import fixed_joint_basis


def headroom(candidate, base, code):
    if len(candidate)!=3 or len(base)!=3 or not any(code):
        return {'passed':False,'reason':'requires nonzero code and three complete paired trials'}
    if not all(t['valid_complete_episode'] for t in candidate+base):
        return {'passed':False,'reason':'incomplete episode in paired verification'}
    c=np.array([t['progress_m'] for t in candidate]);b=np.array([t['progress_m'] for t in base])
    if not np.isfinite(c).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite progress')
    cs=np.array([t['completion'] for t in candidate],dtype=bool)
    bs=np.array([t['completion'] for t in base],dtype=bool)
    extra=bool(cs.sum()>bs.sum() and any(cs[i] and not bs[i] and all(c[j]>=b[j] for j in range(3) if j!=i) for i in range(3)))
    cm,bm=float(np.median(c)),float(np.median(b))
    ratio=cm/bm if bm>0 else None
    progress=ratio is not None and ratio>=1.1
    no_extra_falls=sum(t['fell'] for t in candidate)<=sum(t['fell'] for t in base)
    return {'passed':bool(no_extra_falls and (extra or progress)), 'extra_completion_no_other_progress_loss':extra,
            'median_progress_ratio':ratio,'no_increase_in_falls':no_extra_falls,
            'candidate_safety_violations':sum(t['safety_violation'] for t in candidate)}


def audit_child(directory,p,terrain,seed,code,expected_trials,*,expected_phases=None):
    expected_phases = expected_phases or {"evaluation": expected_trials}
    assert sum(expected_phases.values()) == expected_trials
    state=read(directory/'session_summary.json');manifest=read(directory/'ledger/manifest.json')
    assert manifest['terrain_seed']==terrain and manifest['learner_seed']==seed
    assert manifest['base']['sha256']==p['base_sha256']
    assert manifest['worker_configuration']['arguments']['fixed_code']==code
    assert all(value==0 for phase,value in state['spent'].items() if phase not in expected_phases)
    result={'spent':sum(state['spent'].values()),'uncertain_steps':state['uncertain_steps'],'trials':[],
            'source_sha256':manifest['worker_configuration']['source_sha256'],'input_hashes':{},
            'all_expected_trials_complete':Counter(t['phase'] for t in state['trials'].values())==Counter(expected_phases)}
    for trial in state['trials'].values():
        row={'id':trial['id'],'phase':trial['phase'],'status':trial['status'],'reset_seed':trial['reset_seed'],
             'valid_complete_episode':False}
        result['trials'].append(row)
        assert trial['phase'] in expected_phases
        if trial['status']!='complete':
            result['all_expected_trials_complete']=False
            continue
        path=Path(trial['output']);summary=read(path/'episode_summary.json');trace,arrays,hashes=trace_arrays(path)
        assert trace['status']=='complete' and trace['steps']==summary['steps']==trial['confirmed_steps']
        assert summary['update_attempts']==summary['actor_updates']==summary['critic_updates']==0
        assert summary['replay_transitions_written']==summary['actor_transitions_committed']==0
        before=torch.load(path/'residual_before.pt',map_location='cpu',weights_only=False)
        after=torch.load(path/'residual_after.pt',map_location='cpu',weights_only=False)
        equal_state(before,after)
        assert torch.count_nonzero(before['mean_layer.weight'])==0
        expected=torch.zeros(12) if code is None else fixed_joint_basis()@torch.tensor(code)
        torch.testing.assert_close(before['mean_layer.bias'],expected,atol=1e-7,rtol=0)
        assert not (path/'session.pt').exists()
        row.update(valid_complete_episode=summary['task_stop_reason'] in EpisodeSearch.TERMINALS,
            completion=summary['task_stop_reason']=='completion',stop_reason=summary['task_stop_reason'],
            progress_m=summary['max_x']-summary['initial_x'],fell=summary['fell'],safety_violation=summary['safety_violation'],
            steps=summary['steps'],initial_root_state=trace['metadata']['initial_root_state'],
            requested_residual_rms=float(np.sqrt(np.mean(arrays['requested_residual_action'].astype(float)**2))),
            executed_residual_rms=float(np.sqrt(np.mean(arrays['executed_residual_action'].astype(float)**2))),
            control_timeouts=int(arrays['control_timed_out'].sum()),frozen_actor_equal=True,zero_updates_and_writes=True)
        if trial['phase']=='retention':
            row['flat_retention']=summary['flat_retention']
        result['input_hashes'].update(hashes)
        for file in (path/'episode_summary.json',path/'metrics.jsonl',path/'residual_before.pt',path/'residual_after.pt'):
            result['input_hashes'][str(file)]=sha256_file(file)
    for file in (directory/'session_summary.json',directory/'ledger/manifest.json'):
        result['input_hashes'][str(file)]=sha256_file(file)
    return result


def check_pairs(a,b):
    assert len(a)==len(b)
    for left,right in zip(a,b):
        assert left['reset_seed']==right['reset_seed']
        if left['status']==right['status']=='complete':
            np.testing.assert_allclose(left['initial_root_state'],right['initial_root_state'],atol=1e-6,rtol=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();p=read(args.protocol);root,out=Path(p['output']),Path(args.output)
    assert p['schema']=='lite3_c0_grid_headroom_v1'
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','protocol_sha256':sha256_file(args.protocol),'terrains':{},'actual_control_transitions':0,
            'uncertain_steps':0,'route_qualified':False,'goal_complete':False}
    sources=set()
    for terrain in p['terrains']:
        execution=read(root/f'execution-{terrain}.json');assert execution['status']!='running'
        selection=read(root/f'selection-{terrain}.json')
        assert sha256_file(root/f'selection-{terrain}.json')==execution['selection_sha256']
        records=[grid_record(p,terrain,i,execution['stages'][f'grid-{i:02d}']['status']=='complete') for i in range(len(p['codes']))]
        assert records==selection['grid_records'] and select_grid(records)==selection['selected_indices']
        selected=selection['selected_indices'];item={'selection':selection,'children':{},'headroom':{}}
        specifications=[(f'grid-{i:02d}',p['grid_seed'],code,1) for i,code in enumerate(p['codes'])]
        if len(selected)==2:
            specifications += [('verify-base',p['verification_seed'],None,3)]
            specifications += [(f'verify-rank{i+1}',p['verification_seed'],p['codes'][j],3) for i,j in enumerate(selected)]
        for stage,seed,code,trials in specifications:
            child=audit_child(root/f'{stage}-{terrain}',p,terrain,seed,code,trials)
            item['children'][stage]=child;sources.add(child['source_sha256'])
            report['actual_control_transitions']+=child['spent'];report['uncertain_steps']+=child['uncertain_steps']
        for i in range(1,len(p['codes'])):
            check_pairs(item['children']['grid-00']['trials'],item['children'][f'grid-{i:02d}']['trials'])
        if len(selected)==2:
            base=item['children']['verify-base']['trials']
            for i,index in enumerate(selected):
                candidate=item['children'][f'verify-rank{i+1}']['trials'];check_pairs(candidate,base)
                item['headroom'][f'rank{i+1}']=headroom(candidate,base,p['codes'][index])
        assert sum(c['spent'] for c in item['children'].values())<=p['per_terrain_max_steps']
        report['terrains'][str(terrain)]=item;write_json_atomic(out/'report.json',report)
    assert len(sources)==1 and report['actual_control_transitions']<=p['max_actual_control_transitions']
    report.update(status='complete',headroom_supported=any(x['passed'] for t in report['terrains'].values() for x in t['headroom'].values()))
    write_json_atomic(out/'report.json',report)
    print({k:v for k,v in report.items() if k!='terrains'})


if __name__=='__main__':main()
