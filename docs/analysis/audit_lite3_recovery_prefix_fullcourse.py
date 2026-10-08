"""Verify fixed TRAIN-case prefix follow-through to the original course endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    p=json.loads(Path(args.protocol).read_text());assert sha256_file(p['screen_audit'])==p['screen_audit_sha256']
    screen=json.loads(Path(p['screen_audit']).read_text());assert screen['confirmation_passed']
    assert screen['selected_prefix']==p['selected_prefix']
    assert sha256_file(p['selected_prefix']['path'])==p['selected_prefix']['sha256']
    code=np.load(p['selected_prefix']['path'],allow_pickle=False)
    results={};controls=0;evidence={str(Path(args.protocol)):sha256_file(args.protocol)}
    for spec in p['runs']:
        root=Path(spec['output']);report=json.loads((root/'report.json').read_text());audit_path=Path(spec['audit']);audit=json.loads(audit_path.read_text())
        child=json.loads(Path(spec['protocol']).read_text());assert report['status']=='complete' and audit['status']=='verified'
        assert report['protocol_sha256']==sha256_file(spec['protocol'])==spec['protocol_sha256']
        assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        assert report['models']['trial']['actual_sha256']==p['model_sha256']
        assert child['recovery_resets']['fixed_train_case']==26 and child['recovery_resets']['split']=='train'
        if spec['arm']=='prefix':
            assert child['action_prefix']['sha256']==p['expanded_prefix_sha256']
            expanded=np.load(child['action_prefix']['path'],allow_pickle=False);expected=np.zeros_like(expanded);expected[p['target_env_ids']]=code
            np.testing.assert_array_equal(expanded,expected)
        else:assert 'action_prefix' not in child
        controls+=report['ledger']['confirmed'];assert report['ledger']['charged']==report['ledger']['confirmed']
        phase=report['phases']['recovery-trial'];folder=root/'recovery-trial';initial=np.load(folder/'initial.npz',allow_pickle=False);delays=json.loads((folder/'delays.json').read_text());arrays={}
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            with np.load(path) as d:
                for k in ('active','post_root_state','foot_contact_force_w'):arrays.setdefault(k,[]).append(d[k])
        arrays={k:np.concatenate(v) for k,v in arrays.items()};rows=[];groups={}
        for e in phase['episodes']:
            if e['env_id'] not in p['target_env_ids']:continue
            i=e['env_id'];n=int(arrays['active'][:,i].sum());assert n==e['steps'] and e['case']==26
            state=arrays['post_root_state'][:n,i]
            upright=(1-2*np.sum(state[:,4:6]**2,axis=-1)>=np.cos(np.deg2rad(10)))&(np.arange(n)>=12)
            loaded=(arrays['foot_contact_force_w'][:n,i,:,2]>5).sum(-1)>=2
            grounded=n>=25 and bool(((np.convolve(upright,np.ones(25,int),'valid')==25)&(np.convolve(loaded,np.ones(25,int),'valid')>=5)).any())
            recovered=bool(e['success'] and grounded)
            signature=[]
            for joint in ('Hip','Knee'):
                values=[delays[joint][key][i] for key in ('efforts_delay_buffer','positions_delay_buffer','velocities_delay_buffer')]
                assert len(set(values))==1 and values[0] in (0,1);signature.append(str(values[0]))
            group=groups.setdefault('/'.join(signature),dict(success=0,total=0));group['success']+=recovered;group['total']+=1
            rows.append(dict(env_id=i,recovered_and_completed=recovered,steps=n,rms_m=e['body_lateral_rms_m'],peak_m=e['max_body_lateral_m'],initial_lateral_m=float(initial['root'][i,1]-initial['origins'][i,1]),peak_from_start_m=float(np.abs(state[:,1]-initial['root'][i,1]).max()),failure_terms=e.get('failure_terms',{})))
        results[f"{spec['seed']}-{spec['arm']}"]=dict(rows=rows,success=sum(v['recovered_and_completed'] for v in rows),total=len(rows),groups=groups,
            rms_m=float(np.mean([v['rms_m'] for v in rows])),peak_m=max(v['peak_m'] for v in rows),peak_from_start_m=max(v['peak_from_start_m'] for v in rows),initial_sha256=phase['initial_sha256'])
        evidence[str(audit_path)]=sha256_file(audit_path)
    gates={}
    for seed in p['seeds']:
        a=results[f'{seed}-base'];b=results[f'{seed}-prefix'];assert a['initial_sha256']==b['initial_sha256']
        coverage=set(b['groups'])=={'0/0','0/1','1/0','1/1'}
        gates[str(seed)]=coverage and b['success']>=p['minimum_positives'] and b['success']>a['success'] and all(v['success']/v['total']>=p['minimum_stratum_fraction'] for v in b['groups'].values())
    assert controls<=p['max_controls']
    write_json_atomic(args.output,dict(status='verified',controls=controls,uncertain=0,results=results,gates=gates,fullcourse_followthrough_passed=all(gates.values()),evidence=evidence,
        scope='Fixed prefix plus frozen PPO on TRAIN case26, repeated slots and new startup seeds. Not a newly trained neural policy, held-out recovery qualification, general controller or hardware evidence.'))


if __name__=='__main__':main()
