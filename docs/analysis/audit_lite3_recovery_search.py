"""Independently reconstruct prefix-search selection, short recovery and cost."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def metrics(folder,p):
    arrays={}
    for path in sorted((folder/'evaluation/smoke-trial/trajectory').glob('chunk-*.npz')):
        with np.load(path,allow_pickle=False) as d:
            for k in ('active','done','success','post_root_state','foot_contact_force_w','joint_computed_torque','joint_applied_torque'):
                arrays.setdefault(k,[]).append(d[k])
    a={k:np.concatenate(v) for k,v in arrays.items()};rows=[]
    for i in p['target_env_ids']:
        n=int(a['active'][:,i].sum());assert np.array_equal(a['active'][:,i],np.arange(len(a['active']))<n)
        roots=a['post_root_state'][:n,i]
        tilt=np.rad2deg(np.arccos(np.clip(1-2*np.sum(roots[:,4:6]**2,axis=-1),-1,1)))
        upright=(tilt<=10)&(np.arange(n)>=p['knots']*p['hold_steps'])
        loaded=(a['foot_contact_force_w'][:n,i,:,2]>5).sum(-1)>=2
        grounded=False
        if n>=25:grounded=bool(((np.convolve(upright,np.ones(25,int),'valid')==25)&(np.convolve(loaded,np.ones(25,int),'valid')>=5)).any())
        runs=np.diff(np.flatnonzero(np.r_[True,~upright,True]))-1
        best=int(runs.max()) if len(runs) else 0
        survived=n==p['steps'] and not a['done'][:n,i].any();completed=bool(a['success'][:n,i].any())
        positive=bool((survived or completed) and grounded)
        rows.append(dict(env_id=i,active_controls=n,short_recovery=positive,survived_horizon=bool(survived),completed_course=completed,
            longest_post_handoff_upright=best,mean_tilt_deg=float(tilt.mean()),score=float(1000*positive+2*n-float(tilt.mean())+.5*best),
            peak_applied_joint_torque_nm=float(np.abs(a['joint_applied_torque'][:n,i]).max()),
            changed_torque_component_fraction=float(np.mean(np.abs(a['joint_applied_torque'][:n,i]-a['joint_computed_torque'][:n,i])>1e-5))))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','run','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p=json.loads(Path(args.protocol).read_text());root=Path(args.run);r=json.loads((root/'report.json').read_text())
    assert r['status']=='complete' and r['protocol_sha256']==sha256_file(args.protocol)
    controls=0;used=set();summary={};evidence={str(Path(args.protocol)):sha256_file(args.protocol),str(root/'report.json'):sha256_file(root/'report.json')}
    def phase(case,label,population,seed):
        nonlocal controls
        folder=root/f'case{case}'/label;key=f'case{case}/{label}';used.add(key)
        child=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'evaluation/report.json').read_text());audit=json.loads((folder/'audit.json').read_text())
        assert report['status']=='complete' and audit['status']=='verified'
        assert report['protocol_sha256']==sha256_file(folder/'protocol.json')==r['phases'][key]['protocol_sha256']
        assert sha256_file(folder/'audit.json')==r['phases'][key]['audit_sha256']
        assert audit['evidence'][str(folder/'evaluation/report.json')]==sha256_file(folder/'evaluation/report.json')
        assert child['recovery_resets']['fixed_train_case']==case and child['evaluation_seed']==seed
        assert child['models']==p['evaluation_template']['models'] and child['parent_protocol_sha256']==r['protocol_sha256']
        assert child['recovery_resets']['bank_sha256']==p['evaluation_template']['recovery_resets']['bank_sha256']
        assert child['recovery_resets']['split']=='train'
        if population is None:assert 'action_prefix' not in child
        else:
            prefix=np.load(folder/'prefix.npy');expected=np.zeros_like(prefix);expected[p['target_env_ids']]=population
            np.testing.assert_array_equal(prefix,expected);assert sha256_file(folder/'prefix.npy')==child['action_prefix']['sha256']
        assert report['ledger']['confirmed']==report['ledger']['charged']==p['num_envs']*p['steps']
        controls+=report['ledger']['confirmed']
        rows=metrics(folder,p)
        for a,b in zip(rows,r['phases'][key]['rows']):
            for field,value in b.items():
                if isinstance(value,float):np.testing.assert_allclose(a[field],value,atol=1e-4,rtol=1e-6)
                else:assert a[field]==value
        initial=report['phases']['smoke-trial']['initial_sha256'];assert initial==r['phases'][key]['initial_sha256']
        evidence[str(folder/'audit.json')]=sha256_file(folder/'audit.json')
        return rows,initial
    for case in p['training_cases']:
        baseline,initial=phase(case,'baseline',None,p['search_seed'])
        rng=np.random.default_rng(p['proposal_seed']+case);mean=np.zeros((p['knots'],12),np.float32);std=np.full_like(mean,p['initial_std'])
        best=np.zeros_like(mean);score=-float('inf');positive=False;rounds=[];selected=None
        for iteration in range(p['generations']):
            samples=np.clip(rng.normal(mean,std,size=(len(p['target_env_ids']),*mean.shape)),-p['bound'],p['bound']).astype(np.float32)
            samples[0]=0;samples[1]=best
            label=f'generation{iteration:02d}';rows,paired=phase(case,label,samples,p['search_seed']);assert paired==initial
            scores=np.array([v['score'] for v in rows]);idx=int(np.argmax(scores))
            if scores[idx]>score:best=samples[idx].copy();score=float(scores[idx]);positive=rows[idx]['short_recovery'];selected=dict(phase=label,**rows[idx])
            elite=np.argsort(-scores,kind='stable')[:p['elite_count']]
            assert elite.tolist()==r['cases'][str(case)]['history'][iteration]['elite']
            mean=samples[elite].mean(0);std=np.maximum(samples[elite].std(0),p['std_floor'])
            rounds.append(dict(generation=iteration,short_positives=sum(v['short_recovery'] for v in rows),best_score=float(scores[idx])))
        np.testing.assert_array_equal(best,np.load(root/f'case{case}/selected_prefix.npy'))
        assert sha256_file(root/f'case{case}/selected_prefix.npy')==r['cases'][str(case)]['selected_sha256']
        assert positive==r['cases'][str(case)]['search_positive']
        confirmations=[]
        if positive:
            for seed in p['confirmation_seeds']:
                a,initial=phase(case,f'confirm{seed}-base',None,seed)
                b,paired=phase(case,f'confirm{seed}-prefix',np.broadcast_to(best,(len(p['target_env_ids']),*best.shape)),seed);assert paired==initial
                confirmations.append(dict(seed=seed,baseline_positive=sum(x['short_recovery'] for x in a),prefix_positive=sum(x['short_recovery'] for x in b)))
        assert confirmations==r['cases'][str(case)]['confirmations']
        confirmed=bool(confirmations) and all(v['prefix_positive']>=p['minimum_confirmation_positives'] and v['prefix_positive']>v['baseline_positive'] for v in confirmations)
        assert confirmed==r['cases'][str(case)]['confirmed_positive']
        summary[str(case)]=dict(baseline_positive=sum(v['short_recovery'] for v in baseline),generations=rounds,selected=selected,search_positive=positive,confirmations=confirmations,confirmed_positive=confirmed)
    assert used==set(r['phases'])
    assert controls==r['ledger']['confirmed']==r['ledger']['charged'] and controls<=p['max_controls']
    write_json_atomic(args.output,dict(status='verified',controls=controls,uncertain=0,cases=summary,evidence=evidence,
        scope='TRAIN-source search and paired new-reset confirmations only. Short recovery is not full-course qualification, independent learning replication or held-out recovery success. Search failure cannot prove physical impossibility.'))


if __name__=='__main__':main()
