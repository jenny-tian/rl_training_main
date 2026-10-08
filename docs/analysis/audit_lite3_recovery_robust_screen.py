"""Verify all-candidate delay-stratified screening and untouched confirmations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic
from audit_lite3_recovery_search import metrics


def groups(folder,rows):
    d=json.loads((folder/'evaluation/smoke-trial/delays.json').read_text());counts={}
    for row in rows:
        i=row['env_id'];lag=[]
        for joint in ('Hip','Knee'):
            values=[d[joint][name][i] for name in ('efforts_delay_buffer','positions_delay_buffer','velocities_delay_buffer')]
            assert len(set(values))==1 and values[0] in (0,1);lag.append(str(values[0]))
        key='/'.join(lag);value=counts.setdefault(key,dict(positive=0,total=0));value['positive']+=int(row['short_recovery']);value['total']+=1
    assert set(counts)=={'0/0','0/1','1/0','1/1'}
    return dict(groups=counts,worst_fraction=min(v['positive']/v['total'] for v in counts.values()),positive=sum(v['positive'] for v in counts.values()),mean_score=float(np.mean([r['score'] for r in rows])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','run','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p=json.loads(Path(args.protocol).read_text());root=Path(args.run);r=json.loads((root/'report.json').read_text())
    assert r['status']=='complete' and r['protocol_sha256']==sha256_file(args.protocol)
    assert sha256_file(p['search_audit'])==p['search_audit_sha256']
    parent_root=Path(p['reused_baseline_report']).parents[3];parent=json.loads((parent_root/'report.json').read_text());parent_audit=json.loads(Path(p['search_audit']).read_text())
    assert parent_audit['status']=='verified' and parent_audit['evidence'][str(parent_root/'report.json')]==sha256_file(parent_root/'report.json')
    expected=[];seen=set()
    for label,phase in parent['phases'].items():
        if not label.startswith('case26/generation'):continue
        codes=np.load(parent_root/label/'prefix.npy')
        for row in phase['rows']:
            if not row['short_recovery']:continue
            code=codes[row['env_id']];digest=hashlib.sha256(code.tobytes()).hexdigest()
            if digest in seen:continue
            seen.add(digest);expected.append((label,row['env_id'],code))
    assert len(expected)==len(p['candidates'])
    for (phase,env_id,code),candidate in zip(expected,p['candidates']):
        assert candidate['source_phase']==phase and candidate['source_env_id']==env_id
        assert sha256_file(candidate['path'])==candidate['sha256']
        assert sha256_file(parent_root/phase/'prefix.npy')==candidate['source_prefix_sha256']
        np.testing.assert_array_equal(code,np.load(candidate['path']))
    assert sha256_file(p['reused_baseline_report'])==p['reused_baseline_sha256']
    baseline_initial=json.loads(Path(p['reused_baseline_report']).read_text())['phases']['smoke-trial']['initial_sha256']
    controls=0;used=set();audits={};summaries=[]
    def phase(label,seed,code):
        nonlocal controls
        used.add(label);folder=root/label;child=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'evaluation/report.json').read_text());audit=json.loads((folder/'audit.json').read_text())
        assert audit['status']=='verified' and report['status']=='complete'
        assert audit['evidence'][str(folder/'evaluation/report.json')]==sha256_file(folder/'evaluation/report.json')
        assert report['protocol_sha256']==sha256_file(folder/'protocol.json')
        assert r['phases'][label]['audit_sha256']==sha256_file(folder/'audit.json')
        assert child['robust_parent_sha256']==r['protocol_sha256'] and child['evaluation_seed']==seed
        assert child['models']==p['evaluation_template']['models'] and child['recovery_resets']['fixed_train_case']==26
        assert child['recovery_resets']['split']=='train' and child['recovery_resets']['bank_sha256']==p['evaluation_template']['recovery_resets']['bank_sha256']
        if code is None:assert 'action_prefix' not in child
        else:
            prefix=np.load(folder/'prefix.npy');array=np.zeros_like(prefix);array[p['target_env_ids']]=code;np.testing.assert_array_equal(prefix,array)
            assert sha256_file(folder/'prefix.npy')==child['action_prefix']['sha256']
        assert report['ledger']['charged']==report['ledger']['confirmed']==p['num_envs']*p['steps'];controls+=report['ledger']['confirmed']
        rows=metrics(folder,p);summary=groups(folder,rows);stored=r['phases'][label]['summary']
        assert summary['groups']==stored['groups'] and summary['positive']==stored['positive'] and summary['worst_fraction']==stored['worst_fraction']
        np.testing.assert_allclose(summary['mean_score'],stored['mean_score'],atol=1e-4,rtol=1e-6)
        initial=report['phases']['smoke-trial']['initial_sha256'];assert initial==r['phases'][label]['initial_sha256']
        audits[str(folder/'audit.json')]=sha256_file(folder/'audit.json')
        return summary,initial
    for i,(_,_,code) in enumerate(expected):
        s,h=phase(f'candidate{i:02d}',p['search_seed'],code);assert h==baseline_initial;summaries.append(s)
    selected=max(range(len(summaries)),key=lambda i:tuple(summaries[i][k] for k in ('worst_fraction','positive','mean_score')))
    assert selected==r['selected_candidate']
    s=summaries[selected];passed=s['worst_fraction']>=p['minimum_stratum_fraction'] and s['positive']>=p['minimum_positives'];assert passed==r['screen_passed']
    confirmations=[]
    if passed:
        for seed in p['confirmation_seeds']:
            a,h=phase(f'confirm{seed}-base',seed,None);b,h2=phase(f'confirm{seed}-prefix',seed,expected[selected][2]);assert h==h2
            confirmations.append(dict(seed=seed,baseline=a,prefix=b))
    assert len(confirmations)==len(r['confirmation'])
    confirmed=bool(confirmations) and all(v['prefix']['worst_fraction']>=p['minimum_stratum_fraction'] and v['prefix']['positive']>=p['minimum_positives'] and v['prefix']['positive']>v['baseline']['positive'] for v in confirmations)
    assert confirmed==r['confirmation_passed'] and used==set(r['phases'])
    assert controls==r['ledger']['charged']==r['ledger']['confirmed'] and controls<=p['max_controls']
    write_json_atomic(args.output,dict(status='verified',controls=controls,uncertain=0,candidate_count=len(expected),summaries=summaries,selected_candidate=selected,selected_prefix=p['candidates'][selected],screen_passed=passed,confirmation_passed=confirmed,confirmations=confirmations,
        evidence={str(Path(args.protocol)):sha256_file(args.protocol),str(root/'report.json'):sha256_file(root/'report.json'),**audits},scope='Additional paid TRAIN-case26 screening across original delay strata, not an independent learning replication or held-out teacher qualification.'))


if __name__=='__main__':main()
