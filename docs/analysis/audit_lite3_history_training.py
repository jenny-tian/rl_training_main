"""Audit the registered49D pair's fixed TRAIN replay, model and update invariants."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','protocol','run','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--audit',action='append',required=True)
    args=parser.parse_args();torch.set_num_threads(4);root=Path(args.run);evidence={}
    def read(path):
        evidence[str(path)]=sha256_file(path)
        return json.loads(Path(path).read_text())
    plan=read(args.plan);p=read(args.protocol);r=read(root/'report.json');historical=read(plan['historical_protocol'])
    assert r['status']=='complete' and r['protocol_sha256']==sha256_file(args.protocol)
    for path in args.audit:
        a=read(path);assert a['status']=='verified' and a['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
    assert len(args.audit)==5
    for key in historical:
        if key not in ('registered_at_utc','arm','scope','selection','mean_anchor'):
            assert p[key]==historical[key],key
    assert set(p)==set(historical)|{'route_history','route_history_smoke'}
    assert {k:v for k,v in p['route_history'].items() if k!='enabled'}==plan['history_options']
    assert r['route_history_options']==p['route_history']
    assert p['num_envs']==64 and p['rollout_steps']==32 and p['iterations']==256
    assert r['ledger']['confirmed']==r['ledger']['charged']==plan['training_controls_per_arm']==524288
    assert r['ledger']['uncertain_transitions']==0 and r['optimizer_steps']==[5120]
    assert r['warmstart']['actor_loaded'] and not r['warmstart']['critic_loaded'] and not r['warmstart']['optimizer_loaded']
    assert r['recovery_anchor_model_unchanged'] and r['mean_anchor']['teacher_unchanged']
    spec=p['mean_anchor'];old=historical['mean_anchor'];observed=r['mean_anchor']
    for key in old:
        if key not in ('dataset','dataset_sha256','dataset_audit','dataset_audit_sha256'):assert spec[key]==old[key]
    assert sha256_file(spec['dataset'])==spec['dataset_sha256']==plan['dataset_sha256']==observed['dataset_sha256']
    assert sha256_file(spec['dataset_audit'])==spec['dataset_audit_sha256']==plan['dataset_audit_sha256']
    read(spec['dataset_audit']);evidence[spec['dataset']]=sha256_file(spec['dataset'])
    assert observed['validation_rows_used']==0 and 'fixed_target_groups' not in observed and 'tail_anchor' not in r
    assert observed['group_weights']==spec['group_weights']=={'correction':.5,'flat':1.,'rough101':1.,'rough107':1.}
    assert observed['calls']==5120 and observed['rows']==plan['replay_queries_per_arm']==2621440
    assert observed['weight_denominator']=='all sampled rows times12; not sum of weights'
    generator=torch.Generator().manual_seed(spec['seed']);digest=hashlib.sha256()
    with np.load(spec['dataset'],allow_pickle=False) as data:
        counts={k:len(data[k+'_train_obs']) for k in spec['batch_groups']}
        assert counts==observed['group_rows']
        for _ in range(observed['calls']):
            for k,n in spec['batch_groups'].items():digest.update(torch.randint(counts[k],(n,),generator=generator).numpy().tobytes())
        assert digest.hexdigest()==observed['draw_sha256']
        payload=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
        assert sha256_file(root/'final.pt')==r['final_checkpoint']['sha256']
        weight=payload['actor_state_dict']['mlp.0.weight'];assert weight.shape==(512,49)
        assert bool(weight[:,12].abs().sum()>0)==p['route_history']['enabled']
        candidate=FrozenPPOExpert(payload['actor_state_dict'])
        reference=FrozenPPOExpert(torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict'])
        drift={}
        with torch.no_grad():
            for group in ('correction','flat','rough101','rough107','post_recovery'):
                x=torch.from_numpy(data[group+'_validation_obs'].copy())
                if not p['route_history']['enabled']:x[:,12]=0
                errors=torch.cat([(candidate(x[i:i+4096])-reference(x[i:i+4096])).square().mean(-1) for i in range(0,len(x),4096)])
                drift[group]=dict(rows=len(x),mean_squared_raw_action_change=float(errors.mean()))
    gradient=root/'gradient_diagnostics.jsonl';assert sha256_file(gradient)==r['gradient_diagnostics']['sha256']
    records=[json.loads(line) for line in gradient.read_text().splitlines()]
    assert len(records)==r['gradient_diagnostics']['records']==256 and [x['update'] for x in records]==list(range(1,257))
    assert all(x['minibatch']==0 and all(np.isfinite(y) and y>=0 for y in x['norms'].values()) for x in records)
    evidence[str(gradient)]=sha256_file(gradient)
    for path,h in r['source_files'].items():assert sha256_file(path)==h
    write_json_atomic(args.output,dict(status='verified',controls=524288,mean_anchor_calls=5120,replay_rows=2621440,
        draw_sha256=digest.hexdigest(),stored_action_target_rows_for_updates=0,validation_rows_used_for_updates=0,
        tail_replay_rows=0,gradient_records=256,history_enabled=p['route_history']['enabled'],
        final_checkpoint=r['final_checkpoint'],history_column_norm=float(weight[:,12].norm()),
        offline_action_drift_from_initial_actor=drift,evidence=evidence,
        scope='Fixed49D pair with original48D frozen targets. Main draws, row counts, correction0.5 weights and old denominator preserved. Zero arm masks only inserted column in both online and replay observations. Offline validation diagnostic only.'))


if __name__=='__main__':main()
