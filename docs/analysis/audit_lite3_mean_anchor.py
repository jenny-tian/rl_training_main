"""Audit direct replay updates, source isolation, draw stream, and paired PPO prefix."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','protocol','run','training-audit','anchor-audit','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    plan=json.loads(Path(args.plan).read_text())
    p=json.loads(Path(args.protocol).read_text())
    root=Path(args.run)
    r=json.loads((root/'report.json').read_text())
    assert r['status']=='complete' and r['protocol_sha256']==sha256_file(args.protocol)
    evidence={str(f):sha256_file(f) for f in (Path(args.plan),Path(args.protocol),root/'report.json',Path(args.training_audit),Path(args.anchor_audit))}
    for path in (args.training_audit,args.anchor_audit):
        audit=json.loads(Path(path).read_text())
        assert audit['status']=='verified' and audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
    control=json.loads(Path(plan['control_protocol']).read_text())
    assert sha256_file(plan['control_protocol'])==plan['control_protocol_sha256']
    assert sha256_file(inspect.getfile(PPO))==plan['vendor_ppo_source_sha256']
    for key,value in control.items():
        if key not in ('registered_at_utc','arm','scope','selection'):
            assert p[key]==value,key
    spec=p['mean_anchor'];observed=r['mean_anchor']
    assert spec==plan['mean_anchor'] and observed['validation_rows_used']==0
    assert observed['coefficient']==spec['coefficient'] and observed['sampling_device']=='cpu'
    assert observed['teacher_sha256']==p['base_sha256']==plan['base_sha256']
    assert observed['teacher_unchanged']
    assert sha256_file(spec['dataset'])==observed['dataset_sha256']==spec['dataset_sha256']
    assert sha256_file(spec['dataset_audit'])==spec['dataset_audit_sha256']
    evidence[spec['dataset']]=sha256_file(spec['dataset'])
    with np.load(spec['dataset'],allow_pickle=False) as dataset:
        counts={group:len(dataset[group+'_train_obs']) for group in spec['batch_groups']}
        assert counts==observed['group_rows']
        assert set(counts)=={'correction','flat','rough101','rough107'}
        calls=p['iterations']*p['ppo']['num_learning_epochs']*p['ppo']['num_mini_batches']
        assert observed['calls']==calls and observed['rows']==calls*sum(spec['batch_groups'].values())
        assert r['optimizer_steps']==[calls]
        generator=torch.Generator(device='cpu').manual_seed(spec['seed'])
        digest=hashlib.sha256()
        for _ in range(calls):
            for group,n in spec['batch_groups'].items():
                digest.update(torch.randint(counts[group],(n,),generator=generator).numpy().tobytes())
        assert digest.hexdigest()==observed['draw_sha256']
        teacher=FrozenPPOExpert(torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict'])
        assert sha256_file(root/'final.pt')==r['final_checkpoint']['sha256']
        candidate=FrozenPPOExpert(torch.load(root/'final.pt',map_location='cpu',weights_only=False)['actor_state_dict'])
        offline={}
        with torch.no_grad():
            for group in ('correction','flat','rough101','rough107','post_recovery'):
                values=dataset[group+'_validation_obs']
                deltas=torch.cat([candidate(torch.from_numpy(values[i:i+4096]))-teacher(torch.from_numpy(values[i:i+4096])) for i in range(0,len(values),4096)])
                offline[group]=dict(rows=len(values),mean_squared_raw_action_change=float(deltas.square().mean()),
                    p95_absolute_raw_action_change=float(torch.quantile(deltas.abs().flatten(),.95)),max_absolute_raw_action_change=float(deltas.abs().max()))
    updates=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    assert [u['update'] for u in updates]==list(range(1,p['iterations']+1))
    losses=[u['losses']['mean_anchor'] for u in updates]
    assert np.isfinite(losses).all() and min(losses)>=0 and max(losses)>0
    previous=Path('artifacts/recovery_return_20261008/trial/train/trajectory/chunk-00000.npz')
    first=root/'trajectory/chunk-00000.npz'
    prefix={}
    with np.load(previous,allow_pickle=False) as before,np.load(first,allow_pickle=False) as after:
        for key in ('policy_observation','critic_observation','pre_root_state','requested_action','emitted_action'):
            np.testing.assert_array_equal(before[key],after[key])
            prefix[key]=0.
    for path in (first,previous,root/'updates.jsonl'):
        evidence[str(path)]=sha256_file(path)
    write_json_atomic(args.output,dict(status='verified',controls=r['ledger']['confirmed'],mean_anchor_calls=calls,
        replay_rows=observed['rows'],draw_sha256=digest.hexdigest(),validation_rows_used_for_updates=0,
        final_checkpoint=r['final_checkpoint'],first_rollout_max_absolute_difference=prefix,
        mean_anchor_loss=dict(first=losses[0],last=losses[-1],mean=float(np.mean(losses)),maximum=max(losses)),
        offline_action_drift_from_initial_actor=offline,evidence=evidence,
        scope='One fixed TRAIN replay auxiliary loss ablation; offline validation only used here for diagnostics, not optimizer updates.'))


if __name__=='__main__':
    main()
