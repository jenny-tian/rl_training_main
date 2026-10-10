"""Ridge diagnostics on TRAIN streams; neither policy fitting nor proof of observability."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','protocol','training-audit','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    root = Path(args.run)
    p = json.loads(Path(args.protocol).read_text())
    audit = json.loads(Path(args.training_audit).read_text())
    report = json.loads((root/'report.json').read_text())
    assert audit['status']=='verified' and report['status']=='complete'
    assert report['protocol_sha256']==sha256_file(args.protocol)
    assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
    assert p['recovery_resets']['split']=='train' and audit['exposed_training_cases']==[26] and audit['validation_cases_used']==0
    pieces=[];previous=None
    for path in sorted((root/'trajectory').glob('chunk-*.npz')):
        assert sha256_file(path)==audit['trace_sha256'][path.name]
        with np.load(path,allow_pickle=False) as data:
            obs=data['policy_observation']
            lag=np.concatenate([(previous if previous is not None else obs[0])[None],obs[:-1]])
            delta=(obs-lag)/.02
            slots=np.broadcast_to(np.arange(p['num_envs']),data['recovery_case'].shape)
            mask=(data['recovery_case']==26)&(data['episode_steps']>0)
            pieces.append((obs[mask],delta[mask],data['pre_root_state'][...,8][mask],slots[mask],data['episode_steps'][mask]))
            previous=obs[-1].copy()
    obs,delta,target,slots,ages=[np.concatenate([piece[i] for piece in pieces]) for i in range(5)]
    fit=slots%4!=0;probe=~fit
    assert fit.any() and probe.any() and len(target)==audit['recovery_controls']-audit['recovery_resets']
    predictors={'current48':obs,'current48_plus_one_step_delta48':np.concatenate([obs,delta],-1)}
    predictions={'fit_mean_constant':np.full(probe.sum(),target[fit].mean()),
                 'route_origin_finite_difference':delta[probe,9]*1.65}
    for name,x in predictors.items():
        mu=x[fit].mean(0,dtype=np.float64);std=x[fit].std(0,dtype=np.float64);std[std<1e-5]=1.
        z=np.column_stack([(x-mu)/std,np.ones(len(x))])
        penalty=np.eye(z.shape[1])*1e-3;penalty[-1,-1]=0
        beta=np.linalg.solve(z[fit].T@z[fit]/fit.sum()+penalty,z[fit].T@target[fit]/fit.sum())
        predictions[name]=z[probe]@beta
    results={}
    for name,predicted in predictions.items():
        error=predicted-target[probe]
        results[name]=dict(rmse_m_s=float(np.sqrt(np.mean(error**2))),mae_m_s=float(np.mean(abs(error))),
            r_squared=float(1-np.sum(error**2)/np.sum((target[probe]-target[probe].mean())**2)),by_age={})
        for lo,hi in [(1,12),(12,50),(50,100),(100,200),(200,None)]:
            mask=ages[probe]>=lo
            if hi is not None:mask&=ages[probe]<hi
            results[name]['by_age'][str(lo)]=dict(rows=int(mask.sum()),rmse_m_s=float(np.sqrt(np.mean(error[mask]**2))))
    paths=[Path(args.protocol),Path(args.training_audit),root/'report.json']
    write_json_atomic(args.output,dict(status='verified',rows=len(target),fit_rows=int(fit.sum()),probe_rows=int(probe.sum()),
        fit_env_ids=np.unique(slots[fit]).tolist(),probe_env_ids=np.unique(slots[probe]).tolist(),results=results,
        new_controls=0,policy_optimizer_steps=0,policy_training_labels_created=0,
        scope='All changing-policy TRAIN case26 controls after reset age0 excluded. Scalar root-link COM world-vy diagnostic, fixed ridge1e-3; no hyperparameter sweep. Split whole TRAIN env streams by env_id%4, not independent source cases or held-out benchmark. One past frame only, no future features; fit-only scaling. Low-capacity prediction probe cannot establish that Actor cannot infer velocity. Origin finite difference is a different point from COM and need not match exactly. No diagnostic regressor exported to policy.',
        evidence={str(x):sha256_file(x) for x in paths}))


if __name__=='__main__':
    main()
