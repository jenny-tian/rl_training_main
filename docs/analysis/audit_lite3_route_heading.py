"""Rebuild the centerline heading target and actual reward from pre-reset roots."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','run','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--training-audit')
    args=parser.parse_args();root=Path(args.run)
    p=json.loads(Path(args.protocol).read_text());r=json.loads((root/'report.json').read_text())
    assert r['status']=='complete' and r['protocol_sha256']==sha256_file(args.protocol)
    evidence={str(path):sha256_file(path) for path in (Path(args.protocol),root/'report.json')}
    if args.training_audit:
        a=json.loads(Path(args.training_audit).read_text())
        assert a['status']=='verified' and a['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        evidence[args.training_audit]=sha256_file(args.training_audit)
    else:
        assert r['route_heading_timing_passed'] and r['route_heading_options']==p['route_training']
    options=p['route_training'];limit=np.deg2rad(options['heading_maximum_deg'])
    assert options['heading_lookahead_m']==1. and options['heading_maximum_deg']==20.
    total=done_rows=reset_differences=saturated=0;old_sum=new_sum=target_sum=0.;hashes={};origins=None
    groups={name:dict(rows=0,old_cost_sum=0.,new_cost_sum=0.) for name in ('ordinary','early','tail')}
    for path in sorted((root/'trajectory').glob('chunk-*.npz')):
        hashes[path.name]=sha256_file(path)
        with np.load(path,allow_pickle=False) as d:
            state=d['heading_root_state'];origin=d['heading_origins'];done=d['done'].astype(bool)
            assert state.shape==(*done.shape,13) and origin.shape==(*done.shape,3)
            assert np.isfinite(state).all() and np.isfinite(origin).all()
            if origins is None:origins=origin[0].copy()
            np.testing.assert_array_equal(origin,np.broadcast_to(origins,origin.shape))
            y=state[...,1]-origin[...,1]
            w,x,qy,z=np.moveaxis(state[...,3:7],-1,0)
            yaw=np.arctan2(2*(w*z+x*qy),1-2*(qy*qy+z*z))
            target=np.clip(np.arctan2(-y,np.full_like(y,options['heading_lookahead_m'])),-limit,limit)
            cost=1-np.cos(yaw-target);old=1-np.cos(yaw)
            np.testing.assert_allclose(d['heading_target'],target,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(d['heading_cost'],cost,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(d['heading_zero_target_cost'],old,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(d['heading_reward'],options['heading_weight']*cost*.02,atol=1e-6,rtol=1e-5)
            next_route=d['next_policy_observation'][...,9:12]
            expected=np.stack((np.clip(y/1.65,-2,2),np.sin(yaw),np.cos(yaw)),axis=-1)
            np.testing.assert_allclose(next_route[~done],expected[~done],atol=2e-6,rtol=1e-5)
            if args.training_audit:
                np.testing.assert_allclose(d['returned_root_state'][~done],state[~done],atol=2e-6,rtol=1e-5)
            reset_differences+=int((np.max(abs(next_route-expected),axis=-1)[done]>1e-3).sum())
            ordinary=d['recovery_case']<0;early=(~ordinary)&(d['episode_steps']<12)
            for name,mask in [('ordinary',ordinary),('early',early),('tail',~ordinary&~early)]:
                g=groups[name];g['rows']+=int(mask.sum());g['old_cost_sum']+=float(old[mask].sum());g['new_cost_sum']+=float(cost[mask].sum())
            total+=done.size;done_rows+=int(done.sum());saturated+=int((abs(target)>=limit-1e-6).sum())
            old_sum+=float(old.sum());new_sum+=float(cost.sum());target_sum+=float(abs(target).sum())
    assert total==r['ledger']['charged']==r['ledger']['confirmed'] and r['ledger']['uncertain_transitions']==0
    assert done_rows>0 and reset_differences>0 and all(g['rows']>0 for g in groups.values())
    assert new_sum!=old_sum and saturated>0
    if not args.training_audit:
        with np.load(root/'initial.npz',allow_pickle=False) as initial:np.testing.assert_array_equal(origins,initial['origins'])
        evidence[str(root/'initial.npz')]=sha256_file(root/'initial.npz')
    write_json_atomic(args.output,dict(status='verified',controls=total,done_rows=done_rows,
        pre_reset_vs_returned_observation_differences=reset_differences,saturated_target_rows=saturated,
        mean_old_zero_heading_cost=old_sum/total,mean_new_heading_cost=new_sum/total,
        mean_absolute_heading_target_deg=float(np.rad2deg(target_sum/total)),groups=groups,trace_sha256=hashes,evidence=evidence,
        scope='Actual reward-time geometry, dt and reset timing verified. Old-target cost is counterfactual on this same rollout, not a separately executed baseline reward or performance claim.'))


if __name__=='__main__':main()
