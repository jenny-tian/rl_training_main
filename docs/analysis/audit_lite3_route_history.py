"""Independently audit causal history at every recorded control, including resets."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','run','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.run)
    p=json.loads(Path(args.protocol).read_text());r=json.loads((root/'report.json').read_text())
    assert r['status']=='complete' and r['protocol_sha256']==sha256_file(args.protocol)
    o=p['route_history'];total=resets=nonzero=0;hashes={};evidence={str(f):sha256_file(f) for f in [Path(args.protocol),root/'report.json']}
    folders=[root/name for name in r['phases']] if 'phases' in r else [root]
    for folder in folders:
        previous=None;last_done=np.ones(p['num_envs'],bool);part={}
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            part[path.name]=sha256_file(path)
            with np.load(path,allow_pickle=False) as d:
                obs=d['policy_observation'];assert obs.shape[-1]==49 and np.isfinite(obs).all()
                y=obs[...,9]
                for i in range(len(obs)):
                    prior=y[i].copy() if previous is None else np.where(last_done,y[i],previous)
                    velocity=np.clip((y[i]-prior)*np.float32(o['lateral_scale_m'])/np.float32(o['dt']),-o['clip_m_s'],o['clip_m_s'])
                    feature=velocity/np.float32(o['speed_scale_m_s']) if o['enabled'] else np.zeros_like(velocity)
                    np.testing.assert_allclose(obs[i,:,12],feature,atol=2e-6,rtol=1e-5)
                    for key,expected in [('history_previous_y',prior),('history_current_y',y[i]),('history_velocity',velocity),('history_input',feature)]:
                        np.testing.assert_allclose(d[key][i],expected,atol=2e-6,rtol=1e-5)
                    np.testing.assert_array_equal(d['history_reset'][i],last_done)
                    resets+=int(last_done.sum());nonzero+=int((abs(feature)>1e-8).sum());total+=len(feature)
                    last_done=d['done'][i].astype(bool);previous=y[i].copy()
                    if 'next_policy_observation' in d:
                        next_y=d['next_policy_observation'][i,:,9]
                        next_v=np.clip((next_y-y[i])*np.float32(o['lateral_scale_m'])/np.float32(o['dt']),-o['clip_m_s'],o['clip_m_s'])
                        next_feature=np.where(last_done,0,next_v/np.float32(o['speed_scale_m_s'])) if o['enabled'] else np.zeros_like(next_v)
                        np.testing.assert_allclose(d['next_policy_observation'][i,:,12],next_feature,atol=2e-6,rtol=1e-5)
        hashes[folder.name if folder!=root else 'root']=part
        if 'phases' in r:
            with np.load(folder/'initial.npz',allow_pickle=False) as initial,np.load(folder/'initial_history.npz',allow_pickle=False) as history:
                np.testing.assert_array_equal(np.concatenate([history['policy_observation'][:,:12],history['policy_observation'][:,13:]],-1),initial['observation'])
                assert not history['policy_observation'][:,12].any() and history['history_reset'].all()
            evidence[str(folder/'initial_history.npz')]=sha256_file(folder/'initial_history.npz')
    assert total==r['ledger']['confirmed']==r['ledger']['charged'] and r['ledger']['uncertain_transitions']==0
    assert resets>=p['num_envs'] and (nonzero>0 if o['enabled'] else nonzero==0)
    write_json_atomic(args.output,dict(status='verified',controls=total,reset_rows=resets,nonzero_history_rows=nonzero,
        options=o,phase_trace_sha256=hashes,evidence=evidence,
        scope='Every recorded pre-action49D history rebuilt from current/prior route observations; first/reset rows zero, no cross-episode difference. Includes inactive evaluation slots; training returned history checked separately. Canonical48D initial state preserved for baseline hashes.'))


if __name__=='__main__':main()
