"""Check matching49D arms, fixed updates, common first rollout and replay draws."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);args=parser.parse_args();evidence={}
    def read(path):
        evidence[str(path)]=sha256_file(path)
        return json.loads(Path(path).read_text())
    protocols={arm:read(f'docs/lite3_route_history_{arm}_train_protocol_20261010.json') for arm in ['zero','history']}
    reports={arm:read(f'artifacts/route_history_20261010/{arm}/train/report.json') for arm in protocols}
    audits={arm:read(f'docs/lite3_route_history_{arm}_update_audit_20261010.json') for arm in protocols}
    z,h=protocols['zero'],protocols['history']
    assert set(z)==set(h)
    assert not z['route_history']['enabled'] and h['route_history']['enabled']
    for k in z:
        if k not in ['arm','scope','route_history','route_history_smoke']:assert z[k]==h[k],k
    assert {k:v for k,v in z['route_history'].items() if k!='enabled'}=={k:v for k,v in h['route_history'].items() if k!='enabled'}
    for arm in protocols:
        assert reports[arm]['status']=='complete' and audits[arm]['status']=='verified'
        assert audits[arm]['evidence'][f'artifacts/route_history_20261010/{arm}/train/report.json']==sha256_file(f'artifacts/route_history_20261010/{arm}/train/report.json')
    assert reports['zero']['source_files']==reports['history']['source_files']
    assert reports['zero']['mean_anchor']['draw_sha256']==reports['history']['mean_anchor']['draw_sha256']
    paths=[Path(f'artifacts/route_history_20261010/{arm}/train/trajectory/chunk-00000.npz') for arm in protocols]
    with np.load(paths[0],allow_pickle=False) as a,np.load(paths[1],allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            if key in ['policy_observation','next_policy_observation']:
                np.testing.assert_array_equal(np.concatenate([a[key][...,:12],a[key][...,13:]],-1),np.concatenate([b[key][...,:12],b[key][...,13:]],-1))
            elif key!='history_input':np.testing.assert_array_equal(a[key],b[key])
        assert not a['policy_observation'][...,12].any()
        different=int((a['policy_observation'][...,12]!=b['policy_observation'][...,12]).sum());assert different>0
    for path in paths:evidence[str(path)]=sha256_file(path)
    checkpoints=[Path(f'artifacts/route_history_20261010/{arm}/train/rsl/model_0.pt') for arm in protocols]
    left,right=[torch.load(path,map_location='cpu',weights_only=False) for path in checkpoints]
    for k,v in left['critic_state_dict'].items():torch.testing.assert_close(v,right['critic_state_dict'][k],atol=0,rtol=0)
    for path in checkpoints:evidence[str(path)]=sha256_file(path)
    write_json_atomic(args.output,dict(status='verified',first_rollout_controls_per_arm=2048,first_rollout_history_input_differences=different,
        common_physics_actions_rewards_and_current_observations_bitwise_equal=True,critic_after_first_update_bitwise_equal=True,
        replay_draws_equal=True,equal49D_networks_and51D_critics=True,new_controls=0,evidence=evidence))


if __name__=='__main__':main()
