"""Read-only TRAIN coverage and auxiliary-gradient diagnosis, with no optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    plan_path=Path('docs/lite3_demo_target_plan_20261009.json')
    plan=json.loads(plan_path.read_text())
    spec=plan['mean_anchor']
    assert sha256_file(spec['dataset'])==spec['dataset_sha256']
    assert sha256_file(spec['dataset_audit'])==spec['dataset_audit_sha256']
    checkpoints=dict(initial=plan['base_checkpoint'],
        self_mean='artifacts/mean_anchor_long_20261009/train/final.pt',
        demonstration='artifacts/demo_target_20261009/train-retry01/final.pt')
    reference=FrozenPPOExpert(torch.load(checkpoints['initial'],map_location='cpu',weights_only=False)['actor_state_dict'])
    groups=('correction','flat','rough101','rough107','post_recovery')
    states,targets,coverage={},{},{}
    with np.load(spec['dataset'],allow_pickle=False) as dataset:
        for group in groups:
            x=dataset[group+'_train_obs']
            assert np.isfinite(x).all() and x.shape[1]==48
            states[group]=torch.from_numpy(x.copy())
            with torch.no_grad():
                targets[group]=(torch.from_numpy(dataset[group+'_train_target'].copy()) if group=='correction' else reference(states[group]))
            # sim-localization-v1 encodes lateral error divided by1.65m.
            lateral=x[:,9]*1.65
            coverage[group]=dict(train_rows=len(x),lateral_m=dict(min=float(lateral.min()),max=float(lateral.max()),
                p05=float(np.quantile(lateral,.05)),median=float(np.median(lateral)),p95=float(np.quantile(lateral,.95))))
    outcomes={}
    for name,path in checkpoints.items():
        before=sha256_file(path)
        actor=FrozenPPOExpert(torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']).requires_grad_(True)
        parameters=list(actor.parameters())
        gradients,losses={},{}
        for group in groups:
            actor.zero_grad(set_to_none=True)
            total=0.
            x,y=states[group],targets[group]
            for start in range(0,len(x),1024):
                loss=(actor(x[start:start+1024])-y[start:start+1024]).square().sum()/y.numel()
                total+=float(loss.detach())
                loss.backward()
            gradients[group]=torch.cat([p.grad.flatten() for p in parameters]).double()
            losses[group]=total
        ordinary=sum(gradients[g] for g in ('flat','rough101','rough107'))
        gradients['old_combined_auxiliary']=25.*(gradients['correction']+ordinary)
        gradients['proposed_tail_auxiliary']=25.*gradients['post_recovery']
        angles={}
        for other in ('correction','flat','rough101','rough107','old_combined_auxiliary'):
            a,b=gradients['post_recovery'],gradients[other]
            norm=float(a.norm()*b.norm())
            angles[other]=None if norm<1e-15 else float(torch.dot(a,b)/norm)
        outcomes[name]=dict(mse=losses,gradient_norm={g:float(v.norm()) for g,v in gradients.items()},
            cosine_with_tail_preservation=angles)
        assert sha256_file(path)==before
    write_json_atomic(args.output,dict(status='verified',coverage=coverage,models=outcomes,
        new_controls=0,optimizer_steps=0,validation_rows_read=0,
        evidence={str(path):sha256_file(path) for path in [plan_path,spec['dataset'],spec['dataset_audit'],*checkpoints.values()]},
        scope='Full TRAIN arrays only. Gradients are local auxiliary MSE derivatives, not PPO gradients or proof of closed-loop benefit. Negative cosine indicates local conflict. Frozen checkpoint files unchanged.'))


if __name__=='__main__':main()
