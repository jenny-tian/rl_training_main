"""Descriptive diagnostics of saved pilot trajectories; no causal claims."""
import sys

import numpy as np
import torch

from reference_online_c1b_reset_api import OUTPUT,DOCS,registration
from reference_online_c1b_recover import recovered_completed
from reference_online_targets import read
from reference_stairs_eval import ROOT,save,sha

sys.path.insert(0,str(ROOT/'source/rl_training'))
from rl_training.ea_lpl.c1b import returns_to_go


def main():
    registration()
    result=dict(simulation_steps_added=0,online_updates_added=0,
        caveat='Descriptive post hoc analysis; does not select a checkpoint or change registered success criteria',runs={})
    for name in ('train','target'):
        directory=OUTPUT/name
        recovered_completed(directory)
        records=read(directory/'episodes.json')['records']
        specs=read(directory/'config.json')['specs']
        with np.load(directory/'trace.npz') as archive:
            data={key:archive[key] for key in ('phase','code','residual_request','learned_effect','paused','decision','features','episode','score')}
        arms={}
        for arm in ('Frozen','Updated'):
            ids=[i for i,s in enumerate(specs) if s['arm']==arm]
            policy=data['phase'][:,ids]==1
            request=data['residual_request'][:,ids][policy]
            effect=data['learned_effect'][:,ids][policy]
            codes=np.tanh(data['code'][:,ids][policy])
            scale=np.array([.125,.25,.25]*4)
            r=[r for r in records if r['arm']==arm]
            arms[arm]=dict(policy_samples=int(policy.sum()),mean_tanh_code=codes.mean(0).tolist(),
                tanh_code_std=codes.std(0).tolist(),mean_abs_requested_action=float(np.abs(request).mean()),
                mean_abs_learned_action=float(np.abs(effect).mean()),
                mean_abs_learned_joint_target_radians=float(np.abs(effect*scale).mean()),
                p95_max_abs_learned_joint_target_radians=float(np.quantile(np.abs(effect*scale).max(-1),.95)),
                paused_fraction=float(data['paused'][:,ids][policy].mean()),
                requested_vs_executed_changed_fraction=float((np.abs(effect-request).max(-1)>1.e-6).mean()),
                outcomes={outcome:dict(n=len(group),mean_discounted_return=float(np.mean([x['discounted_return'] for x in group])),
                    mean_return_sum=float(np.mean([x['return_sum'] for x in group])),mean_policy_steps=float(np.mean([x['policy_steps'] for x in group])))
                    for outcome in dict.fromkeys(x['outcome'] for x in r)
                    if (group:=[x for x in r if x['outcome']==outcome])})
        result['runs'][name]=dict(arms=arms,trace_sha256=sha(directory/'trace.npz'))
    checkpoint=torch.load(OUTPUT/'train/final.pt',map_location='cpu',weights_only=True)
    result['final_updated_parameter_norms']=[dict(session=i,weight_norm=float(checkpoint['heads'][2*i+1]['state_dict']['weight'].norm()),
        bias=checkpoint['heads'][2*i+1]['state_dict']['bias'].tolist()) for i in range(8)]
    result['registered_gamma99_weights']={str(seconds):.99**round(seconds/.02) for seconds in (1,2,5,10,20)}
    result['source_sha256']=sha(ROOT/'scripts/tools/reference_online_c1b_diagnostics.py')
    save(DOCS/'DIAGNOSTICS.json',result)
    print(result['runs']['target']['arms'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
