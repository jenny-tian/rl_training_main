"""Offline C1-b trace, pairing and checkpoint audit; no new simulation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import sys

import numpy as np
import torch

from reference_online_c1b import completed, protocol
from reference_online_c1b_reset_api import OUTPUT, DOCS, registration
from reference_stairs_eval import ROOT, save, sha
from reference_online_targets import read

sys.path.insert(0,str(ROOT/'source/rl_training'))
from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
from rl_training.ea_lpl.c1b import digest, exploration_table, returns_to_go
from rl_training.online_adaptation.safety import SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def audit(name):
    registration()
    directory = OUTPUT/name
    report = completed(directory)
    episodes = read(directory/'episodes.json')
    specs = read(directory/'config.json')['specs']
    with np.load(directory/'trace.npz') as archive:
        data = {k:torch.from_numpy(archive[k]) for k in archive.files}
    steps,n = data['phase'].shape
    assert steps==report['control_steps'] and n==report['environments']
    assert steps*n==report['interactions']
    counts = {key:(data['phase']==index).sum(0).tolist() for index,key in enumerate(('preparation','policy','waiting'))}
    assert counts == report['counts'] == episodes['counts']
    assert sum(sum(value) for value in counts.values()) == steps*n
    live = CurrentModelSafetyFilter(SafetyLimits(residual_limits=torch.tensor(protocol()['residual_limits'])))
    action_error = memory_error = effect_error = 0.
    for t in range(steps):
        snapshot = live.temporal_snapshot(data['base_action'][t])
        for k,value in snapshot.items():
            memory_error=max(memory_error,float((value.float()-data[f'memory.{k}'][t].float()).abs().max()))
        state = SafetyState(**{f.name:data[f'state.{f.name}'][t] for f in fields(SafetyState)})
        decision = live.project(data['base_action'][t],data['residual_request'][t],state)
        action_error=max(action_error,float((decision.action-data['action'][t]).abs().max()))
        effect_error=max(effect_error,float((live.execution_components()['learned_effect']-data['learned_effect'][t]).abs().max()))
        reset=data['filter_reset_after'][t]
        if reset.any(): live.reset_temporal_state(torch.where(reset)[0])
    assert max(action_error,memory_error,effect_error)<=1.e-6
    # Every explicit reset preserves keys, offsets and actuator lags within a
    # pair. Reusing a key for a second episode in the same arm is forbidden.
    seen=set()
    for reset in episodes['resets']:
        ids=reset['ids']
        assert len(ids)%2==0
        for k in range(0,len(ids),2):
            assert ids[k]%2==0 and ids[k+1]==ids[k]+1
            assert reset['keys'][k]==reset['keys'][k+1]
            assert reset['offsets'][k]==reset['offsets'][k+1]
            assert all(lags[k]==lags[k+1] for lags in reset['delays'])
        for i,key in zip(ids,reset['keys']):
            identity=(i,*key)
            assert identity not in seen
            seen.add(identity)
    initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=True)
    final=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
    for i,spec in enumerate(specs):
        head=final['heads'][i]
        model=torch.nn.Linear(8,2); model.load_state_dict(head['state_dict'])
        assert digest(model)==report['final_head_digests'][i]==head['parameter_sha256']
        assert head['updates']==report['session_updates'][i]
        if spec['arm']=='Frozen' or report['kind']!='train':
            assert head['parameter_sha256']==initial['heads'][i]['parameter_sha256']
    assert sum(h['updates'] for h in final['heads'])==report['adapter_updates']
    noise=exploration_table()
    assert np.array_equal(np.load(directory/'noise_table.npy'),noise)
    # Frozen has zero mean. At evaluation, every arm must use its saved
    # deterministic mean. Updated training means evolve between episodes.
    code_error=0.
    for i,spec in enumerate(specs):
        ticks=torch.where(data['decision'][:,i])[0]
        if report['kind'] in ('target','retention'):
            model=torch.nn.Linear(8,2); model.load_state_dict(final['heads'][i]['state_dict'])
            with torch.no_grad(): expected=model(data['features'][ticks,i])
        elif spec['arm']=='Frozen' or report['kind']=='smoke':
            eps=data['episode'][ticks,i].numpy(); ages=data['age'][ticks,i].numpy()//10
            expected=.6*torch.from_numpy(noise[spec['session'],eps,ages])
        else: continue
        if len(ticks): code_error=max(code_error,float((expected-data['code'][ticks,i]).abs().max()))
    assert code_error<=1.e-6
    return_error=0.
    for record in episodes['records']:
        i,ep=record['env_id'],record['episode']
        mask=(data['phase'][:,i]==1)&(data['episode'][:,i]==ep)
        scores=data['score'][mask,i].numpy()
        assert len(scores)==record['policy_steps']
        return_error=max(return_error,abs(float(returns_to_go(scores)[0])-record['discounted_return']),
                         abs(float(scores.sum())-record['return_sum']))
    assert return_error<1.e-8
    result=dict(run=name,simulation_steps_added=0,online_optimizer_updates_added=0,
        report=report,counts_totals={k:sum(v) for k,v in counts.items()},
        unique_env_reset_keys=len(seen),pairing_verified=True,
        replay=dict(action_max_abs_error=action_error,memory_max_abs_error=memory_error,
                    learned_effect_max_abs_error=effect_error,code_max_abs_error=code_error,return_max_abs_error=return_error),
        outcomes={arm:dict(Counter(r['outcome'] for r in episodes['records'] if r['arm']==arm)) for arm in ('Frozen','Updated')},
        source_sha256=sha(ROOT/'scripts/tools/reference_online_c1b_audit.py'),
        artifacts=[dict(path=str(directory/file),sha256=sha(directory/file)) for file in
            ('execution.json','report.json','episodes.json','trace.npz','initial.pt','final.pt','prepared_states.json')])
    save(DOCS/f'AUDIT_{name}.json',result)
    print(name,result['replay'],result['outcomes'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',choices=('smoke_3','train','target','retention'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    audit(args.run)
