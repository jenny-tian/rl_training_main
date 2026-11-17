"""CPU replay of C1-a actions, history, scores, learning and exact-budget snapshots."""
from __future__ import annotations

import argparse
from dataclasses import fields
import sys

import numpy as np
import torch

from reference_online_c1a import OUTPUT,DOCS,MARKS,completed,protocol,source_manifest,jobs
from reference_online_targets import read
from reference_stairs_eval import ROOT,save,sha

sys.path.insert(0,str(ROOT/'source/rl_training'))
from rl_training.ea_lpl.c1a import Session,exploration_table,episode_return
from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.current_safety import CurrentModelSafetyFilter
from rl_training.online_adaptation.safety import SafetyLimits
from rl_training.online_adaptation.types import SafetyState


def exact(left,right):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and torch.equal(left,right)
    elif isinstance(left,dict):
        assert set(left)==set(right)
        for key in left: exact(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right): exact(a,b)
    else: assert left==right,(left,right)


def audit(name):
    assert source_manifest()==read(OUTPUT/'lock.json'),'registered sources changed'
    report = completed(name)
    directory = OUTPUT/name
    log,config = read(directory/'episodes.json'),read(directory/'config.json')
    specs = config['specs']; n = len(specs)
    records = {(r['control_step'],r['env_id']):r for r in log['records']}
    assert len(records)==len(log['records'])
    initial = torch.load(directory/'initial.pt',map_location='cpu',weights_only=True)
    final = torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
    heads = [Session(report['kind']=='train' and s['arm']=='Updated') for s in specs]
    if report['kind'] in ('target','retention'):
        job = next(j for j in jobs() if j['name']==name)
        path = OUTPUT/'train'/('initial.pt' if job['mark']==0 else f'at_{job["mark"]}.pt')
        source = torch.load(path,map_location='cpu',weights_only=True)
        assert source['control_steps']==job['mark']
        for spec,snapshot in zip(specs,initial['heads']):
            expected = source['heads'][2*spec['session']+int(spec['arm']=='Updated')]
            exact(snapshot['b'],expected['b'])
            assert snapshot['parameter_sha256']==expected['parameter_sha256']
        for head,snapshot in zip(heads,initial['heads']): head.load_evaluation_mean(snapshot)
    for h,snapshot in zip(heads,initial['heads']): exact(h.snapshot(),snapshot)
    noise = exploration_table()
    assert np.array_equal(noise,np.load(directory/'noise_table.npy'))
    live = CurrentModelSafetyFilter(SafetyLimits(residual_limits=torch.tensor(protocol()['residual_limits'])))
    code = np.zeros((n,2)); starts = np.zeros(n); ages = np.zeros(n,int); velocity = np.zeros(n)
    counts = np.zeros((3,n),int)
    errors = dict(action=0.,memory=0.,effect=0.,request=0.,code=0.,score=0.,velocity=0.)
    samples, changed, paused = (np.zeros(n,int) for _ in range(3))
    index = read(directory/'trace_index.json')
    tick = found = 0
    for chunk in index['chunks']:
        path = directory/chunk['file']
        assert sha(path)==chunk['sha256'] and chunk['start']==tick
        with np.load(path) as archive: arrays = {k:archive[k] for k in archive.files}
        assert len(arrays['phase'])==chunk['stop']-chunk['start']
        data = {k:torch.from_numpy(v) for k,v in arrays.items()}
        for t in range(len(arrays['phase'])):
            tick += 1
            phase, decision = arrays['phase'][t],arrays['decision'][t]
            assert np.array_equal(decision,(phase==1)&(arrays['age'][t]==0))
            for k in range(3): counts[k] += phase==k
            for i in np.flatnonzero(decision):
                eps = noise[specs[i]['session'],arrays['episode'][t,i]] if report['kind'] in ('smoke','train') else None
                code[i] = heads[i].begin(eps); starts[i] = arrays['before_root'][t,i,0]
                ages[i] = 0; velocity[i] = 0
            code[phase!=1] = 0
            errors['code'] = max(errors['code'],float(np.abs(code-arrays['code'][t]).max()))
            request = torch.einsum('jk,nk->nj',basis('sag'),torch.tanh(torch.tensor(code,dtype=torch.float32)))
            request[phase!=1] = 0
            errors['request'] = max(errors['request'],float((request-data['residual_request'][t]).abs().max()))
            for key,value in live.temporal_snapshot(data['base_action'][t]).items():
                errors['memory'] = max(errors['memory'],float((value.float()-data[f'memory.{key}'][t].float()).abs().max()))
            state = SafetyState(**{f.name:data[f'state.{f.name}'][t] for f in fields(SafetyState)})
            executed = live.project(data['base_action'][t],request,state)
            errors['action'] = max(errors['action'],float((executed.action-data['action'][t]).abs().max()))
            errors['effect'] = max(errors['effect'],float((live.execution_components()['learned_effect']-data['learned_effect'][t]).abs().max()))
            assert not np.count_nonzero(arrays['action'][t,phase!=1])
            if report['kind'] in ('target','retention'): assert not np.count_nonzero(arrays['learned_effect'][t,::2])
            eligible = (phase==1)&(arrays['age'][t]>=10)
            samples += eligible
            changed += eligible & (np.abs(arrays['learned_effect'][t]-arrays['residual_request'][t]).max(-1)>1.e-6)
            paused += eligible & arrays['paused'][t]
            ages += phase==1
            velocity += (phase==1)*((arrays['body_velocity'][t]-arrays['command'][t,:,:2])**2).sum(-1)
            for i in range(n):
                record = records.get((tick,i))
                if record is None:
                    assert arrays['terminal_return'][t,i]==0.
                    continue
                found += 1
                assert phase[i]==1 and ages[i]==record['policy_steps']
                assert arrays['episode'][t,i]==record['episode']
                displacement = float(arrays['root'][t,i,0]-starts[i])
                value = episode_return(displacement,record['outcome']=='success',record['outcome']=='posture_failure')
                errors['score'] = max(errors['score'],abs(value-record['episode_return']),abs(value-arrays['terminal_return'][t,i]))
                errors['velocity'] = max(errors['velocity'],abs(velocity[i]-record['xy_velocity_error_sum']))
                stats = heads[i].finish(value,learn=report['kind']=='train')
                for key,value in stats.items(): exact(value,record[key])
            reset = data['filter_reset_after'][t]
            if reset.any(): live.reset_temporal_state(torch.where(reset)[0])
            if report['kind']=='train' and tick in MARKS:
                snapshot = torch.load(directory/f'at_{tick}.pt',map_location='cpu',weights_only=True)
                assert snapshot['control_steps']==tick
                for h,saved in zip(heads,snapshot['heads']): exact(h.snapshot(),saved)
    assert found==len(records) and tick==report['control_steps']==index['control_steps']
    assert tick*n==report['interactions']
    assert {key:counts[i].tolist() for i,key in enumerate(('preparation','policy','waiting'))}==report['counts']==log['counts']
    assert max(errors.values())<=1.e-6,errors
    for h,saved in zip(heads,final['heads']): exact(h.snapshot(),saved)
    assert sum(h.updates for h in heads)==report['adapter_updates']
    seen = set()
    for reset in log['resets']:
        ids = reset['ids']
        for k in range(0,len(ids),2):
            assert ids[k]%2==0 and ids[k+1]==ids[k]+1
            assert reset['keys'][k]==reset['keys'][k+1]
            assert reset['offsets'][k]==reset['offsets'][k+1]
            assert all(lags[k]==lags[k+1] for lags in reset['delays'])
        for i,key in zip(ids,reset['keys']):
            assert key[:3]==[20261117,protocol()['namespaces'][report['kind']],specs[i]['slot']]
            assert (i,*key) not in seen
            seen.add((i,*key))
    assert report['paired_max_error']<=1.e-6
    for prepared in read(directory/'prepared_states.json'):
        ids = prepared['ids']
        assert len(ids)%2==0
        for key in ('root','joint_pos','joint_vel','observation'):
            values = np.asarray(prepared[key])
            assert len(values)==len(ids) and np.abs(values[::2]-values[1::2]).max()<=1.e-6
    result = dict(status='passed',name=name,report_sha256=sha(directory/'report.json'),
        trace_index_sha256=sha(directory/'trace_index.json'),final_sha256=sha(directory/'final.pt'),
        replay_errors=errors,parameters_adam_batches_and_snapshots_exact=True,
        counts={key:int(counts[i].sum()) for i,key in enumerate(('preparation','policy','waiting'))},
        ea_diagnostic=[dict(session=s['session'],arm=s['arm'],samples=int(samples[i]),
            changed=int(changed[i]),paused=int(paused[i]),
            p_changed=float(changed[i]/samples[i]) if samples[i] else None,
            p_paused=float(paused[i]/samples[i]) if samples[i] else None) for i,s in enumerate(specs)],
        simulation_steps_added=0,online_updates_added=0)
    save(DOCS/f'AUDIT_{name}.json',result)
    print(name,'CPU audit passed',errors,flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--job',required=True)
    torch.set_num_threads(1)
    audit(parser.parse_args().job)
