"""Verify fresh-rollout labels, whole-stream split, and fitted actor metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','dataset','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--training');parser.add_argument('--training-protocol');args=parser.parse_args()
    p=json.loads(Path(args.plan).read_text());folder=Path(args.dataset);report=json.loads((folder/'report.json').read_text());assert report['status']=='complete' and report['protocol_sha256']==sha256_file(args.plan)
    assert report['dataset_sha256']==sha256_file(folder/'dataset.npz');dataset=np.load(folder/'dataset.npz',allow_pickle=False);checked=set()
    controls=0;evidence={args.plan:sha256_file(args.plan),str(folder/'report.json'):sha256_file(folder/'report.json')}
    for spec in p['collections']:
        root=Path(spec['output']);r=json.loads((root/'report.json').read_text());audit=json.loads(Path(spec['audit']).read_text());child=json.loads(Path(spec['protocol']).read_text())
        assert r['status']=='complete' and audit['status']=='verified' and child['purpose']=='fresh_training_demonstration'
        assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')==report['evidence'][str(root/'report.json')]
        assert r['protocol_sha256']==spec['protocol_sha256']==sha256_file(spec['protocol'])
        controls+=r['ledger']['confirmed'];phase=r['phases'][spec['phase']];path=root/spec['phase'];initial=np.load(path/'initial.npz')
        assert [e['env_id'] for e in phase['episodes']]==list(range(64))
        success=np.array([e['success'] for e in phase['episodes']]);a={}
        for file in sorted((path/'trajectory').glob('chunk-*.npz')):
            assert sha256_file(file)==report['evidence'][str(file)]
            with np.load(file) as d:
                for k in ('policy_observation','emitted_action','active'):a.setdefault(k,[]).append(d[k])
        a={k:np.concatenate(v) for k,v in a.items()};width=len(success);time=np.arange(len(a['active']))[:,None];valid=a['active']&success[None,:]
        if spec['role']=='prefix':
            assert child['recovery_resets']['fixed_train_case']==26 and child['recovery_resets']['split']=='train'
            with np.load(child['recovery_resets']['bank_path']) as bank:assert not bank['validation'][26]
            states=initial['case'][None,:]==26;masks={'correction':valid&states&(time<12),'post_recovery':valid&states&(time>=12)}
        else:
            assert child['conditions']==['course'] and 'action_prefix' not in child
            types=initial['terrain'][None,:];masks={'flat':valid&(types<2),'rough101':valid&(types>=2)&(types<5),'rough107':valid&(types>=5)}
        for group,mask in masks.items():
            for split in ('train','validation'):
                selected=mask&((np.arange(width)[None,:]%4==0)==(split=='validation'))
                for suffix,k in (('obs','policy_observation'),('target','emitted_action')):
                    key=f'{group}_{split}_{suffix}';np.testing.assert_array_equal(dataset[key],a[k][selected]);checked.add(key)
    assert checked==set(dataset.files) and controls<=p['max_collection_controls']
    result=dict(status='verified',collection_controls=controls,dataset_counts={k:len(dataset[k]) for k in dataset.files if k.endswith('_obs')},new_physics=0,evidence=evidence)
    if args.training:
        root=Path(args.training);r=json.loads((root/'report.json').read_text());t=json.loads(Path(args.training_protocol).read_text())
        assert r['status']=='complete' and r['updates']==p['updates']==t['updates']
        assert r['optimizer_steps']==[t['updates']] and r.get('preservation_weight',1.)==t.get('preservation_weight',1.)
        assert r['protocol_sha256']==sha256_file(args.training_protocol) and r['dataset_sha256']==report['dataset_sha256']
        assert sha256_file(root/'final.pt')==r['final_checkpoint']['sha256'] and sha256_file(t['base_checkpoint'])==t['base_sha256']
        assert t['batch_groups']==p['batch_groups'] and t['seed']==p['learner_seed'] and t['learning_rate']==p['learning_rate']
        torch.set_num_threads(4)
        base=FrozenPPOExpert(torch.load(t['base_checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict'])
        model=FrozenPPOExpert(torch.load(root/'final.pt',map_location='cpu',weights_only=False)['actor_state_dict'])
        for label,actor in (('before',base),('after',model)):
            with torch.no_grad():
                for split in ('train','validation'):
                    for group in p['batch_groups']:
                        x=dataset[f'{group}_{split}_obs'];y=dataset[f'{group}_{split}_target'];total=0
                        for start in range(0,len(x),4096):total+=float(F.mse_loss(actor(torch.from_numpy(x[start:start+4096])),torch.from_numpy(y[start:start+4096]),reduction='sum'))
                        np.testing.assert_allclose(total/(len(x)*12),r[label][split][group],atol=1e-6,rtol=2e-4)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,base.state_dict()[k])];assert changed==r['changed_tensors']
        result.update(preservation_weight=t.get('preservation_weight',1.),training_updates=r['updates'],changed_tensors=changed,final_checkpoint=r['final_checkpoint'],before=r['before'],after=r['after'])
        result['evidence'][str(root/'report.json')]=sha256_file(root/'report.json');result['evidence'][args.training_protocol]=sha256_file(args.training_protocol)
    result['scope']='Supervised full-mean-actor fit only; offline validation streams are not held-out source cases. Closed-loop performance must be evaluated separately.'
    write_json_atomic(args.output,result)


if __name__=='__main__':main()
