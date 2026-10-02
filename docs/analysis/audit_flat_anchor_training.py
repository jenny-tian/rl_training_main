"""Verify every flat target against the original checkpoint and every recorded anchor cost."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from docs.analysis.audit_sac_huber_coverage import read
from rl_training.lite3_joint_conventions import LITE3_SAFE_ACTION_LOWER, LITE3_SAFE_ACTION_UPPER
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def check_cost(action, target, mask, cost):
    assert action.shape==target.shape and action.shape[-1]==12
    assert mask.shape==cost.shape==action.shape[:-1] and mask.dtype==np.bool_
    assert all(np.isfinite(x).all() for x in (action,target,cost))
    expected=np.where(mask,np.square(action.astype(float)-target.astype(float)).mean(-1),0.)
    np.testing.assert_allclose(cost,expected,atol=1e-6,rtol=1e-6)
    assert np.count_nonzero(target[~mask])==0 and np.count_nonzero(cost[~mask])==0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    p=read(args.protocol);root=Path(p['output'])/'train';train=read(root/'report.json');trace=read(root/'trajectory/manifest.json')
    assert train['status']==trace['status']=='complete' and train['flat_anchor_model_unchanged']
    assert train['protocol_sha256']==sha256_file(args.protocol) and train['flat_anchor_weight']==p['flat_anchor_weight']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(p['base_checkpoint'],map_location=args.device,weights_only=False)['actor_state_dict']
    lower=torch.tensor(LITE3_SAFE_ACTION_LOWER,device=args.device);upper=torch.tensor(LITE3_SAFE_ACTION_UPPER,device=args.device)
    report={'status':'running','training_report_sha256':sha256_file(root/'report.json'),'protocol_sha256':sha256_file(args.protocol),
            'rows':0,'flat_target_rows':0,'max_target_error':0.,'chunk_hashes':{},'additional_environment_controls':0}
    for chunk in trace['chunks']:
        path=root/'trajectory'/chunk['file'];report['chunk_hashes'][str(path)]=sha256_file(path)
        with np.load(path) as data:
            obs=data['policy_observation'];target=data['flat_anchor_target'];mask=data['flat_anchor_mask'];cost=data['flat_anchor_cost']
            assert obs.shape==(chunk['steps'],p['num_envs'],48)
            assert p['num_envs']%8==0
            expected=np.broadcast_to(np.arange(p['num_envs'])<p['num_envs']//4,mask.shape)
            np.testing.assert_array_equal(mask,expected)
            check_cost(data['emitted_action'],target,mask,cost)
            selected=obs[mask];proprio=np.concatenate((selected[:,:9],selected[:,12:]),-1)
            with torch.no_grad():
                value=torch.as_tensor(proprio,device=args.device)
                for index in (0,2,4,6):
                    value=F.linear(value,state[f'mlp.{index}.weight'],state[f'mlp.{index}.bias'])
                    if index!=6:value=F.elu(value)
                value=torch.maximum(torch.minimum(value,upper),lower)
                recorded=torch.as_tensor(target[mask],device=args.device)
                torch.testing.assert_close(value,recorded,atol=2e-5,rtol=1e-5)
                report['max_target_error']=max(report['max_target_error'],float((value-recorded).abs().max()))
            report['rows']+=mask.size;report['flat_target_rows']+=int(mask.sum())
    assert report['rows']==train['ledger']['confirmed']==p['training_control_transitions']
    assert report['flat_target_rows']==train['flat_anchor_label_rows']==report['rows']//4
    assert sum(c['steps'] for c in trace['chunks'])==trace['steps']
    report['status']='complete';write_json_atomic(Path(args.output),report)
    print({k:v for k,v in report.items() if k!='chunk_hashes'})


if __name__=='__main__':main()
