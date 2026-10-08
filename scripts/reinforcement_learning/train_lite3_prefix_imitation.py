"""Supervised full-actor distillation of fresh recovery demonstrations, no PPO update."""
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
    for name in ('protocol','output','device'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p=json.loads(Path(args.protocol).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    dataset_report=json.loads(Path(p['dataset_report']).read_text());assert dataset_report['status']=='complete' and dataset_report['protocol_sha256']==p['collection_protocol_sha256']
    assert sha256_file(p['dataset'])==dataset_report['dataset_sha256']
    assert sha256_file(p['dataset_report'])==p['dataset_report_sha256']
    assert sha256_file(p['dataset_audit'])==p['dataset_audit_sha256'] and json.loads(Path(p['dataset_audit']).read_text())['status']=='verified'
    torch.set_num_threads(4);torch.manual_seed(p['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)
    model=FrozenPPOExpert(payload['actor_state_dict']).to(args.device).requires_grad_(True).train()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    data={k:torch.as_tensor(v,device=args.device) for k,v in np.load(p['dataset'],allow_pickle=False).items()}
    generator=torch.Generator(device=args.device).manual_seed(p['seed'])
    optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    groups=p['batch_groups'];report=dict(status='running',protocol_sha256=sha256_file(args.protocol),dataset_sha256=dataset_report['dataset_sha256'],algorithm='supervised full-actor MSE; no PPO/SAC/FlowDAgger updates',updates=0,new_physics=0)
    base_path=Path(__file__).resolve().parents[2]
    report['source_files']={str(f.relative_to(base_path)):sha256_file(f) for folder in (base_path/'source/rl_training/rl_training',base_path/'scripts/reinforcement_learning') for f in folder.rglob('*.py')}
    report['versions']={'torch':torch.__version__,'numpy':np.__version__}
    report['validation_not_used_for_updates']=True
    report['preservation_weight']=p.get('preservation_weight',1.)
    assert report['preservation_weight']>0
    write_json_atomic(out/'report.json',report)
    def losses(split):
        result={}
        with torch.no_grad():
            for group in groups:
                obs=data[group+'_'+split+'_obs'];target=data[group+'_'+split+'_target'];total=0
                for start in range(0,len(obs),4096):total+=float(F.mse_loss(model(obs[start:start+4096]),target[start:start+4096],reduction='sum'))
                result[group]=total/(len(obs)*12)
        return result
    report['before']={split:losses(split) for split in ('train','validation')}
    try:
        for step in range(p['updates']):
            loss=0;parts={}
            for group,n in groups.items():
                obs=data[group+'_train_obs'];target=data[group+'_train_target'];idx=torch.randint(len(obs),(n,),device=args.device,generator=generator)
                part=F.mse_loss(model(obs[idx]),target[idx]);parts[group]=float(part.detach())
                weight=1. if group=='correction' else p.get('preservation_weight',1.)
                loss=loss+(weight*n/sum(groups.values()))*part
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),p['max_grad_norm'],error_if_nonfinite=True);optimizer.step()
            report['updates']=step+1
            if (step+1)%100==0:
                with (out/'updates.jsonl').open('a') as stream:stream.write(json.dumps(dict(update=step+1,loss=float(loss.detach()),groups=parts))+'\n')
        model.eval();report['after']={split:losses(split) for split in ('train','validation')}
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[k])];assert changed and all(torch.isfinite(v).all() for v in model.state_dict().values())
        checkpoint=dict(schema='lite3_prefix_imitation_v1',actor_state_dict={k:v.cpu() for k,v in model.state_dict().items()},training_protocol_sha256=report['protocol_sha256'])
        report['optimizer_steps']=sorted({int(v['step']) for v in optimizer.state.values()})
        assert report['optimizer_steps']==[p['updates']]
        torch.save(checkpoint,out/'final.pt')
        report.update(status='complete',updates=p['updates'],changed_tensors=changed,final_checkpoint=dict(path=str((out/'final.pt').resolve()),sha256=sha256_file(out/'final.pt')))
        assert sha256_file(args.protocol)==report['protocol_sha256'] and sha256_file(p['base_checkpoint'])==p['base_sha256']
        assert sha256_file(p['dataset'])==report['dataset_sha256']
        assert all(sha256_file(base_path/f)==h for f,h in report['source_files'].items())
        write_json_atomic(out/'report.json',report)
    except BaseException as error:report.update(status='failed',error=repr(error));write_json_atomic(out/'report.json',report);raise


if __name__=='__main__':main()
