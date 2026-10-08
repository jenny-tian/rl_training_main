"""Build fresh TRAIN-rollout imitation data; never consume confirmation rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    p=json.loads(Path(args.protocol).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    pools={};counts={};evidence={}
    for spec in p['collections']:
        root=Path(spec['output']);report=json.loads((root/'report.json').read_text());audit=json.loads(Path(spec['audit']).read_text())
        assert report['status']=='complete' and audit['status']=='verified'
        assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        assert report['protocol_sha256']==spec['protocol_sha256']
        evidence[str(root/'report.json')]=sha256_file(root/'report.json');evidence[spec['audit']]=sha256_file(spec['audit'])
        phase=report['phases'][spec['phase']];folder=root/spec['phase'];initial=np.load(folder/'initial.npz')
        success=np.array([e['success'] for e in phase['episodes']],bool);terrain=initial['terrain'];case=initial['case'];offset=0
        counts[spec['role']]=dict(completed_episodes=int(success.sum()),failed_or_cutoff_episodes=int((~success).sum()))
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            evidence[str(path)]=sha256_file(path)
            with np.load(path) as d:
                n=len(d['active']);time=np.arange(offset,offset+n)[:,None];valid=d['active']&success[None,:]
                masks=({'correction':valid&(case[None,:]==26)&(time<12),'post_recovery':valid&(case[None,:]==26)&(time>=12)} if spec['role']=='prefix' else
                    {'flat':valid&(terrain[None,:]<2),'rough101':valid&(terrain[None,:]>=2)&(terrain[None,:]<5),'rough107':valid&(terrain[None,:]>=5)})
                for name,mask in masks.items():
                    for split in ('train','validation'):
                        # Hold out entire environment streams before fitting; not a source-state holdout.
                        selected=mask&((np.arange(len(success))[None,:]%4==0)==(split=='validation'))
                        key=f'{name}_{split}';obs=d['policy_observation'][selected];target=d['emitted_action'][selected]
                        pools.setdefault(key+'_obs',[]).append(obs);pools.setdefault(key+'_target',[]).append(target)
                offset+=n
    arrays={k:np.concatenate(v) for k,v in pools.items()}
    for k,v in arrays.items():assert len(v)>0 and np.isfinite(v).all(),k
    np.savez_compressed(out/'dataset.npz',**arrays)
    write_json_atomic(out/'report.json',dict(status='complete',protocol_sha256=sha256_file(args.protocol),dataset_sha256=sha256_file(out/'dataset.npz'),counts={k:len(v) for k,v in arrays.items() if k.endswith('_obs')},episodes=counts,evidence=evidence,
        scope='Only preregistered fresh training collections. Full successful first episodes only; stream env_id%4==0 withheld from fitting. These validation rows are same TRAIN case26/terrain, not the original7 held-out source states.'))


if __name__=='__main__':main()
