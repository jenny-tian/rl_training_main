"""Rebuild existing replay row order and attach true past-frame route differences."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();evidence={}
    def read(path):
        evidence[str(path)]=sha256_file(path)
        return json.loads(Path(path).read_text())
    parent=read('docs/lite3_prefix_imitation_plan_20261008.json')
    coverage=read('docs/lite3_retention_coverage_plan_20261008.json')
    oldpath=Path('artifacts/retention_coverage_20261008/dataset/dataset.npz')
    oldaudit=read('docs/lite3_retention_coverage_dataset_audit_20261008.json')
    assert oldaudit['status']=='verified'
    oldreport=read('artifacts/retention_coverage_20261008/dataset/report.json')
    assert sha256_file(oldpath)==oldreport['dataset_sha256']
    evidence[str(oldpath)]=sha256_file(oldpath)
    collections=list(parent['collections'])+[dict(output=coverage['collection_run'],protocol=coverage['collection_protocol'],
        protocol_sha256=coverage['collection_protocol_sha256'],audit=coverage['collection_audit'],phase='course-trial',role='coverage')]
    pools={};provenance={};resets=0;total_rows=0
    for source,spec in enumerate(collections):
        root=Path(spec['output']);report=read(root/'report.json');audit=read(spec['audit']);protocol=read(spec['protocol'])
        assert report['status']=='complete' and audit['status']=='verified'
        assert report['protocol_sha256']==spec['protocol_sha256']==sha256_file(spec['protocol'])
        assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        assert 'train' in protocol['purpose']
        phase=report['phases'][spec['phase']];folder=root/spec['phase']
        with np.load(folder/'initial.npz',allow_pickle=False) as initial:
            terrain=initial['terrain'].copy();case=initial['case'].copy()
        evidence[str(folder/'initial.npz')]=sha256_file(folder/'initial.npz')
        success=np.array([e['success'] for e in phase['episodes']],bool)
        offset=0;previous=None;previous_done=np.ones(len(case),bool)
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            assert sha256_file(path)==audit['phases'][spec['phase']]['trace_sha256'][path.name]
            evidence[str(path)]=sha256_file(path)
            with np.load(path,allow_pickle=False) as d:
                obs=d['policy_observation'];time=np.arange(offset,offset+len(obs))[:,None]
                y=obs[...,9]
                lag=np.concatenate([(y[0] if previous is None else previous)[None],y[:-1]])
                reset=np.concatenate([previous_done[None],d['done'][:-1].astype(bool)])
                lag=np.where(reset,y,lag)
                velocity=np.clip((y-lag)*np.float32(1.65)/np.float32(.02),-1.,1.)
                feature=velocity/np.float32(.3)
                enriched=np.concatenate([obs[...,:12],feature[...,None],obs[...,12:]],axis=-1)
                valid=d['active']&(success[None,:] if spec['role']!='coverage' else True)
                if spec['role']=='coverage':valid&=abs(y*np.float32(1.65))<=.3
                masks=({'correction':valid&(case[None,:]==26)&(time<12),
                        'post_recovery':valid&(case[None,:]==26)&(time>=12)} if spec['role']=='prefix' else
                       {'flat':valid&(terrain[None,:]<2),'rough101':valid&(terrain[None,:]>=2)&(terrain[None,:]<5),'rough107':valid&(terrain[None,:]>=5)})
                for name,mask in masks.items():
                    for split in ('train','validation'):
                        selected=mask&((np.arange(len(case))[None,:]%4==0)==(split=='validation'))
                        key=f'{name}_{split}_obs'
                        pools.setdefault(key,[]).append(enriched[selected].copy())
                        slots=np.broadcast_to(np.arange(len(case)),selected.shape)
                        times=np.broadcast_to(time,selected.shape)
                        provenance.setdefault(key,[]).append(np.stack([np.full(int(selected.sum()),source),times[selected],slots[selected],reset[selected]],axis=-1))
                        resets+=int(reset[selected].sum());total_rows+=int(selected.sum())
                previous=y[-1].copy();previous_done=d['done'][-1].astype(bool).copy();offset+=len(obs)
    arrays={k:np.concatenate(v) for k,v in pools.items()}
    provenance={k:np.concatenate(v) for k,v in provenance.items()}
    with np.load(oldpath,allow_pickle=False) as old:
        for key in old.files:
            if key.endswith('_obs'):
                np.testing.assert_array_equal(np.concatenate([arrays[key][:,:12],arrays[key][:,13:]],-1),old[key])
                assert arrays[key].shape==(len(old[key]),49)
            else:arrays[key]=old[key].copy()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'dataset.npz',**arrays)
    np.savez_compressed(output/'row_provenance.npz',**provenance)
    write_json_atomic(output/'report.json',dict(status='complete',dataset_sha256=sha256_file(output/'dataset.npz'),
        provenance_sha256=sha256_file(output/'row_provenance.npz'),parent_dataset_sha256=sha256_file(oldpath),
        options=dict(dt=.02,lateral_scale_m=1.65,speed_scale_m_s=.3,clip_m_s=1.),
        counts={k:len(v) for k,v in arrays.items() if k.endswith('_obs')},total_rows=total_rows,reset_rows=resets,
        history_summary={k:dict(zero_rows=int((v[:,12]==0).sum()),max_abs_feature=float(abs(v[:,12]).max())) for k,v in arrays.items() if k.endswith('_obs')},
        source_order=[x['output'] for x in collections],new_controls=0,original_observations_and_labels_bitwise_preserved=True,evidence=evidence,
        scope='History computed before row/domain/success filtering and across chunk boundaries; reset rows zero, no cross-episode or future frame. Exact original row order, targets and train/validation streams preserved. Only original training collections; no pilot or original7-source data.'))


if __name__=='__main__':main()
