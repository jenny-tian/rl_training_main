"""Reconstruct route geometry on existing fresh TRAIN collections only."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    plan_path=Path('docs/lite3_prefix_imitation_plan_20261008.json')
    plan=json.loads(plan_path.read_text());results={};evidence={str(plan_path):sha256_file(plan_path)}
    for spec in plan['collections']:
        root=Path(spec['output']);phase=spec['phase'];folder=root/phase
        report=json.loads((root/'report.json').read_text());audit=json.loads(Path(spec['audit']).read_text())
        assert report['status']=='complete' and audit['status']=='verified'
        assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        assert report['protocol_sha256']==spec['protocol_sha256']==sha256_file(spec['protocol'])
        for path in (root/'report.json',Path(spec['audit']),Path(spec['protocol'])):evidence[str(path)]=sha256_file(path)
        with np.load(folder/'initial.npz',allow_pickle=False) as initial:
            origins=initial['origins'].copy();cases=initial['case'].copy()
        keep=np.arange(len(origins))%4!=0
        if spec['role']=='prefix':keep&=cases==26
        values={key:[] for key in ('body','feet','yaw','vy','current_lateral_cost','body_only_cost','current_heading_cost','lookahead_heading_cost')}
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            assert sha256_file(path)==audit['phases'][phase]['trace_sha256'][path.name]
            with np.load(path,allow_pickle=False) as d:
                active=d['active']&keep[None,:]
                root_state=d['post_root_state'];body=root_state[...,1]-origins[None,:,1]
                feet=np.abs(d['foot_position_w'][...,1]-origins[None,:,None,1]).max(-1)
                w,x,qy,z=np.moveaxis(root_state[...,3:7],-1,0)
                yaw=np.arctan2(2*(w*z+x*qy),1-2*(qy*qy+z*z))
                target=np.clip(np.arctan2(-body,np.ones_like(body)),-np.deg2rad(20),np.deg2rad(20))
                row=dict(body=body,feet=feet,yaw=yaw,vy=root_state[...,8],current_lateral_cost=np.maximum(np.abs(body),feet)**2,
                    body_only_cost=body**2,current_heading_cost=1-np.cos(yaw),lookahead_heading_cost=1-np.cos(yaw-target))
                for key,value in row.items():values[key].append(value[active])
        values={key:np.concatenate(v) for key,v in values.items()}
        mask=abs(values['body'])>.05
        results[spec['role']]=dict(rows=len(values['body']),mean_abs_y_m=float(abs(values['body']).mean()),
            foot_dominant_fraction=float((values['feet']>abs(values['body'])).mean()),
            mean_costs={k:float(values[k].mean()) for k in ('current_lateral_cost','body_only_cost','current_heading_cost','lookahead_heading_cost')},
            offroute_rows=int(mask.sum()),offroute_yaw_outward_fraction=float((values['body'][mask]*np.sin(values['yaw'][mask])>0).mean()),
            offroute_velocity_outward_fraction=float((values['body'][mask]*values['vy'][mask]>0).mean()))
    write_json_atomic(args.output,dict(status='verified',results=results,new_controls=0,optimizer_steps=0,
        lookahead_m=1.,maximum_heading_deg=20.,validation_streams_used=0,evidence=evidence,
        scope='Post-control geometry from existing fresh TRAIN collections; env_id%4==0 withheld. Alternative heading cost is a counterfactual calculation, not a rollout outcome. No reward coefficients or acceptance gates changed here.'))


if __name__=='__main__':main()
