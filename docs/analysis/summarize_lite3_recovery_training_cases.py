"""Diagnose the original training cases without importing validation states."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    manifest=Path('docs/lite3_recovery_bank_20261005.json');meta=json.loads(manifest.read_text())
    bank_path=Path('artifacts/recovery_teacher_20261005/bank-v1-attempt02/bank.npz');assert sha256_file(bank_path)==meta['bank_sha256']
    bank=np.load(bank_path,allow_pickle=False);cases={}
    for i,e in enumerate(meta['episodes']):
        if e['split']!='train':continue
        w,x,y,z=bank['root'][i,3:7]
        axis=np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
        tilt_rate=-np.cross(bank['root'][i,10:13],axis)[2]/np.sqrt(1-axis[2]**2)
        cases[str(i)]=dict(initial_tilt_rate_deg_s=float(np.rad2deg(tilt_rate)),source_episode=e['id'],terrain=e['terrain'],tilt_deg=e['tilt_deg'],
            roll_deg=float(np.rad2deg(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))),
            pitch_deg=float(np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))),angular_velocity_world=bank['root'][i,10:13].tolist())
    results={}
    for run in args.runs:
        root=Path(run);report=json.loads((root/'report.json').read_text());assert report['status']=='complete'
        totals={k:dict(controls=0,success=0,failure=0,deadline=0) for k in cases};hashes={}
        for path in sorted((root/'trajectory').glob('chunk-*.npz')):
            hashes[path.name]=sha256_file(path)
            with np.load(path,allow_pickle=False) as d:
                for k,v in totals.items():
                    mask=d['recovery_case']==int(k);v['controls']+=int(mask.sum())
                    for outcome in ('success','failure','deadline'):v[outcome]+=int((mask&d['task_'+outcome]).sum())
        results[run]=dict(cases=totals,trace_sha256=hashes,report_sha256=sha256_file(root/'report.json'))
    write_json_atomic(args.output,dict(status='complete',training_cases=cases,runs=results,scope='Training-only diagnostics. Training task success is not held-out recovery qualification. Per-contact failure subtypes were not recorded by this trainer; do not infer them.',manifest_sha256=sha256_file(manifest)))


if __name__=='__main__':main()
