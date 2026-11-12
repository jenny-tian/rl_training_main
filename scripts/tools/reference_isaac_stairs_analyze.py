"""Summarize pre-reset frozen-policy trajectories; never infer causality from rewards."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from reference_stairs_eval import save, sha


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def analyze(directory):
    report=json.loads((directory/'report.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    execution=json.loads((directory/'execution.json').read_text())
    assert report['status']=='passed' and execution['status']=='completed' and not report['probe']
    with np.load(directory/'trace.npz') as archive:
        data={key:archive[key] for key in archive.files}
    active=data['active']
    assert active.shape==(report['steps'],report['episodes'])
    times=(np.arange(report['steps'])+1)*report['step_dt']
    reward_names=report['reward_names']
    episodes, windows = [], []
    for row in report['records']:
        i=row['env_id']
        valid=active[:,i].copy()
        assert abs(valid.sum()*report['step_dt']-row['duration_s'])<1e-5
        assert np.all(valid[:valid.sum()]) and not np.any(valid[valid.sum():])
        geom=config['geometry'][row['column']]
        valid &= times>=1.
        feet=data['feet'][:,i]
        root=data['root'][:,i]
        if geom['fronts']:
            front=geom['fronts'][0]
            at_first=(feet[:,:2,0].max(-1)>front-.07)&(root[:,0]<front+.08)&valid
            approach=valid&(root[:,0]<front-.35)
        else:
            at_first=np.zeros_like(valid)
            approach=valid
        rates=data['reward_rates'][:,i]
        final_mask=valid&(times>=row['duration_s']-5.)
        last_progress=float(root[final_mask,0][-1]-root[final_mask,0][0]) if final_mask.any() else None
        forward=float(data['velocity'][valid,i,0].mean()) if valid.any() else None
        episodes.append(dict(profile=report['profile'],direction=report.get('direction','up'),**row,
            mean_forward_velocity=forward,last5_progress_m=last_progress,
            first_face_seconds=float(at_first.sum()*report['step_dt']),
            front_feet_z_p95_at_first=float(np.percentile(feet[at_first,:2,2],95)) if at_first.any() else None,
            front_contact_horizontal_force_p95=float(np.percentile(np.linalg.norm(data['forces'][at_first,i,:2,:2],axis=-1),95)) if at_first.any() else None,
            max_action_abs=float(np.abs(data['action'][active[:,i],i]).max()),
            return_rate=float(rates[valid].sum(-1).mean()) if valid.any() else None))
        for name, mask in (('after_ramp',valid),('approach',approach),('first_face',at_first),('last5',final_mask)):
            if not mask.any(): continue
            values=rates[mask].mean(0)
            windows.append(dict(profile=report['profile'],direction=report.get('direction','up'),env_id=i,height=row['height'],speed=row['speed'],
                repeat=row['repeat'],outcome=row['outcome'],window=name,samples=int(mask.sum()),
                mean_forward_velocity=float(data['velocity'][mask,i,0].mean()),
                mean_base_above_scan=float((root[mask,2]-data['scan_mean'][mask,i]).mean()),
                mean_tilt_deg=float(np.degrees(np.arccos(np.clip(-data['gravity'][mask,i,2],-1,1))).mean()),
                **{f'reward_{key}':float(v) for key,v in zip(reward_names,values)},
                total_rate=float(values.sum())))
    # Hash actual files, including original source/config/complete trajectory data.
    manifest=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(directory.iterdir()) if p.is_file()]
    return report,config,data,episodes,windows,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    all_episodes,all_windows,manifest,groups=[],[],[],[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for directory in args.runs:
        report,config,data,episodes,windows,files=analyze(directory)
        prefix=directory.name
        all_episodes.extend(episodes); all_windows.extend(windows); manifest.extend(files)
        groups.extend([dict(run=prefix,profile=report['profile'],direction=report.get('direction','up'),**g) for g in report['groups']])
        for name in ('report.json','config.json','geometry.json','execution.json'):
            shutil.copy2(directory/name,args.output/f'{prefix}_{name}')
        # Four fixed-start examples, all at 0.4 m/s and repeat 0.
        fig,axes=plt.subplots(3,4,figsize=(15,8))
        for col,h in enumerate((0.,.05,.1,.15)):
            row=next(r for r in report['records'] if r['height']==h and r['speed']==.4 and r['repeat']==0)
            i=row['env_id']; valid=data['active'][:,i]
            t=(np.arange(len(valid))[valid]+1)*report['step_dt']
            geom=config['geometry'][row['column']]
            axes[0,col].plot(t,data['root'][valid,i,0],label='body x')
            for edge in geom['fronts']: axes[0,col].axhline(edge,color='gray',alpha=.2)
            axes[0,col].set_title(f"{h*100:.0f} cm / {row['outcome']}")
            axes[0,col].set_ylabel('Forward position (m)')
            axes[1,col].plot(t,data['feet'][valid,i,0,2],label='front left')
            axes[1,col].plot(t,data['feet'][valid,i,1,2],label='front right',alpha=.8)
            upper=max(h+.04,float(data['feet'][valid,i,:2,2].max())*1.15)
            for top in geom['tops']:
                if top<upper: axes[1,col].axhline(top,color='gray',alpha=.3)
            lower=min([0.]+geom['tops'])-.03 if report.get('direction','up')=='down' else 0.
            axes[1,col].set_ylim(lower,upper)
            axes[1,col].set_ylabel('Front foot centre height (m)')
            for key in ('track_lin_vel_xy_exp','base_height_l2','flat_orientation_l2','phase_foot_trajectory_exp'):
                j=report['reward_names'].index(key)
                axes[2,col].plot(t,data['reward_rates'][valid,i,j],label=key,alpha=.8)
            axes[2,col].set_xlabel('Time (s)'); axes[2,col].set_ylabel('Weighted reward rate')
        axes[1,0].legend(fontsize=8)
        axes[2,0].legend(fontsize=6)
        fig.suptitle(f"Isaac Lab / {report['profile']} / {report.get('direction','up')} / final10000 / fixed 0.4 m/s, start 0")
        fig.tight_layout(); fig.savefig(args.output/f'{prefix}_trajectories.png',dpi=140)
        fig.savefig(args.output/f'{prefix}_trajectories.pdf'); plt.close(fig)
    write_csv(args.output/'episodes.csv',all_episodes)
    write_csv(args.output/'reward_windows.csv',all_windows)
    flat_groups=[{k:v for k,v in g.items() if k!='outcomes'}|{f'outcome_{k}':g['outcomes'].get(k,0)
                 for k in ('success','stuck','orientation_failure','lateral_exit','timeout')} for g in groups]
    write_csv(args.output/'groups.csv',flat_groups)
    save(args.output/'summary.json',dict(episodes=len(all_episodes),groups=groups))
    save(args.output/'raw_manifest.json',manifest)
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    main()
