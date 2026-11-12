"""First-episode turn metrics with explicit survivor counts and paired comparisons."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from reference_stairs_eval import sha,save
from reference_turn_diagnose import CASES

WINDOWS={'steady':(4.,8.),'late':(10.,14.),'tail':(16.,20.)}


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def analyze(directory):
    r=json.loads((directory/'report.json').read_text())
    assert r['status']=='passed' and not r['probe'] and r['actor_unchanged']
    assert json.loads((directory/'execution.json').read_text())['status']=='completed'
    with np.load(directory/'trace.npz') as a: data={k:a[k] for k in a.files}
    n=len(data['active']); time=np.arange(n)*r['step_dt']; rows=[]
    for ep in r['records']:
        i=ep['env_id']; first=data['active'][:,i]; valid=first&~data['terminated'][:,i]
        assert np.count_nonzero(first)==round(ep['duration_s']/r['step_dt'])
        for window,(start,end) in WINDOWS.items():
            mask=valid&(time>=start)&(time<end)
            expected=round((end-start)/r['step_dt'])
            row=dict(model=r['model'],profile=r['profile'],case=ep['case'],repeat=ep['repeat'],env_id=i,
                outcome=ep['outcome'],duration_s=ep['duration_s'],window=window,samples=int(mask.sum()),
                complete_window=bool(mask.sum()==expected),yaw_command=None,yaw_actual=None,yaw_mae=None,
                xy_speed=None,heading_error_deg=None,yaw_change_deg=None,position_drift_m=None,stand_penalty=None,
                air_turn_reward=None,phase_reward=None)
            if mask.any():
                cmd=data['command'][mask,i]; vel=data['velocity'][mask,i]; pos=data['root'][mask,i]
                yaw=np.unwrap(data['heading'][mask,i])
                rw=data['reward_rates'][mask,i]
                row.update(yaw_command=float(cmd[:,2].mean()),yaw_actual=float(vel[:,5].mean()),
                    yaw_mae=float(np.abs(cmd[:,2]-vel[:,5]).mean()),xy_speed=float(np.linalg.norm(vel[:,:2],axis=-1).mean()),
                    heading_error_deg=float(np.degrees(np.abs(data['heading_error'][mask,i])).mean()) if ep['case'].startswith('heading') else None,
                    yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),position_drift_m=float(np.linalg.norm(pos[-1,:2]-pos[0,:2])),
                    stand_penalty=float(rw[:,r['reward_names'].index('stand_still')].mean()),
                    air_turn_reward=float(rw[:,r['reward_names'].index('feet_air_time_ang_z_lite3')].mean()),
                    phase_reward=float(rw[:,r['reward_names'].index('phase_foot_trajectory_exp')].mean()))
            rows.append(row)
    groups=[]
    for case in CASES:
        episodes=[ep for ep in r['records'] if ep['case']==case]
        for window in WINDOWS:
            selected=[row for row in rows if row['case']==case and row['window']==window and row['complete_window']]
            g=dict(model=r['model'],profile=r['profile'],case=case,window=window,starts=len(episodes),
                survived_20s=sum(ep['outcome']=='survived' for ep in episodes),
                orientation_failures=sum(ep['outcome']=='orientation_failure' for ep in episodes),
                failures_before_command=sum(ep['outcome']=='orientation_failure' and ep['duration_s']<=2. for ep in episodes),
                failures_after_command=sum(ep['outcome']=='orientation_failure' and ep['duration_s']>2. for ep in episodes),
                complete_windows=len(selected))
            for key in ('yaw_command','yaw_actual','yaw_mae','xy_speed','heading_error_deg','position_drift_m','stand_penalty','air_turn_reward','phase_reward'):
                values=[row[key] for row in selected if row[key] is not None]
                g[key]=float(np.mean(values)) if values else None
            groups.append(g)
    return r,data,rows,groups


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    all_rows=[]; groups=[]; reports=[];manifest=[]; initial={}; pairing=[]
    for run in args.runs:
        report,data,rows,group=analyze(run)
        all_rows.extend(rows); groups.extend(group); reports.append(report)
        with np.load(run/'initial.npz') as a:
            values={k:a[k] for k in a.files}
        profile=report['profile']
        if profile not in initial: initial[profile]=values
        else:
            for k,v in values.items(): assert np.array_equal(v,initial[profile][k]),(profile,k)
        pairing.append(dict(run=run.name,initial_matches_same_profile=True))
        for name in ('report.json','config.json','execution.json'):
            shutil.copy2(run/name,args.output/f'{run.name}_{name}')
        for path in run.iterdir():
            if path.is_file(): manifest.append(dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size))
    write_csv(args.output/'episode_windows.csv',all_rows); write_csv(args.output/'groups.csv',groups)
    # Paired changes only use repetitions which finish the same full window in both models.
    pairs=[]
    for profile in ('nominal','training_randomized'):
        for earlier in ('update4001','update6001','update8001'):
            for case in CASES:
                for window in WINDOWS:
                    old={r['repeat']:r for r in all_rows if (r['model'],r['profile'],r['case'],r['window'])==(earlier,profile,case,window)}
                    new={r['repeat']:r for r in all_rows if (r['model'],r['profile'],r['case'],r['window'])==('update10000',profile,case,window)}
                    common=[i for i in set(old)&set(new) if old[i]['complete_window'] and new[i]['complete_window']]
                    if not common: continue
                    delta=np.array([new[i]['yaw_mae']-old[i]['yaw_mae'] for i in common])
                    pairs.append(dict(profile=profile,earlier=earlier,case=case,window=window,n=len(common),
                        yaw_mae_delta_final_minus_earlier=float(delta.mean()),
                        worsened=sum(delta>1e-6).item(),improved=sum(delta< -1e-6).item()))
    if pairs: write_csv(args.output/'paired_changes.csv',pairs)
    save(args.output/'summary.json',dict(status='passed',episodes=sum(r['episodes'] for r in reports),
        groups=groups,paired_initial_states=pairing,
        metric_note='Velocity/error means use complete surviving windows; failure counts always include all starts. Surviving does not imply turning accurately.'))
    save(args.output/'raw_manifest.json',manifest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    cases=('low_left','low_right','turn_left','turn_right','fast_left','fast_right')
    for row,profile in enumerate(('nominal','training_randomized')):
        for model in ('update4001','update6001','update8001','update10000'):
            selected=[next((g for g in groups if (g['model'],g['profile'],g['case'],g['window'])==(model,profile,c,'steady')),None) for c in cases]
            if any(g is None for g in selected): continue
            axes[row,0].plot(range(6),[g['yaw_mae'] for g in selected],marker='o',label=model)
            axes[row,1].plot(range(6),[g['failures_after_command'] for g in selected],marker='o',label=model)
        for col in range(2):
            axes[row,col].set_xticks(range(6),['L .2','R .2','L .5','R .5','L .8','R .8'])
            axes[row,col].set_title(profile); axes[row,col].grid(alpha=.2)
        axes[row,0].set_ylabel('Yaw MAE, complete 4-8 s windows (rad/s)')
        axes[row,1].set_ylabel('Orientation failures after command starts / 20 starts'); axes[row,1].set_ylim(-.5,20.5)
    axes[0,0].legend(); fig.tight_layout(); fig.savefig(args.output/'turn_comparison.png',dpi=150); fig.savefig(args.output/'turn_comparison.pdf')
    print('Analyzed',sum(r['episodes'] for r in reports),'episodes with matched initial states.')


if __name__=='__main__': main()
