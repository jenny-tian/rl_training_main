"""Compare core500 with the frozen final10000 policy on paired protocols."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_turn_analyze import analyze as analyze_turn, WINDOWS
from reference_turn_diagnose import CASES
from reference_isaac_stairs_analyze import analyze as analyze_stairs
from reference_core500_evaluate import TRAINING, OUTPUT, same_initial


def read(path): return json.loads(path.read_text())


def write_csv(path,rows):
    assert rows
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def paired_turn_rows(old,new,profile):
    pairs=[]
    for case in CASES:
        for window in WINDOWS:
            a={r['repeat']:r for r in old if r['case']==case and r['window']==window}
            b={r['repeat']:r for r in new if r['case']==case and r['window']==window}
            common=[i for i in sorted(set(a)&set(b)) if a[i]['complete_window'] and b[i]['complete_window']]
            if not common: continue
            delta=np.array([b[i]['yaw_mae']-a[i]['yaw_mae'] for i in common])
            pairs.append(dict(profile=profile,case=case,window=window,n=len(common),
                baseline_actual=float(np.mean([a[i]['yaw_actual'] for i in common])),
                candidate_actual=float(np.mean([b[i]['yaw_actual'] for i in common])),
                baseline_mae=float(np.mean([a[i]['yaw_mae'] for i in common])),
                candidate_mae=float(np.mean([b[i]['yaw_mae'] for i in common])),
                mae_delta=float(delta.mean()),improved=int((delta< -1e-6).sum()),worsened=int((delta>1e-6).sum())))
    return pairs


def summarize_stairs(report,kind,profile,direction):
    rows=[]
    for height in (0.,.05,.1,.15):
        episodes=[r for r in report['records'] if r['height']==height]
        counts={key:sum(r['outcome']==key for r in episodes) for key in
                ('success','stuck','orientation_failure','lateral_exit','timeout')}
        assert sum(counts.values())==len(episodes)
        rows.append(dict(policy=kind,profile=profile,direction=direction,height=height,starts=len(episodes),
            first_riser=sum(r['best_risers']>=1 for r in episodes),best_risers=max(r['best_risers'] for r in episodes),**counts))
    return rows


def compare_video(video,formal,dest):
    assert read(video/'execution.json')['status']=='completed'
    checks={}
    for name in ('initial','trace'):
        with np.load(video/f'{name}.npz') as a,np.load(formal/f'{name}.npz') as b:
            checks[name]={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
            assert all(checks[name].values()),(video,name,checks[name])
    assert read(video/'report.json')['records']==read(formal/'report.json')['records']
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('rollout.mp4','snapshot.png'):shutil.copy2(video/name,dest/name)
    subprocess.run(['ffmpeg','-v','error','-i',str(dest/'rollout.mp4'),'-f','null','-'],check=True)
    save(dest/'replay_validation.json',dict(status='passed',all_records_equal=True,decode='passed',arrays=checks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/lite3-core500-results-20261112')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    complete=read(OUTPUT/'complete.json');assert complete['status']=='passed'
    matrix=read(OUTPUT/'matrix.json')
    turns=[];windows=[];paired=[];stairs=[];stair_groups=[];episodes=[];rewards=[];checks=[];raw=set()
    for entry in matrix:
        old,new=Path(entry['baseline']),Path(entry['candidate'])
        assert same_initial(old,new)
        pair_windows={}
        for kind,run in [('baseline',old),('candidate',new)]:
            report=read(run/'report.json')
            assert read(run/'execution.json')['status']=='completed' and report['actor_unchanged']
            assert not report['probe']
            if entry['kind']=='turn':
                _,data,rows,groups=analyze_turn(run)
                windows.extend([dict(policy=kind,**r) for r in rows]);turns.extend([dict(policy=kind,**g) for g in groups])
                pair_windows[kind]=rows
            else:
                _,_,data,ep,rw,_=analyze_stairs(run)
                episodes.extend([dict(policy=kind,**r) for r in ep]);rewards.extend([dict(policy=kind,**r) for r in rw])
                stairs.extend(summarize_stairs(report,kind,entry['profile'],entry['direction']))
                for g in report['groups']:
                    stair_groups.append(dict(policy=kind,profile=entry['profile'],direction=entry['direction'],
                        **{k:v for k,v in g.items() if k!='outcomes'},
                        **{f'outcome_{key}':g['outcomes'].get(key,0) for key in ('success','stuck','orientation_failure','lateral_exit','timeout')}))
            assert all(np.isfinite(v).all() for v in data.values())
            checks.append(dict(run=str(run),all_arrays_finite=True,first_episodes_only=True,records=len(report['records'])))
            raw.update(p for p in run.iterdir() if p.is_file())
            for name in ('config.json','report.json','execution.json'):
                shutil.copy2(run/name,out/f'{run.name}_{name}')
        if entry['kind']=='turn': paired.extend(paired_turn_rows(pair_windows['baseline'],pair_windows['candidate'],entry['profile']))
    write_csv(out/'turn_groups.csv',turns);write_csv(out/'turn_windows.csv',windows);write_csv(out/'turn_paired.csv',paired)
    write_csv(out/'stairs_heights.csv',stairs);write_csv(out/'stairs_groups.csv',stair_groups)
    write_csv(out/'stairs_episodes.csv',episodes);write_csv(out/'stairs_reward_windows.csv',rewards)
    for video,formal in [('video_turn','core10500_turn_nominal'),('video_stairs','core10500_up_nominal')]:
        compare_video(OUTPUT/video,OUTPUT/formal,out/'clips'/video)
        raw.update(p for p in (OUTPUT/video).iterdir() if p.is_file())
    training=read(TRAINING/'report.json');execution=read(TRAINING/'execution.json')
    assert training['status']=='passed' and execution['status']=='completed'
    assert training['completed_updates']==500 and training['cumulative_updates']==10500
    assert sha(Path(training['output_checkpoint']))==complete['candidate_sha256']==training['output_sha256']
    for name in ('report.json','execution.json','reward_override_audit.json','checkpoint_restore_audit.json',
                 'runtime_overrides.json','simulation_audit.json'):
        shutil.copy2(TRAINING/name,out/f'training_{name}')
    raw.update(p for p in TRAINING.rglob('*') if p.is_file())
    raw.update(p for p in OUTPUT.iterdir() if p.is_file())
    save(out/'raw_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(raw)])
    save(out/'summary.json',dict(status='passed',candidate_episodes=1600,baseline_episodes=1600,
        model_sha256=complete['candidate_sha256'],training_updates=500,matrix=matrix,
        turn_groups=turns,turn_paired=paired,stairs_heights=stairs,rollout_checks=checks,
        caveat='Single seed and 500-update screen. Reference reward scoring is shared in frozen evaluation; behavior metrics determine improvement.'))
    plot(out,turns,stairs,training)
    print('Core500 analysis complete; paired protocols and both video replays verified.')


def plot(out,turns,stairs,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    cases=('low_left','low_right','turn_left','turn_right','fast_left','fast_right')
    for row,profile in enumerate(('nominal','training_randomized')):
        for policy,label in [('baseline','final10000'),('candidate','core10500')]:
            g=[next(r for r in turns if (r['policy'],r['profile'],r['case'],r['window'])==(policy,profile,c,'steady')) for c in cases]
            axes[row,0].plot(range(6),[r['yaw_mae'] for r in g],marker='o',label=label)
            axes[row,1].plot(range(6),[r['failures_after_command'] for r in g],marker='o',label=label)
        for col in range(2):
            axes[row,col].set_xticks(range(6),['L .2','R .2','L .5','R .5','L .8','R .8']);axes[row,col].set_title(profile);axes[row,col].grid(alpha=.2)
        axes[row,0].set_ylabel('Yaw MAE, complete 4-8 s windows (rad/s)')
        axes[row,1].set_ylabel('Orientation failures after 2 s / 20 starts');axes[row,1].set_ylim(-.5,20.5)
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'turn_comparison.png',dpi=150);fig.savefig(out/'turn_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,profile in enumerate(('nominal','training_randomized')):
        for col,direction in enumerate(('up','down')):
            for policy,label,offset in [('baseline','final10000',-.18),('candidate','core10500',.18)]:
                g=[next(r for r in stairs if (r['policy'],r['profile'],r['direction'],r['height'])==(policy,profile,direction,h)) for h in (.05,.1,.15)]
                axes[row,col].bar(np.arange(3)+offset,[r['first_riser'] for r in g],.36,label=label)
            axes[row,col].set_title(f'{profile} / {direction}');axes[row,col].set_xticks(range(3),['5 cm','10 cm','15 cm'])
            axes[row,col].set_ylim(0,60);axes[row,col].set_ylabel('First step cleared / 60 starts')
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'stairs_comparison.png',dpi=150);fig.savefig(out/'stairs_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));u=training['updates']
    axes[0].plot([r['update'] for r in u],[r['mean_terrain_level'] for r in u]);axes[0].set_ylabel('Mean terrain level')
    axes[1].plot([r['update'] for r in u],[r['learning_rate'] for r in u]);axes[1].set_ylabel('Learning rate')
    for ax in axes:ax.set_xlabel('Additional PPO updates');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training_progress.png',dpi=150);plt.close(fig)


if __name__=='__main__': main()
