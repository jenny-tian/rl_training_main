"""Run the approved frozen core500 matrix after its bounded training finishes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_six_watch import snapshot

TRAINING = ROOT/'logs/reference_six/reward_experiments/core_seed42_500'
OUTPUT = ROOT/'logs/reference_six/core500_evaluation_20261112'
OLD_TURN = ROOT/'logs/reference_six/code_turn_audit_20261112'
OLD_STAIRS = ROOT/'logs/reference_six/isaac_stairs_diagnosis_20261112'


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return None


def same_initial(left,right):
    with np.load(left/'initial.npz') as a,np.load(right/'initial.npz') as b:
        return a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-training',action='store_true')
    parser.add_argument('--gpu',type=int,choices=(0,1),default=1)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    while True:
        execution=read(TRAINING/'execution.json')
        report=read(TRAINING/'report.json')
        if execution and execution['status']=='failed': raise RuntimeError('training supervisor failed')
        if execution and execution['status']=='completed': break
        if not args.wait_for_training: raise RuntimeError('training has not completed')
        progress=snapshot(TRAINING)
        if progress:
            save(OUTPUT/'training_progress.json',progress)
            if progress['status']=='awaiting_supervisor_result':
                time.sleep(5)
                if read(TRAINING/'execution.json')['status']=='running':
                    raise RuntimeError('training worker exited without supervisor completion')
        time.sleep(20)
    assert report['status']=='passed' and report['completed_updates']==500 and report['cumulative_updates']==10500
    assert report['reward_variant']=='core' and report['environment_steps']==12000
    checkpoint=Path(report['output_checkpoint']); model_hash=sha(checkpoint)
    assert model_hash==report['output_sha256']
    save(OUTPUT/'training_progress.json',snapshot(TRAINING))
    candidate=['--model','core10500','--checkpoint',str(checkpoint),'--checkpoint-sha256',model_hash]
    runs=[]
    def run(name,script,flags):
        directory=OUTPUT/name
        command=[sys.executable,str(ROOT/'scripts/tools'/script),*flags,'--gpu',str(args.gpu),'--output',str(directory)]
        # Only reuse previously completed formal jobs with the same model hash.
        if directory.exists():
            result=read(directory/'report.json'); execution=read(directory/'execution.json')
            if not result or not execution or execution['status']!='completed' or result['status']!='passed':
                raise RuntimeError(f'incomplete prior evaluation: {directory}')
            if '--checkpoint-sha256' in flags:
                assert result.get('model_sha256',result.get('checkpoint_sha256'))==model_hash
        else:
            print('EVALUATE',name,flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
        return directory
    for profile in ('nominal','training_randomized'):
        turn=run(f'core10500_turn_{profile}','reference_turn_diagnose.py',candidate+['--profile',profile])
        baseline=OLD_TURN/f'update10000_{profile}'
        if not same_initial(turn,baseline):
            baseline=run(f'baseline_turn_{profile}','reference_turn_diagnose.py',['--model','update10000','--profile',profile])
        assert same_initial(turn,baseline)
        runs.append(dict(kind='turn',profile=profile,candidate=str(turn),baseline=str(baseline),initial_equal=True))
        save(OUTPUT/'matrix.json',runs)
        for direction in ('up','down'):
            flags=['--profile',profile,'--direction',direction]
            stairs=run(f'core10500_{direction}_{profile}','reference_isaac_stairs_diagnose.py',candidate+flags)
            baseline=OLD_STAIRS/f'{profile}_v1' if direction=='up' else None
            if baseline is None or not same_initial(stairs,baseline):
                baseline=run(f'baseline_{direction}_{profile}','reference_isaac_stairs_diagnose.py',flags)
            assert same_initial(stairs,baseline)
            runs.append(dict(kind='stairs',direction=direction,profile=profile,candidate=str(stairs),baseline=str(baseline),initial_equal=True))
            save(OUTPUT/'matrix.json',runs)
    for name,script,flags in (
        ('video_turn','reference_turn_diagnose.py',['--profile','nominal','--video-env','220']),
        ('video_stairs','reference_isaac_stairs_diagnose.py',['--profile','nominal','--direction','up','--video-env','140']),
    ):
        run(name,script,candidate+flags)
    save(OUTPUT/'complete.json',dict(status='passed',candidate_episodes=1600,candidate_sha256=model_hash,
        training_updates=500,automatic_training_extension=False,matrix=runs))
    print('CORE500 MATRIX COMPLETE: 1600 candidate episodes, paired baseline, two video replays',flush=True)


if __name__=='__main__': main()
