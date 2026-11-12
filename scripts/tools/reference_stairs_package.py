"""Archive paired stair results and plots without copying mutable model weights."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from reference_stairs_eval import ROOT, MODELS, summarize, save, sha

RAW = ROOT / 'logs/reference_six/stairs_eval_20261112'
OUT = ROOT / 'docs/lite3-reference-stairs-eval-20261112'


def partial_risers(row):
    """Diagnostic only: all four feet beyond a riser and above its height for >=0.1 s."""
    if row['direction'] != 'up':
        return None
    best = 0
    for n in range(1, 7):
        first = None
        for p in row['trace']:
            clear = (min(p['feet_x']) > 1.15 + (n-1)*.3 + .022
                     and min(p['feet_z']) > n*.1 -.01
                     and p['xyz'][2] > n*.1 + .18 and p['tilt_deg'] < 44.)
            if clear:
                if first is None:
                    first = p['t']
                if p['t'] - first >= .099:
                    best = n
                    break
            else:
                first = None
    return best


def collect():
    records = []
    manifest = []
    for directory, names in ((RAW/'full_v1',('update8001','update10000')), (RAW/'reference_ort_v1',('reference',))):
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['status']=='completed' and summary['models_unchanged']
        for path in sorted((directory/'episodes').glob('*.json')):
            row=json.loads(path.read_text())
            if row['model'] not in names:
                continue
            row['partial_risers']=partial_risers(row)
            row['source_path']=str(path)
            records.append(row)
            manifest.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    assert len(records)==1080
    unique={(r['model'],r['profile'],r['direction'],r['speed'],r['repeat']) for r in records}
    assert len(unique)==1080
    return records,manifest


def table_csv(path, rows):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records,manifest=collect()
    summary=summarize(records)
    summary.update(models_unchanged=True,reference_backend='original ONNX Runtime',
                   full_training_updates=10000,intermediate_checkpoint_updates=8001,
                   partial_riser_definition='All four feet beyond/above a riser, base height >=0.18 m, tilt <44 degrees for at least 0.10 s; diagnostic only, not complete six-riser success.')
    aggregate=[]
    for profile in ('deployment','training_nominal'):
        for name in MODELS:
            for direction in ('flat','up','down'):
                rows=[r for r in records if (r['profile'],r['model'],r['direction'])==(profile,name,direction)]
                aggregate.append(dict(profile=profile,model=name,direction=direction,episodes=len(rows),
                     **{outcome:sum(r['outcome']==outcome for r in rows) for outcome in ('success','orientation_failure','stuck','timeout','lateral_exit','policy_numerical_failure')},
                     reached_first_riser=sum((r['partial_risers'] or 0)>=1 for r in rows),
                     best_partial_risers=max((r['partial_risers'] or 0) for r in rows),
                     mean_velocity_xy_error=float(np.mean([r['mean_velocity_xy_error'] for r in rows if r['mean_velocity_xy_error'] is not None]))))
    summary['aggregate']=aggregate
    save(OUT/'summary.json',summary)
    save(OUT/'raw_manifest.json',manifest)
    table_csv(OUT/'by_speed.csv',summary['groups'])
    table_csv(OUT/'aggregate.csv',aggregate)
    table_csv(OUT/'episodes.csv',[{k:v for k,v in r.items() if k not in ('trace','numerical_failure')} for r in records])
    for run in ('full_v1','reference_ort_v1'):
        for name in ('audit.json','execution.json'):
            shutil.copy2(RAW/run/name,OUT/f'{run}_{name}')
    shutil.copy2(RAW/'onnx_runtime_crosscheck.json',OUT/'onnx_runtime_crosscheck.json')
    train=ROOT/'logs/reference_six/training/scratch_seed42_10000'
    for name in ('progress.json','execution.json'):
        shutil.copy2(train/name,OUT/f'training_{name}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    colors=('#507bb5','#e5a02a','#52996e')
    for rid,profile in enumerate(('deployment','training_nominal')):
        for cid,direction in enumerate(('up','down')):
            ax=axes[rid,cid]
            for idx,name in enumerate(MODELS):
                rows=[r for r in summary['groups'] if (r['profile'],r['direction'],r['model'])==(profile,direction,name)]
                values=[r['success_rate']*100 for r in rows]
                bars=ax.bar(np.arange(4)+(idx-1)*.24,values,.23,label=name,color=colors[idx])
                for bar,r in zip(bars,rows):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1+idx*2.7,f"{r['success']}/20",ha='center',fontsize=7)
            ax.set_xticks(range(4),['0.2','0.4','0.6','0.8'])
            ax.set_ylim(0,112)
            ax.set_xlabel('Commanded forward velocity (m/s)')
            ax.set_ylabel('Complete six-riser traversal (%)')
            ax.set_title(f'{profile} / {direction}')
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(loc='upper left')
    fig.suptitle('Lite3 paired evaluation: 10 cm risers, 30 cm treads, 20 identical starts per cell')
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=160)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    # Fixed-start clips are chosen independently of outcome. An extra reference
    # example can illustrate partial climbing but never changes the full statistics.
    clips=[r for r in records if r['profile']=='deployment' and r['direction'] in ('up','down') and r['speed']==.4 and r['repeat']==0]
    best=max([r for r in records if r['model']=='reference' and r['direction']=='up'],key=lambda r:(r['partial_risers'],r['progress_m']))
    if best['source_path'] not in {r['source_path'] for r in clips}:
        clips.append(best)
    save(OUT/'video_selection.json',[{k:r[k] for k in ('model','profile','direction','speed','repeat','outcome','source_path','partial_risers')} for r in clips])
    from PIL import Image, ImageDraw
    montage=Image.new('RGB',(1440,640),(20,25,32))
    for i,row in enumerate(clips):
        clip_name=(f"{row['model']}_{row['direction']}_0.4_00" if row['speed']==.4 and row['repeat']==0
                   else 'reference_best_partial_up')
        source=RAW/'videos'/clip_name
        replay=json.loads((source/'replay.json').read_text())
        assert replay['outcome']==row['outcome'] and replay['initial_state_sha256']==row['initial_state_sha256']
        assert abs(replay['duration_s']-row['duration_s'])<.011
        videos=OUT/'videos'
        videos.mkdir(exist_ok=True)
        shutil.copy2(source/'rollout.mp4',videos/f'{clip_name}.mp4')
        shutil.copy2(source/'replay.json',videos/f'{clip_name}.json')
        at=min(8.,row['duration_s']*.75) if clip_name!='reference_best_partial_up' else 9.8
        image_path=videos/f'{clip_name}.png'
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(at),'-i',str(videos/f'{clip_name}.mp4'),'-frames:v','1',str(image_path)],check=True)
        if i<6:
            # Fixed order: one column per policy, uphill then downhill.
            col=list(MODELS).index(row['model'])
            line=0 if row['direction']=='up' else 1
            montage.paste(Image.open(image_path).resize((480,320)),(col*480,line*320))
    montage.save(OUT/'video_montage.png')
    shutil.copy2(RAW/'tests.xml',OUT/'tests.xml')
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in paths))
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    main()
