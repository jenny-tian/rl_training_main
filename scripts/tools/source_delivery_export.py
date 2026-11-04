"""Export the bounded trial's actual curves, load check and diagnostic comparison."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]


def export(root):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rl_training.source_scan.frozen_actor import FrozenSourceActor
    from rl_training.source_scan.delivery_stage import file_digest
    from rl_training.source_scan.smoke_artifacts import atomic_json
    from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
    root=Path(root);out=root/'deliverables';out.mkdir(exist_ok=True)
    checkpoint=root/'numerics/training/actor_update_90.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);actor=FrozenSourceActor(state)
    value=actor(torch.zeros(1,45))
    if value.shape!=(1,12) or not torch.isfinite(value).all():raise ValueError('independent actor load/inference failed')
    curves=[json.loads(line) for line in (root/'numerics/training/curves.jsonl').read_text().splitlines()]
    if [r['update'] for r in curves]!=list(range(1,91)):raise ValueError('training curves lack 90 real updates')
    def csv_file(name,rows):
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    flat=[{k:v for k,v in row.items() if k!='losses'}|{('loss_'+k):v for k,v in row['losses'].items()} for row in curves]
    csv_file('training_curves.csv',flat)
    fig,axes=plt.subplots(2,2,figsize=(11,7));x=[r['update'] for r in curves]
    axes[0,0].plot(x,[r['mean_step_reward'] for r in curves]);axes[0,0].set_ylabel('Mean reward per step')
    for name in curves[0]['losses']:axes[0,1].plot(x,[r['losses'][name] for r in curves],label=name)
    axes[0,1].legend();axes[0,1].set_ylabel('PPO loss')
    axes[1,0].plot(x,[r['mean_action_std'] for r in curves]);axes[1,0].set_ylabel('Mean action std')
    axes[1,1].plot(x,[r['learning_rate'] for r in curves]);axes[1,1].set_ylabel('Learning rate')
    for ax in axes.flat:ax.set_xlabel('Real PPO update');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_digest(checkpoint),
        independent_cpu_load=True,independent_cpu_inference=True,ppo_updates=90,new_training_transitions=259200,
        checkpoint_scope='actor-only inference checkpoint; no optimizer/critic resume claim',formal_selection_qualified=False)
    compare=root/'compare/result.json'
    if compare.is_file():
        report=json.loads(compare.read_text());accept=json.loads((compare.parent/'acceptance.json').read_text())
        if accept.get('accepted') is not True:raise ValueError('comparison did not pass diagnostic integrity checks')
        names=list(SOURCE_RESET_BLOCK_COUNTS);rows=[dict(condition=names[r['column']],**r) for r in report['paired_conditions']]
        csv_file('condition_comparison.csv',rows)
        fig,axes=plt.subplots(1,3,figsize=(18,7));positions=list(range(15))
        for ax,metric,label in zip(axes,['forward_progress_m','velocity_error_rms_mps','completed'],
                ['Final forward progress (m)','RMS velocity error (m/s)','Completed first episodes (%)']):
            for prefix,delta in [('old',-.18),('new',.18)]:
                values=[100*r[prefix+'_completed']/r['pairs'] if metric=='completed' else r[prefix+'_'+metric] for r in rows]
                ax.bar([p+delta for p in positions],values,width=.36,label=prefix)
            ax.set_xticks(positions,names,rotation=90,fontsize=7);ax.set_ylabel(label);ax.legend();ax.grid(axis='y',alpha=.2)
        fig.suptitle('Matched fixed-condition diagnostics; no independent success-rate or formal-selection claim')
        fig.tight_layout();fig.savefig(out/'condition_comparison.png',dpi=160);plt.close(fig)
        result.update(condition_count=15,episode_pairs=120,
            old_completed=sum(r['old_completed'] for r in rows),new_completed=sum(r['new_completed'] for r in rows),
            equal_condition_mean_progress_delta_m=sum(r['delta_forward_progress_m'] for r in rows)/15,
            equal_condition_mean_velocity_error_delta_mps=sum(r['delta_velocity_error_rms_mps'] for r in rows)/15)
    atomic_json(out/'delivery_summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();export(args.root)
