"""Summarize the direct mean replay ablation against its fixed paid PPO control."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=dict(plan='docs/lite3_mean_anchor_plan_20261009.json',update_audit='docs/lite3_mean_anchor_update_audit_20261009.json',
        pilot_result='docs/lite3_mean_anchor_pilot_result_20261009.json',run='artifacts/mean_anchor_20261009/pilot-evaluation',
        control_run='artifacts/recovery_return_20261008/trial/pilot-evaluation',output='docs/lite3_mean_anchor_result_20261009.json',
        figure='docs/figures/lite3-mean-anchor-20261009')
    for key,value in defaults.items():parser.add_argument('--'+key.replace('_','-'),default=value)
    parser.add_argument('--control-update-audit')
    args=parser.parse_args()
    evidence={}
    def read(path):
        evidence[str(path)]=sha256_file(path)
        return json.loads(Path(path).read_text())
    plan=read(args.plan)
    control=read(plan['control_result'])
    assert sha256_file(plan['control_result'])==plan['control_result_sha256']
    initial=read('docs/lite3_retention_coverage_pilot_result_20261008.json')
    trained=read(args.update_audit)
    result=read(args.pilot_result)
    for r in (control,initial,trained,result):
        assert r['status']=='verified'
        for path,digest in r['evidence'].items():assert sha256_file(path)==digest
    oldrun=read(Path(args.control_run)/'report.json')
    newrun=read(Path(args.run)/'report.json')
    for condition in ('course','recovery'):
        assert oldrun['phases'][condition+'-trial']['initial_sha256']==newrun['phases'][condition+'-trial']['initial_sha256']
    olddiagnostic=(read(args.control_update_audit)['offline_action_drift_from_initial_actor'] if args.control_update_audit else
        read('docs/lite3_recovery_return_result_20261008.json')['offline_action_drift_from_initial_actor']['trial']['groups'])
    ratios={key:value['mean_squared_raw_action_change']/olddiagnostic[key]['mean_squared_raw_action_change']
            for key,value in trained['offline_action_drift_from_initial_actor'].items()}
    assert trained['controls']==plan['training_controls'] and result['controls']<=plan['max_pilot_controls']
    total=trained['controls']+result['controls']
    costs=dict(training=trained['controls'],pilot=result['controls'],replay_new_physics=0,new_uncertain=0,
        stage_through_pilot=total,cumulative_charged_through_pilot=plan['cumulative_before']['charged']+total,
        cumulative_confirmed_through_pilot=plan['cumulative_before']['confirmed']+total,historical_uncertain=64)
    out=dict(status='verified',initial=initial['results']['trial'],control=control['results']['trial'],trial=result['results']['trial'],
        gates=result['gates'],pilot_passed=result['pilot_passed'],costs=costs,final_checkpoint=trained['final_checkpoint'],
        failures={key:[e for e in value['episodes'] if not e['success']] for key,value in newrun['phases'].items()},
        offline_mean_action_mse_ratio_vs_control=ratios,mean_anchor_loss=trained['mean_anchor_loss'],evidence=evidence,
        scope=('Fixed TRAIN replay experiment, learner2281 and developer eval2251. Repeated TRAIN case26 slots, not original7-source qualification or independent learning replication.'
            if plan['schema']=='lite3_mean_anchor_length_ablation_v1' else
            'One fixed mean-loss ablation, same learner2281 and developer eval2251. Repeated TRAIN case26 slots, not original7-source qualification or independent learning replication.'))
    write_json_atomic(args.output,out)
    metrics=[out['initial'],out['control'],out['trial']]
    length=plan['schema']=='lite3_mean_anchor_length_ablation_v1'
    labels=['Supervised init','PPO 128 updates','PPO 256 updates'] if length else ['Supervised init','PPO reward anchor','+ Direct mean loss']
    base=result['results']['base']
    plots=[('course_total',1,'Ordinary completion (of 64)',None),
        ('course_peak_m',100,'Ordinary rough peak lateral error (cm)',base['course_peak_m']*100),
        ('case26_recovery',1,'TRAIN case26 recovery + completion (of 24)',20),
        ('case26_peak_m',100,'Recovery peak lateral error (cm)',70)]
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,(key,scale,title,gate) in zip(axes.flat,plots):
        heights=[m[key]*scale for m in metrics]
        ax.bar(labels,heights,color=['#65758B','#B27C48','#2A8C82'],width=.6)
        for i,value in enumerate(heights):ax.text(i,value,f'{value:.2f}' if scale==100 else str(int(value)),ha='center',va='bottom')
        if gate is not None:
            ax.axhline(gate,color='#B44747',ls='--',label=f'Gate {gate:.2f}')
            ax.legend(frameon=False)
        ax.set_ylim(0,max(heights+[gate or 0])*1.25)
        ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    title='PPO training length' if length else 'Direct mean-action loss'
    fig.suptitle(f"{title}: pilot {'PASSED' if out['pilot_passed'] else 'FAILED'}\nFixed final{plan['iterations']} PPO updates, same evaluation2251, no teacher/prefix at inference",fontsize=14)
    target=Path(args.figure)
    for suffix in ('.png','.svg'):fig.savefig(target.with_suffix(suffix),dpi=160)
    svg=target.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
