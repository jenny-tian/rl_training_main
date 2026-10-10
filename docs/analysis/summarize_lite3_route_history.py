"""Summarize the dimension-matched information ablation and complete physical cost."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--figure',required=True);args=parser.parse_args()
    evidence={}
    def read(path):
        evidence[str(path)]=sha256_file(path)
        return json.loads(Path(path).read_text())
    plan=read('docs/lite3_route_history_plan_20261010.json');pair=read('docs/lite3_route_history_training_pair_audit_20261010.json')
    assert pair['status']=='verified'
    results={};costs={};reports={}
    for arm in plan['arms']:
        result=read(f'docs/lite3_route_history_{arm}_pilot_result_20261010.json')
        update=read(f'docs/lite3_route_history_{arm}_update_audit_20261010.json')
        history=read(f'docs/lite3_route_history_{arm}_pilot_history_audit_20261010.json')
        report=read(f'artifacts/route_history_20261010/{arm}/pilot-combined/report.json')
        smoke=read(f'artifacts/route_history_20261010/{arm}-smoke/report.json')
        assert result['status']==update['status']==history['status']=='verified' and report['status']==smoke['status']=='complete'
        for checked in [result,update,history]:
            for path,digest in checked['evidence'].items():assert sha256_file(path)==digest
        assert result['controls']==history['controls']==report['ledger']['confirmed']<=plan['max_pilot_controls_per_arm']
        costs[arm]=dict(engineering=smoke['ledger']['confirmed'],training=update['controls'],pilot=result['controls'])
        results[arm]=dict(metrics=result['results']['trial'],gates=result['gates'],pilot_passed=result['pilot_passed'],final_checkpoint=update['final_checkpoint'],history_column_norm=update['history_column_norm'])
        reports[arm]=report
    for condition in ['course','recovery']:
        assert reports['zero']['phases'][condition+'-trial']['initial_sha256']==reports['history']['phases'][condition+'-trial']['initial_sha256']
    for stage,limit in [('engineering','max_engineering_controls'),('training','max_training_controls'),('pilot','max_pilot_controls')]:
        assert sum(c[stage] for c in costs.values())<=plan[limit]
    total=sum(sum(c.values()) for c in costs.values());assert total<=plan['max_new_controls']
    costs.update(stage_through_pilot=total,cumulative_charged=plan['cumulative_before']['charged']+total,
        cumulative_confirmed=plan['cumulative_before']['confirmed']+total,historical_uncertain=64,new_uncertain=0)
    out=dict(status='verified',arms=results,costs=costs,history_pilot_passed=results['history']['pilot_passed'],evidence=evidence,
        scope='Same49D networks/initialization dimensions; only causal history input enabled differs. Original six gates unchanged, TRAIN case26 repetitions only. Earlier48D run is not this paired causal control. Physical cost includes both arms and both engineering runs.')
    write_json_atomic(args.output,out)
    baseline=result['results']['base'];labels=['Zero history input\n49D control','Causal history input\n49D policy']
    metrics=[results['zero']['metrics'],results['history']['metrics']]
    plots=[('course_rms_m',100,'Ordinary rough mean episode RMS (cm)',baseline['course_rms_m']*1.05*100),
        ('course_peak_m',100,'Ordinary rough peak lateral error (cm)',baseline['course_peak_m']*100),
        ('case26_recovery',1,'TRAIN case26 recovery + completion (of 24)',20),
        ('case26_peak_m',100,'Recovery peak lateral error (cm)',70)]
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(key,scale,title,gate) in zip(axes.flat,plots):
        values=[m[key]*scale for m in metrics];ax.bar(labels,values,color=['#A87A4B','#278C82'],width=.55)
        for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}' if scale==100 else str(int(value)),ha='center',va='bottom')
        ax.axhline(gate,color='#B44747',ls='--',label=f'Original gate {gate:.2f}');ax.legend(frameon=False)
        ax.set_ylim(0,max(values+[gate])*1.25);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One-frame route history: '+('PASSED' if out['history_pilot_passed'] else 'FAILED')+' original pilot gates\nMatched49D pair, final256, learner2281/eval2251',fontsize=14)
    target=Path(args.figure)
    for suffix in ['.png','.svg']:fig.savefig(target.with_suffix(suffix),dpi=160)
    svg=target.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');plt.close(fig)


if __name__=='__main__':main()
