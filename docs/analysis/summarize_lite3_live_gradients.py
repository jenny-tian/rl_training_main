"""Summarize immutable first-minibatch gradient telemetry after a completed run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','update-audit','output','figure'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();root=Path(args.run)
    report=json.loads((root/'report.json').read_text());audit=json.loads(Path(args.update_audit).read_text())
    trace=root/'gradient_diagnostics.jsonl'
    assert report['status']=='complete' and audit['status']=='verified'
    assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
    assert sha256_file(trace)==report['gradient_diagnostics']['sha256']==audit['evidence'][str(trace)]
    records=[json.loads(line) for line in trace.read_text().splitlines()]
    assert len(records)==256 and [r['update'] for r in records]==list(range(1,257))
    result={}
    for label,selected in [('all',records)]+[(str(lo+1)+'-'+str(lo+64),records[lo:lo+64]) for lo in range(0,256,64)]:
        groups={}
        for group in ('anchor','correction','ordinary'):
            ratios=[r['norms'][group]/r['norms']['ppo'] for r in selected if r['norms']['ppo']>0]
            cos=[r['cosine'][group+'_vs_ppo'] for r in selected if r['cosine'][group+'_vs_ppo'] is not None]
            groups[group]=dict(median_norm_ratio_to_ppo=float(np.median(ratios)),p95_norm_ratio_to_ppo=float(np.quantile(ratios,.95)),
                valid_cosines=len(cos),mean_cosine=float(np.mean(cos)),fraction_cosine_negative=float(np.mean(np.array(cos)<0)))
        result[label]=dict(records=len(selected),groups=groups,
            fraction_ordinary_norm_above_correction=float(np.mean([r['norms']['ordinary']>r['norms']['correction'] for r in selected])))
    evidence={str(p):sha256_file(p) for p in [root/'report.json',Path(args.update_audit),trace]}
    write_json_atomic(args.output,dict(status='verified',blocks=result,new_controls=0,extra_optimizer_steps=0,evidence=evidence,
        scope='One actual first minibatch per update; mean-MLP gradients before clipping/Adam, PPO mixes ordinary/recovery rows. Negative cosine is a local tradeoff, not proof of harmful retention or the actual Adam direction. No control-run live gradient history exists.'))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    colors={'anchor':'#546778','correction':'#BB7850','ordinary':'#238C81'}
    for group in colors:
        ratios=np.array([r['norms'][group]/r['norms']['ppo'] for r in records])
        cos=np.array([r['cosine'][group+'_vs_ppo'] if r['cosine'][group+'_vs_ppo'] is not None else np.nan for r in records])
        axes[0].plot(np.arange(1,257),np.where(ratios>0,ratios,np.nan),label=group,color=colors[group],alpha=.8)
        axes[1].plot(np.arange(1,257),cos,label=group,color=colors[group],alpha=.7)
    axes[0].set(yscale='log',xlabel='PPO update',ylabel='Gradient norm / PPO mean-gradient norm',title='Relative preservation gradient magnitude')
    axes[1].set(xlabel='PPO update',ylabel='Cosine with PPO mean gradient',ylim=(-1.05,1.05),title='Local gradient alignment')
    axes[1].axhline(0,color='#888',linestyle='--');axes[0].axhline(1,color='#888',linestyle='--')
    for ax in axes:ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Correction weight 0.5: actual first minibatch of each update, before clipping and Adam')
    path=Path(args.figure)
    for suffix in ('.png','.svg'):fig.savefig(path.with_suffix(suffix),dpi=160)
    svg=path.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
