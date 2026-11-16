"""两个未获原落脚奖励的事件；只画保存的轨迹和离线候选得分。"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reference_low_stairs_reward_state_probe import OUTPUT,DOCS


def main():
    events=list(csv.DictReader((DOCS/'landing_candidate_events.csv').open()))
    with np.load(OUTPUT/'nominal/reward_state.npz') as archive:
        data={k:archive[k] for k in ('bottom','ground','support','wall','land_peak_at_check')}
    with np.load(OUTPUT/'nominal/initial.npz') as initial:origins=initial['origins']
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for ax,(i,f,leg,lo,hi,touch) in zip(axes,((196,2,'HL',9.1,9.85,9.42),(199,0,'FL',3.4,4.15,3.78))):
        a,b=round(lo/.02),round(hi/.02);t=(np.arange(a,b)+1)*.02;z=origins[i,2]
        k=round(touch/.02)-1;top=data['ground'][k,i,f]-z
        foot=(data['bottom'][a:b,i,f]-z)*1000
        ax.plot(t,foot,label='Foot bottom',color='#2166ac',linewidth=2)
        ax.step(t,(data['ground'][a:b,i,f]-z)*1000,where='post',color='#333333',label='Ray surface')
        ax.axhline((top+.015)*1000,color='#b2182b',linestyle='--',label='Original peak requirement: +15 mm')
        ax.axhline((top+.005)*1000,color='#1b7837',linestyle='--',label='Candidate peak requirement: +5 mm')
        support=data['support'][a:b,i,f]&~data['wall'][a:b,i,f]
        ax.scatter(t[support],foot[support],s=13,color='#777777',zorder=3,label='Support proxy')
        for e in events:
            if e['profile']=='nominal' and int(e['env_id'])==i and e['leg']==leg and e['candidate']=='True' and lo<float(e['time_s'])<hi:
                award=float(e['time_s']);idx=round(award/.02)-1
                ax.scatter([award],[(data['bottom'][idx,i,f]-z)*1000],marker='*',s=180,color='#762a83',zorder=5,label='Candidate credit (offline only)')
        peak=(data['land_peak_at_check'][k,i,f]-data['ground'][k,i,f])*1000
        ax.set_title(f'Nominal {i}, {leg}: swing peak {peak:.3f} mm above next tread; original credit = 0')
        ax.set_xlabel('Time from recorded episode start (s)');ax.set_ylabel('Height above origin (mm)')
        ax.grid(alpha=.2);ax.set_xlim(lo,hi)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
    fig.savefig(DOCS/'landing_peak_examples.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
