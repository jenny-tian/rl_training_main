"""Illustrate one fixed target slot from the paid full-course follow-through."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(1,3,figsize=(17,4.6));env_id=40
for arm,color,label in [('base','#ad6262','原冻结PPO'),('prefix','#238a84','0.24秒修正 + 冻结PPO')]:
    folder=ROOT/f'artifacts/recovery_search_20261008/fullcourse-seed2213-{arm}/recovery-trial'
    initial=np.load(folder/'initial.npz');states=[];active=[]
    for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
        with np.load(path) as d:states.append(d['post_root_state'][:,env_id]);active.append(d['active'][:,env_id])
    state=np.concatenate(states)[np.concatenate(active)];time=(np.arange(len(state))+1)*.02
    tilt=np.rad2deg(np.arccos(np.clip(1-2*np.sum(state[:,4:6]**2,axis=1),-1,1)))
    axes[0].plot(time,tilt,color=color,label=label,lw=2)
    axes[1].plot(time,state[:,0]-initial['root'][env_id,0],color=color,label=label,lw=2)
    axes[2].plot(time,100*(state[:,1]-initial['origins'][env_id,1]),color=color,label=label,lw=2)
    report=json.loads((folder/'report.json').read_text());episode=next(e for e in report['episodes'] if e['env_id']==env_id)
    axes[1].scatter(time[-1],state[-1,0]-initial['root'][env_id,0],color=color,s=35,zorder=5)
    axes[1].annotate('完成原路段' if episode['success'] else '失败终止',(time[-1],state[-1,0]-initial['root'][env_id,0]),xytext=(-4,-20) if episode['success'] else (0,10),textcoords='offset points',ha='right' if episode['success'] else 'left',color=color)
axes[0].set(xlim=(0,1.2),ylim=(0,50),xlabel='仿真时间（秒）',ylabel='机身倾角（度）',title='早期恢复：修正结束后仍能保持回正')
axes[0].axhline(10,color='#8a9299',ls='--',lw=1);axes[0].axvspan(0,.24,color='#d9ecea',alpha=.6)
axes[0].axvline(.24,color='#657987',ls=':',lw=1);axes[0].text(.27,44,'交还原PPO',color='#657987')
axes[1].set(xlabel='仿真时间（秒）',ylabel='相对恢复起点的前进距离（米）',title='后续行走：保持原失败规则并到达终点')
axes[2].set(xlabel='仿真时间（秒）',ylabel='相对赛道中线的横偏（厘米）',title='横向偏移：恢复后仍需改进循迹')
axes[2].axhline(0,color='#8a9299',ls='--',lw=1)
for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[1].legend(frameon=False,loc='lower right')
fig.suptitle('TRAIN case26：现有限制下找到可恢复动作',fontweight='bold',fontsize=17,y=1.02)
fig.text(.5,-.025,'示例固定显示seed2213首个目标槽env40；完整统计为两个新启动种子各24/24，均来自同一个训练来源。',ha='center',fontsize=10,color='#536273')
fig.tight_layout();out=ROOT/'docs/figures';out.mkdir(exist_ok=True)
fig.savefig(out/'lite3-recovery-prefix-followthrough-20261008.png',dpi=160,bbox_inches='tight')
svg=out/'lite3-recovery-prefix-followthrough-20261008.svg';fig.savefig(svg,bbox_inches='tight');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
