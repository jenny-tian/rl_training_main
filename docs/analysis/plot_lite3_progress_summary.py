"""Render the audited project overview and paired recovery results (no new physics)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),
                     'axes.unicode_minus':False,'font.size':11,'savefig.facecolor':'white'})
OUT=ROOT/'docs/figures';OUT.mkdir(exist_ok=True)

def read(name):return json.loads((ROOT/'docs'/name).read_text())
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight')
    svg=OUT/(name+'.svg')
    fig.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def roadmap():
    fig,ax=plt.subplots(figsize=(15,9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,1.015,'Lite3：已完成工作的技术路线',fontsize=22,fontweight='bold',color='#18334d')
    ax.text(.02,.97,'截至 2026-10-08｜箭头表示初始化或技术依赖；每条路线的成绩须按各自协议解读',color='#526272')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.012',facecolor=color,edgecolor='#b5c5d4',linewidth=1.3))
        ax.text(x,y+h*.23,title,ha='center',va='center',fontsize=14,fontweight='bold',color='#18334d')
        ax.text(x,y-h*.12,body,ha='center',va='center',fontsize=11,linespacing=1.5,color='#25364a')
    def arrow(a,b,color='#6c8294',style='-'):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':color,'lw':1.5,'linestyle':style})
    box(.34,.86,.48,.09,'原平地 PPO 基础策略','所有后续方法均继承已有行走能力', '#eff4f8')
    box(.16,.64,.27,.22,'早期适应路线','Residual SAC / LoRA-PPO\nC0 低维回合搜索\n均已运行，未获稳定学习增益','#f1f3f5')
    box(.50,.64,.28,.22,'完整 PPO 微调','路线观测 + 镜像 + 平地锚定\n283 / 307 独立训练\n六地形 59/60；其中新几何 39/40','#e3f2ec')
    box(.84,.64,.27,.22,'FlowDAgger','PPO283 蒸馏条件 flow\n冻结 flow，只训练噪声策略\nv8 开发通过，复现 61/64 失败','#fff1df')
    arrow((.19,.80),(.16,.765));arrow((.48,.80),(.50,.765));arrow((.66,.64),(.69,.64))
    box(.50,.35,.45,.21,'PPO 恢复教师','由 PPO283 + 训练失败初态重新学习\n同一模型冻结普通复测 320/320\n原方法独立重训：留出恢复 0/48','#e8effa')
    arrow((.50,.515),(.50,.47))
    ax.annotate('',xy=(.745,.36),xytext=(.84,.515),arrowprops={'arrowstyle':'-|>','color':'#b88440','lw':1.5,'linestyle':'--','connectionstyle':'angle3,angleA=-90,angleB=0'})
    ax.text(.83,.405,'恢复纠正样本不足\n推动教师开发',ha='center',fontsize=10,color='#986b30',bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':2})
    box(.50,.105,.90,.16,'恢复改进实验：固定恢复槽 → 倾角课程 → 按来源固定槽','最高留出恢复 39/48，但仍有来源全失败；部分方案损害普通行走精度\n教师资格尚未通过，新恢复教师尚未接入 FlowDAgger','#fbe9e7')
    arrow((.50,.23),(.50,.20))
    save(fig,'lite3-progress-roadmap-20261008')


def paired_results():
    first=read('lite3_p2_sampling_ablation_result_20261006.json')['results']
    second=read('lite3_p2_sampling_replication_result_20261006.json')['results']
    course=read('lite3_curriculum_ablation_result_20261007.json')['results']
    source=read('lite3_source_stratified_ablation_result_20261008.json')['results']
    rows=[('原教师\n学习953',second,'candidate'),('固定恢复槽\n学习953',second,'trial'),
          ('原采样重训\n学习2063',first,'candidate'),('固定恢复槽\n学习2063',first,'trial'),
          ('倾角课程\n学习2063',course,'trial'),('来源固定槽\n学习2063',source,'trial')]
    x=np.array([0,1,2.5,3.5,4.5,5.5]);c=[d['course-'+m] for _,d,m in rows];r=[d['recovery-'+m] for _,d,m in rows]
    fig,axs=plt.subplots(3,1,figsize=(14,11),sharex=True,gridspec_kw={'height_ratios':[1,1.15,1.5]})
    fig.suptitle('恢复教师实验：通关、恢复与横向精度必须一起看',fontsize=20,x=.06,ha='left',y=.99,fontweight='bold',color='#18334d')
    fig.text(.06,.948,'均从 PPO283 独立初始化、各训练512轮；在相同评测种子内比较。不同组之间不能直接作因果比较。',fontsize=11,color='#536273')
    for ax in axs:
        ax.axvspan(-.55,1.55,color='#edf4fa',zorder=0);ax.axvspan(1.95,6.05,color='#f5f5f1',zorder=0)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18,zorder=0);ax.set_xlim(-.55,6.05)
    axs[0].text(.5,76,'评测 seed 971',ha='center',fontweight='bold',color='#45627c')
    axs[0].text(4,76,'评测 seed 2081',ha='center',fontweight='bold',color='#626855')
    vals=[v['success'] for v in c];axs[0].bar(x,vals,.54,color='#327f9e',zorder=3)
    for a,v in zip(x,vals):axs[0].text(a,v+1,f'{v}/64',ha='center',fontsize=12)
    axs[0].set(ylim=(0,74),ylabel='正常起步通关数')
    vals=[v['recovered'] for v in r];axs[1].bar(x,vals,.54,color='#db9444',zorder=3)
    for a,v in zip(x,vals):axs[1].text(a,v+1,f'{v}/48',ha='center',fontsize=12)
    axs[1].axhline(38.4,color='#b54d4d',ls='--',lw=1.3)
    axs[1].text(6,46,'80%门槛：至少39/48\n还须每个来源至少成功一次',ha='right',fontsize=10,color='#a24242')
    axs[1].set(ylim=(0,53),ylabel='留出恢复完成数')
    for delta,key,color,label in [(-.16,'rough_rms','#3a8e87','平均回合横偏 RMS'),(.16,'rough_peak','#9a637f','最坏横偏')]:
        vals=[100*v[key] for v in c];axs[2].bar(x+delta,vals,.30,color=color,label=label,zorder=3)
        for a,v in zip(x+delta,vals):axs[2].text(a,v+.8,f'{v:.1f}',ha='center',fontsize=10)
    axs[2].set(ylim=(0,65),ylabel='正常起步粗糙地形横偏（cm）');axs[2].legend(frameon=False,loc='upper right')
    axs[2].set_xticks(x,[label for label,_,_ in rows],fontsize=11)
    fig.text(.06,.047,'所有列均未通过完整恢复教师资格。48个恢复槽重复7个留出来源，并非48个独立来源。',fontsize=11,color='#9b4343')
    fig.text(.06,.020,'本图仅为原101/107地形开发评测；不包含早期六地形验收或冻结教师五种子复测。RMS须结合失败截短和完成率解读。',fontsize=10,color='#536273')
    fig.subplots_adjust(top=.875,bottom=.12,hspace=.28,left=.085,right=.975)
    save(fig,'lite3-recovery-comparison-20261008')


if __name__=='__main__':roadmap();paired_results()
