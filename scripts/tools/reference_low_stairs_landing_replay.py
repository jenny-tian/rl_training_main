"""CPU 上重放已保存输入，不调用仿真、策略或优化器，不产生训练样本。"""
from pathlib import Path

import numpy as np
import torch

from reference_low_stairs_rewards import LandingState
from reference_low_stairs_landing_candidate import LandingPeak5mmState
from reference_low_stairs_reward_state_probe import OUTPUT,DOCS,BASELINES,DIGEST,read
from reference_low_stairs_reward_state_analyze import npz,LEGS,FOCUS,DT
from reference_stairs_eval import sha,save
from reference_core500_analyze import write_csv

INPUTS=('feet','bottom','ground','support','root_xy','direction','enabled','upright')


def replay(profile):
    path=OUTPUT/profile;data=npz(path/'reward_state.npz');initial=npz(path/'shadow_initial.npz')
    trace=npz(path/'trace.npz');origins=npz(path/'initial.npz')['origins']
    report=read(path/'report.json');n,count=trace['active'].shape
    old=LandingState(count,'cpu');new=LandingPeak5mmState(count,'cpu')
    # PD 准备阶段命令全零，落脚奖励未启用；同一状态可作为两者的起点。
    with np.load(path/'preparation_state.npz') as p:assert not p['enabled'].any()
    for state in (old,new):
        for k in vars(state):setattr(state,k,torch.from_numpy(initial['land_'+k].copy()))
        assert not state.pending.any()
    original_awards=np.zeros((n,count,4),bool);candidate_awards=np.zeros_like(original_awards)
    converted={k:torch.from_numpy(data[k]) for k in INPUTS}
    for t in range(n):
        for k,v in vars(old).items():assert np.array_equal(v.numpy(),data['land_before_'+k][t]),(profile,t,k,'before')
        x={k:v[t] for k,v in converted.items()}
        for state,out in ((old,original_awards),(new,candidate_awards)):
            previous=state.highest_credited_z.clone()
            rates=state.update(**x,dt=DT)
            out[t]=((state.highest_credited_z>previous+.008)&~torch.from_numpy(data['land_touch_first'][t])).numpy()
            assert torch.equal(rates,torch.from_numpy(out[t]).float().mean(-1)/DT)
            if state is old:
                assert np.array_equal(rates.numpy(),data['landing_rate'][t])
                assert np.array_equal(out[t],data['land_awarded'][t])
                for k,v in vars(state).items():assert np.array_equal(v.numpy(),data['land_after_'+k][t]),(profile,t,k,'after')
        reset=np.flatnonzero(data['terminated'][t]|data['truncated'][t])
        if len(reset):old.reset(reset);new.reset(reset)
    rows=[];events=[]
    for r in report['records']:
        i=r['env_id'];active=trace['active'][:,i,None]
        a=original_awards[:,i]&active;b=candidate_awards[:,i]&active
        levels=np.rint((data['ground'][:,i]-origins[i,2])/max(r['height'],.01)).astype(int)
        aset={(int(f),int(levels[t,f])) for t,f in np.argwhere(a)}
        bset={(int(f),int(levels[t,f])) for t,f in np.argwhere(b)}
        assert len(aset)==int(a.sum()) and len(bset)==int(b.sum()),'同足同高度重复发奖'
        atimes={(int(f),int(levels[t,f])):int(t) for t,f in np.argwhere(a)}
        btimes={(int(f),int(levels[t,f])):int(t) for t,f in np.argwhere(b)}
        rows.append(dict(profile=profile,env_id=i,height=r['height'],speed=r['speed'],outcome=r['outcome'],
            original_awards=int(a.sum()),candidate_awards=int(b.sum()),delta_awards=int(b.sum()-a.sum()),
            added_foot_levels=len(bset-aset),lost_foot_levels=len(aset-bset),
            shifted_common_award_times=sum(atimes[k]!=btimes[k] for k in aset&bset),
            earlier_common_award_times=sum(btimes[k]<atimes[k] for k in aset&bset),
            original_wall_awards=int((a & data['wall'][:,i]).sum()),candidate_wall_awards=int((b & data['wall'][:,i]).sum())))
        for t,f in np.argwhere(a|b):
            events.append(dict(profile=profile,env_id=i,leg=LEGS[f],time_s=(int(t)+1)*DT,
                height=r['height'],speed=r['speed'],outcome=r['outcome'],level=int(levels[t,f]),
                original=bool(a[t,f]),candidate=bool(b[t,f]),wall=bool(data['wall'][t,i,f])))
    return rows,events,dict(profile=profile,exact_original_state_frames=n*count,
        exact_original_before_and_after=True,exact_original_rates_and_awards=True)


def main():
    rows=[];events=[];checks=[]
    for profile in BASELINES:
        r,e,c=replay(profile);rows+=r;events+=e;checks.append(c)
    write_csv(DOCS/'landing_candidate_episodes.csv',rows);write_csv(DOCS/'landing_candidate_events.csv',events)
    groups=[]
    for profile,height,speed,outcome in sorted({tuple(r[k] for k in ('profile','height','speed','outcome')) for r in rows}):
        chosen=[r for r in rows if (r['profile'],r['height'],r['speed'],r['outcome'])==(profile,height,speed,outcome)]
        groups.append(dict(profile=profile,height=height,speed=speed,outcome=outcome,episodes=len(chosen),
            **{k:sum(r[k] for r in chosen) for k in ('original_awards','candidate_awards','added_foot_levels','lost_foot_levels',
                                                   'shifted_common_award_times','earlier_common_award_times',
                                                   'original_wall_awards','candidate_wall_awards')}))
    write_csv(DOCS/'landing_candidate_groups.csv',groups)
    files=[OUTPUT/p/f for p in BASELINES for f in ('reward_state.npz','shadow_initial.npz','preparation_state.npz','trace.npz','initial.npz','report.json')]
    summary=dict(status='passed',new_simulation_episodes=0,optimizer_updates=0,training_data_created=False,
        checkpoint_sha256=DIGEST,original_peak_margin_m=.015,candidate_peak_margin_m=.005,
        clearance_margin_unchanged_m=.025,checks=checks,
        **{k:sum(r[k] for r in rows) for k in ('original_awards','candidate_awards','added_foot_levels','lost_foot_levels',
                                             'shifted_common_award_times','earlier_common_award_times',
                                             'original_wall_awards','candidate_wall_awards')},
        focus=[r for r in rows if r['env_id'] in FOCUS[r['profile']]],
        sources=[dict(path=str(p),sha256=sha(p)) for p in files],
        implementation_sources=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),Path(__file__).with_name('reference_low_stairs_landing_candidate.py'))],
        limitation='仅比较相同轨迹的发奖事件；未改变动作或证明通关、恢复、转向改善。')
    save(DOCS/'landing_candidate_summary.json',summary)
    print({k:v for k,v in summary.items() if k not in ('sources','implementation_sources')},flush=True)


if __name__=='__main__':main()
