"""分析旁路状态，不以公共评估奖励代替实际训练的台阶奖励。"""
from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT,sha,save
from reference_low_stairs_reward_state_probe import OUTPUT,DOCS,BASELINES,CHECKPOINT,DIGEST,read,verify,sources
from reference_low_stairs_reward_observer import LANDING_RULES
from reference_core500_analyze import write_csv

DT=.02
LEGS=('FL','FR','HL','HR')
FOCUS={'nominal':(196,199),'training_randomized':(121,231)}


def npz(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def segments(mask):
    delta=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1)))


def geometry_check(data,trace,initial,config):
    """实测方形环台阶作为独立几何校验，排除距边缘 2.1 mm 内的点。"""
    max_ground=max_obstacle=0.;ground_count=obstacle_count=0
    offsets=np.array([0.,.06,.12,.18,.24],np.float32)
    for column,geom in enumerate(config['geometry']):
        ids=np.array([s['env_id'] for s in config['specs'] if s['column']==column])
        origin=initial['origins'][ids]
        world=data['feet'][:,ids,:,None,:2]+data['direction'][:,ids,None,None,:]*offsets[None,None,None,:,None]
        local=world-origin[None,:,None,None,:2]
        radius=np.abs(local).max(-1)
        edges=np.asarray(geom['fronts'],np.float32)
        tops=np.array([0.]+geom['tops'],np.float32)
        level=(radius[...,None]>=edges).sum(-1)
        expected=tops[level]+origin[None,:,None,None,2]
        away=np.abs(radius[...,None]-edges).min(-1)>.0021 if len(edges) else np.ones(radius.shape,bool)
        active=trace['active'][:,ids,None]
        good_ground=active & away[...,0]
        good_obstacle=active & away.all(-1)
        err_ground=np.abs(data['ground'][:,ids]-expected[...,0])
        err_obstacle=np.abs(data['obstacle'][:,ids]-expected.max(-1))
        max_ground=max(max_ground,float(err_ground[good_ground].max(initial=0.)))
        max_obstacle=max(max_obstacle,float(err_obstacle[good_obstacle].max(initial=0.)))
        ground_count+=int(good_ground.sum());obstacle_count+=int(good_obstacle.sum())
    assert max_ground<1e-4 and max_obstacle<1e-4,(max_ground,max_obstacle)
    return dict(max_ground_error_m=max_ground,max_obstacle_error_m=max_obstacle,
        checked_ground_foot_frames=ground_count,checked_obstacle_foot_frames=obstacle_count)


def episode(record,data,trace,profile):
    i=record['env_id'];n=int(trace['active'][:,i].sum())
    def v(k):return data[k][:n,i]
    support=v('support') & ~v('wall')
    speed=np.linalg.norm(v('foot_velocity')[...,:2],axis=-1)
    raw_higher=v('land_raw_touch') & v('land_rule_higher_than_last') & v('enabled')[:,None]
    achieved=v('bottom')>=v('clear_after_target')+.025
    bad=achieved & (v('clear_after_cost')>1e-9)
    assert not bad.any()
    row=dict(profile=profile,env_id=i,height=record['height'],speed=record['speed'],repeat=record['repeat'],
        outcome=record['outcome'],duration_s=n*DT,best_risers=record['best_risers'],
        clearance_return=float(v('clearance_rate').sum()*DT),progress_return=float(v('progress_rate').sum()*DT),
        landing_return=float(v('landing_rate').sum()*DT),land_awards=int(v('land_awarded').sum()),
        clear_starts=int(v('clear_start').sum()),clear_support_releases=int(v('clear_release_support').sum()),
        clear_lower_releases=int(v('clear_release_lower').sum()),clear_scope_releases=int(v('clear_release_scope').sum()),
        clear_tilt_releases=int(v('clear_release_tilt').sum()),clear_direction_releases=int(v('clear_release_direction').sum()),
        raw_higher_touches=int(raw_higher.sum()),landing_qualifications=int(v('land_qualifies').sum()),
        support_horizontal_foot_speed=float(speed[support].mean()) if support.any() else None,
        support_foot_seconds=float(support.sum()*DT),wall_foot_seconds=float(v('wall').sum()*DT),
        support_wall_overlap_foot_seconds=float((v('wall')&v('support')).sum()*DT),
        awards_while_wall=int((v('land_awarded')&v('wall')).sum()),
        charged_after_sufficient_clearance=int(bad.sum()))
    for key in LANDING_RULES:
        row[f'higher_touch_rejected_{key}']=int((raw_higher & ~v('land_rule_'+key)).sum())
    assert abs(row['landing_return']-row['land_awards']/4)<1e-4
    return row


def clearance_cycles(record,data,trace,profile,origin_z):
    i=record['env_id'];n=int(trace['active'][:,i].sum());rows=[]
    for foot,leg in enumerate(LEGS):
        starts=np.flatnonzero(data['clear_start'][:n,i,foot])
        release=(data['clear_release_support'][:n,i,foot] | data['clear_release_scope'][:n,i,foot]
            | data['clear_release_tilt'][:n,i,foot] | data['clear_release_direction'][:n,i,foot])
        for pos,start in enumerate(starts):
            next_start=int(starts[pos+1]) if pos+1<len(starts) else n
            ends=np.flatnonzero(release[start:next_start])
            end=int(start+ends[0]) if len(ends) else next_start-1
            target=float(data['clear_after_target'][start,i,foot]);bottom=data['bottom'][start:end+1,i,foot]
            reason='still_active_at_end'
            for k in ('support','scope','tilt','direction'):
                if data['clear_release_'+k][end,i,foot]:reason=k;break
            rows.append(dict(profile=profile,env_id=i,height=record['height'],speed=record['speed'],
                outcome=record['outcome'],leg=leg,start_s=(start+1)*DT,end_s=(end+1)*DT,
                duration_s=(end-start+1)*DT,target_z_m=target-origin_z,
                max_bottom_z_m=float(bottom.max())-origin_z,
                required_bottom_z_m=target+.025-origin_z,
                attained_margin=bool((bottom>=target+.025-1e-6).any()),release_reason=reason,
                released_lower=bool(data['clear_release_lower'][end,i,foot]),
                final_ground_z_m=float(data['ground'][end,i,foot])-origin_z,
                wall_seconds=float(data['wall'][start:end+1,i,foot].sum()*DT)))
    return rows


def supported_levels(record,data,trace,profile,origin_z):
    """至少 60 ms 的更高支撑事件；与奖励判据分开列出，仍为力/几何代理。"""
    if record['height']==0.:return []
    i=record['env_id'];n=int(trace['active'][:,i].sum());rows=[]
    support=data['support'][:n,i] & ~data['wall'][:n,i]
    support &= data['upright'][:n,i,None] & (data['support'][:n,i].sum(-1)>=2)[:,None]
    ground=data['ground'][:n,i]-origin_z
    levels=np.rint(ground/record['height']).astype(int)
    for foot,leg in enumerate(LEGS):
        for level in range(1,8):
            spans=segments(support[:,foot] & (levels[:,foot]==level))
            spans=[(start,end) for start,end in spans if end-start>=3]
            if not spans:continue
            start,end=spans[0];event=start+2
            awarded=data['land_awarded'][:n,i,foot] & (levels[:,foot]==level)
            credited_this=bool(awarded[:end].any())
            touches=np.flatnonzero(data['land_raw_touch'][:event+1,i,foot] & (levels[:event+1,foot]==level))
            touch=int(touches[-1]) if len(touches) else None
            row=dict(profile=profile,env_id=i,height=record['height'],speed=record['speed'],outcome=record['outcome'],
                leg=leg,level=level,support_begin_s=(start+1)*DT,verified_s=(event+1)*DT,
                support_end_s=end*DT,credited_by_support_end=credited_this,
                credited_by_episode_end=bool(awarded.any()),
                raw_touch_s=(touch+1)*DT if touch is not None else None,
                raw_touch_lag_from_support_begin_s=(start-touch)*DT if touch is not None else None,
                raw_touch_in_first_span=touch is not None and start<=touch<end,
                air_time_s=float(data['land_airtime_at_check'][touch,i,foot]) if touch is not None else None,
                foot_progress_m=float(data['land_displacement'][touch,i,foot]) if touch is not None else None,
                body_progress_m=float(data['land_body_progress'][touch,i,foot]) if touch is not None else None,
                peak_above_ground_m=float(data['land_peak_at_check'][touch,i,foot]-data['ground'][touch,i,foot]) if touch is not None else None)
            for k in LANDING_RULES:row['rule_'+k]=bool(data['land_rule_'+k][touch,i,foot]) if touch is not None else False
            rows.append(row)
    return rows


def pending_events(record,data,trace,profile):
    """追踪每次资格通过到发奖、取消或回合结束；允许多个取消条件并存。"""
    i=record['env_id'];n=int(trace['active'][:,i].sum());rows=[]
    for foot,leg in enumerate(LEGS):
        for start in np.flatnonzero(data['land_qualifies'][:n,i,foot]):
            end=int(start)
            while end<n-1 and data['land_after_pending'][end,i,foot]:end+=1
            awarded=bool(data['land_awarded'][end,i,foot])
            pending=bool(data['land_after_pending'][end,i,foot])
            reasons=[]
            if not awarded and not pending:
                if not data['support'][end,i,foot]:reasons.append('foot_support_lost')
                if not data['enabled'][end,i]:reasons.append('disabled')
                if not data['upright'][end,i]:reasons.append('tilted')
                if data['support'][end,i].sum()<2:reasons.append('fewer_than_two_supports')
                if abs(data['ground'][end,i,foot]-data['land_before_pending_z'][end,i,foot])>=.005:
                    reasons.append('support_level_changed')
                assert reasons,(profile,i,leg,start,end)
            rows.append(dict(profile=profile,env_id=i,height=record['height'],speed=record['speed'],
                outcome=record['outcome'],leg=leg,touch_s=(start+1)*DT,end_s=(end+1)*DT,
                result='awarded' if awarded else 'episode_end' if pending else 'cancelled',
                reasons='|'.join(reasons),dwell_before_end_s=float(data['land_before_dwell'][end,i,foot])))
    return rows


def common_window_comparisons(report,data,trace,profile):
    """与全部同高度同速度成功起点比较相同时间段，避免存活时长混淆。"""
    rows=[]
    for focus in FOCUS[profile]:
        ref=next(r for r in report['records'] if r['env_id']==focus)
        peers=[r for r in report['records'] if r['height']==ref['height'] and r['speed']==ref['speed']
               and r['outcome']=='success']
        selected=[ref]+peers
        lo=50  # 0.5 s 静止及 0.5 s 加速后，从第 51 步开始。
        hi=min(int(trace['active'][:,r['env_id']].sum()) for r in selected)
        assert hi>lo and peers
        for r in selected:
            i=r['env_id'];mask=data['support'][lo:hi,i] & ~data['wall'][lo:hi,i]
            v=data['foot_velocity'][lo:hi,i];speed=np.linalg.norm(v[...,:2],axis=-1)
            rows.append(dict(profile=profile,focus_id=focus,env_id=i,role='focus' if i==focus else 'success_control',
                height=r['height'],speed=r['speed'],outcome=r['outcome'],start_s=lo*DT,end_s=hi*DT,
                support_foot_seconds=float(mask.sum()*DT),
                support_horizontal_foot_speed=float(speed[mask].mean()),
                support_abs_y_foot_speed=float(np.abs(v[...,1][mask]).mean()),
                support_signed_y_foot_speed=float(v[...,1][mask].mean()),
                root_y_displacement=float(trace['root'][hi-1,i,1]-trace['root'][lo-1,i,1]),
                root_x_displacement=float(trace['root'][hi-1,i,0]-trace['root'][lo-1,i,0]),
                mean_abs_root_vy=float(np.abs(data['velocity'][lo:hi,i,1]).mean()),
                wall_foot_seconds=float(data['wall'][lo:hi,i].sum()*DT),
                lower_releases=int(data['clear_release_lower'][lo:hi,i].sum()),
                land_awards=int(data['land_awarded'][lo:hi,i].sum())))
    return rows


def wall_awards(report,data,trace,profile):
    rows=[]
    for t,i,foot in np.argwhere(data['land_awarded'] & data['wall'] & trace['active'][...,None]):
        r=next(r for r in report['records'] if r['env_id']==i)
        rows.append(dict(profile=profile,env_id=int(i),leg=LEGS[foot],time_s=(int(t)+1)*DT,
            height=r['height'],speed=r['speed'],outcome=r['outcome'],
            vertical_force_n=float(data['force'][t,i,foot,2]),
            bottom_above_ground_m=float(data['bottom'][t,i,foot]-data['ground'][t,i,foot]),
            wall_distance_min_m=float(data['wall_distance'][t,i,foot].min())))
    return rows


def focus_windows(record,data,trace,profile):
    i=record['env_id'];n=int(trace['active'][:,i].sum());rows=[]
    for start in range(0,20,2):
        lo,hi=round(start/DT),min(n,round((start+2)/DT))
        if hi<=lo:continue
        for foot,leg in enumerate(LEGS):
            mask=data['support'][lo:hi,i,foot] & ~data['wall'][lo:hi,i,foot]
            velocity=data['foot_velocity'][lo:hi,i,foot]
            rows.append(dict(profile=profile,env_id=i,leg=leg,start_s=start,end_s=hi*DT,
                x_begin=float(trace['root'][lo,i,0]),x_end=float(trace['root'][hi-1,i,0]),
                y_begin=float(trace['root'][lo,i,1]),y_end=float(trace['root'][hi-1,i,1]),
                starts=int(data['clear_start'][lo:hi,i,foot].sum()),
                lower_releases=int(data['clear_release_lower'][lo:hi,i,foot].sum()),
                wall_s=float(data['wall'][lo:hi,i,foot].sum()*DT),support_s=float(mask.sum()*DT),
                support_horizontal_speed=float(np.linalg.norm(velocity[mask,:2],axis=-1).mean()) if mask.any() else None,
                support_velocity_y=float(velocity[mask,1].mean()) if mask.any() else None,
                clearance_return=float(-data['clear_after_cost'][lo:hi,i,foot].sum()*DT/4),
                land_awards=int(data['land_awarded'][lo:hi,i,foot].sum())))
    return rows


def analyze_profile(profile):
    path=OUTPUT/profile;report=read(path/'report.json');cfg=read(path/'config.json')
    data,trace,initial=npz(path/'reward_state.npz'),npz(path/'trace.npz'),npz(path/'initial.npz')
    check=geometry_check(data,trace,initial,cfg)
    tables={k:[] for k in TABLES}
    for record in report['records']:
        tables['episodes'].append(episode(record,data,trace,profile))
        tables['cycles']+=clearance_cycles(record,data,trace,profile,float(initial['origins'][record['env_id'],2]))
        tables['supported_levels']+=supported_levels(record,data,trace,profile,float(initial['origins'][record['env_id'],2]))
        tables['pending_events']+=pending_events(record,data,trace,profile)
        if record['env_id'] in FOCUS[profile]:tables['focus_windows']+=focus_windows(record,data,trace,profile)
    tables['common_windows']=common_window_comparisons(report,data,trace,profile)
    tables['wall_awards']=wall_awards(report,data,trace,profile)
    return tables,check


TABLES=('episodes','cycles','supported_levels','focus_windows','pending_events','common_windows','wall_awards')


def main():
    complete=read(OUTPUT/'complete.json');assert complete['status']=='passed' and complete['new_episodes']==480
    tables={k:[] for k in TABLES};checks=[]
    for profile in BASELINES:
        result=verify(OUTPUT/profile,profile)
        current,geometry=analyze_profile(profile)
        checks.append(dict(**result,geometry=geometry))
        for k,rows in current.items():tables[k]+=rows
    groups=[]
    for profile,height,speed,outcome in sorted({tuple(r[k] for k in ('profile','height','speed','outcome')) for r in tables['episodes']}):
        selected=[r for r in tables['episodes'] if (r['profile'],r['height'],r['speed'],r['outcome'])==(profile,height,speed,outcome)]
        row=dict(profile=profile,height=height,speed=speed,outcome=outcome,episodes=len(selected))
        for key in ('clearance_return','progress_return','landing_return','land_awards','clear_starts','clear_lower_releases',
                    'raw_higher_touches','landing_qualifications','support_horizontal_foot_speed','wall_foot_seconds'):
            values=[r[key] for r in selected if r[key] is not None]
            row['mean_'+key]=float(np.mean(values)) if values else None
        groups.append(row)
    tables['groups']=groups
    for key,rows in tables.items():write_csv(DOCS/f'{key}.csv',rows)
    lost=[r for r in tables['supported_levels'] if not r['credited_by_support_end']]
    only_peak=[r for r in lost if not r['rule_peak_clear'] and all(r['rule_'+k] for k in LANDING_RULES if k!='peak_clear')]
    common=[]
    for profile,ids in FOCUS.items():
        for i in ids:
            rows=[r for r in tables['common_windows'] if r['profile']==profile and r['focus_id']==i]
            focus=next(r for r in rows if r['role']=='focus');peers=[r for r in rows if r['role']=='success_control']
            row=dict(profile=profile,env_id=i,start_s=focus['start_s'],end_s=focus['end_s'],
                control_ids=[r['env_id'] for r in peers])
            for key in ('support_horizontal_foot_speed','support_abs_y_foot_speed','mean_abs_root_vy',
                        'root_y_displacement','wall_foot_seconds','lower_releases','land_awards'):
                values=[r[key] for r in peers]
                row[key]=dict(focus=focus[key],control_mean=float(np.mean(values)),control_min=min(values),control_max=max(values))
            common.append(row)
    summary=dict(status='passed',scope='轨迹及旁路状态核验；未进行训练或证明修正有效',new_episodes=480,
        optimizer_updates=0,checkpoint_sha256=DIGEST,checks=checks,
        supported_level_events=len(tables['supported_levels']),uncredited_by_first_span_end=len(lost),
        uncredited_by_episode_end=sum(not r['credited_by_episode_end'] for r in tables['supported_levels']),
        first_span_uncredited_only_peak_failed=len(only_peak),
        first_span_only_peak_with_at_least_5mm=sum(r['peak_above_ground_m']>=.005 for r in only_peak),
        uncredited_no_raw_touch=sum(r['raw_touch_s'] is None for r in lost),
        uncredited_rule_failures_at_nearest_touch={k:sum(not r['rule_'+k] for r in lost if r['raw_touch_s'] is not None) for k in LANDING_RULES},
        pending_event_results={key:sum(r['result']==key for r in tables['pending_events']) for key in ('awarded','cancelled','episode_end')},
        focus_pending_cancelled=[r for r in tables['pending_events'] if r['env_id'] in FOCUS[r['profile']] and r['result']!='awarded'],
        wall_award_details=tables['wall_awards'],
        common_window_comparisons=common,
        focus=[r for r in tables['episodes'] if r['env_id'] in FOCUS[r['profile']]],
        charged_after_sufficient_clearance=sum(r['charged_after_sufficient_clearance'] for r in tables['episodes']),
        awards_while_wall=sum(r['awards_while_wall'] for r in tables['episodes']),
        clear_lower_releases=sum(r['clear_lower_releases'] for r in tables['episodes']),
        sources=sources()+[dict(path=str(Path(__file__)),sha256=sha(__file__))])
    for key in ('probe','recovery','candidate'):
        suites=ET.parse(DOCS/f'{key}_tests.xml').getroot().findall('testsuite')
        assert all(int(s.get(k))==0 for s in suites for k in ('errors','failures','skipped'))
        summary[key+'_tests_passed']=sum(int(s.get('tests')) for s in suites)
    save(DOCS/'summary.json',summary)
    raw={}
    for f in OUTPUT.rglob('*'):
        if not f.is_file():continue
        raw[str(f)]=dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size)
        if f.suffix=='.json':
            dest=DOCS/'runs'/f.relative_to(OUTPUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for row in read(OUTPUT/'plan.json')['reused']:
        assert sha(row['path'])==row['sha256'];raw[row['path']]=row
    assert sha(CHECKPOINT)==DIGEST
    raw[str(CHECKPOINT)]=dict(path=str(CHECKPOINT),sha256=DIGEST,bytes=CHECKPOINT.stat().st_size)
    save(DOCS/'raw_manifest.json',list(raw.values()))
    print({k:v for k,v in summary.items() if k not in ('checks','sources','focus')},flush=True)
    print('FOCUS',summary['focus'],flush=True)


if __name__=='__main__':main()
