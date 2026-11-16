"""配对分析冻结航向反馈；保留全部起点的收益、退化和失败类型。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_heading_probe import OUTPUT, DOCS, CHECKPOINT, DIGEST, PROTOCOL, ORIGINAL
from reference_low_stairs_heading_evaluate import read, verify_pair, historical_check, PROBE
from reference_low_stairs_zero_preservation_analyze import paired_records
from reference_low_stairs_equal100_analyze import validate_trace, stagnation
from reference_low_stairs_failure_audit import stair_features, DT
from reference_low_stairs_regression_analyze import yaw, lateral_components
from reference_core500_analyze import write_csv

OUTCOMES = ('success', 'orientation_failure', 'lateral_exit', 'stuck', 'timeout')


def load_trace(path):
    with np.load(path/'trace.npz') as archive:
        data = {k: archive[k] for k in archive.files}
    validate_trace(read(path/'report.json'), data)
    return data


def transition_rows(before, after, profile, direction):
    a, b = paired_records(before, after, 'stairs')
    rows = []
    for i in sorted(a):
        old, new = a[i], b[i]
        rows.append(dict(profile=profile, direction=direction, env_id=i,
            height=old['height'], speed=old['speed'], repeat=old['repeat'], column=old['column'],
            before=old['outcome'], after=new['outcome'],
            gained_success=old['outcome']!='success' and new['outcome']=='success',
            lost_success=old['outcome']=='success' and new['outcome']!='success',
            new_orientation_failure=old['outcome']!='orientation_failure' and new['outcome']=='orientation_failure',
            before_duration_s=old['duration_s'], after_duration_s=new['duration_s'],
            before_best_risers=old['best_risers'], after_best_risers=new['best_risers']))
    return rows


def grouped(rows, keys):
    results = []
    for values in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in keys)==values]
        row = dict(zip(keys, values), starts=len(selected))
        for side in ('before', 'after'):
            counts = Counter(r[side] for r in selected)
            assert set(counts) <= set(OUTCOMES)
            row.update({f'{side}_{name}':counts[name] for name in OUTCOMES})
        for key in ('gained_success', 'lost_success', 'new_orientation_failure'):
            row[key] = sum(r[key] for r in selected)
        assert row['after_success']-row['before_success'] == row['gained_success']-row['lost_success']
        results.append(row)
    return results


def episode_metrics(record, data, cfg, feedback_enabled):
    i = record['env_id']
    n = int(data['active'][:, i].sum())
    root = data['root'][:n, i]
    heading = yaw(data['quat'][:n, i])
    tilt = np.rad2deg(np.arccos(np.clip(-data['gravity'][:n, i, 2], -1., 1.)))
    after_ramp = np.arange(n)*DT >= 1.
    row = dict(profile=cfg['profile'], direction=cfg['direction'], feedback=feedback_enabled,
        **{k:v for k,v in record.items() if k not in ('final_root', 'termination_terms')},
        final_x_m=float(root[-1, 0]), final_y_m=float(root[-1, 1]),
        final_heading_deg=float(np.rad2deg(heading[-1])), max_tilt_deg=float(tilt.max()),
        last5_progress_m=float(root[-1, 0]-root[max(0,n-251), 0]),
        yaw_abs_mean_deg=float(np.abs(np.rad2deg(heading[after_ramp])).mean()) if after_ramp.any() else None,
        vx_mean_m_s=float(data['velocity'][:n, i, 0][after_ramp].mean()) if after_ramp.any() else None,
        collision_foot_seconds=None, wall_foot_seconds=None,
        first_stagnation_window_end_s=None, slow_window_seconds=None, longest_slow_window_run_s=None)
    if cfg['direction']=='up' and record['height']>0:
        first_tilt = np.flatnonzero(tilt>20.)
        cutoff = int(first_tilt[0]) if len(first_tilt) else n
        before_tilt = (np.arange(n)<cutoff) & (root[:,0]>.85)
        features = stair_features(data['feet'][:n,i], data['forces'][:n,i], cfg['geometry'][record['column']])
        row.update(collision_foot_seconds=float(features['collision'][before_tilt].sum()*DT),
            wall_foot_seconds=float(features['wall'][before_tilt].sum()*DT),
            **stagnation(root[:,0], before_tilt))
    return row


def common_windows(pairs, old, new):
    """仅在双方都完成的固定时间窗统计，分母和未纳入数单列。"""
    rows = []
    for start, end in ((1.,3.), (3.,6.), (6.,10.), (10.,20.)):
        lo, hi = round(start/DT), round(end/DT)
        for pair in pairs:
            i = pair['env_id']
            complete = all(d['active'].shape[0]>=hi and d['active'][:hi,i].all() for d in (old,new))
            row = {k:pair[k] for k in ('profile','direction','env_id','height','speed','repeat')}
            row.update(window_start_s=start, window_end_s=end, complete=bool(complete))
            for side, data in (('before',old), ('after',new)):
                row[f'{side}_yaw_abs_mean_deg'] = float(np.abs(np.rad2deg(yaw(data['quat'][lo:hi,i]))).mean()) if complete else None
                row[f'{side}_lateral_abs_mean_m'] = float(np.abs(data['root'][lo:hi,i,1]).mean()) if complete else None
                row[f'{side}_vx_mean_m_s'] = float(data['velocity'][lo:hi,i,0].mean()) if complete else None
            rows.append(row)
    return rows


def timeout_analysis(path, data, record, feedback_enabled):
    cfg = read(path/'config.json'); i=record['env_id']
    n=int(data['active'][:,i].sum());root=data['root'][:n,i];geom=cfg['geometry'][record['column']]
    features=stair_features(data['feet'][:n,i],data['forces'][:n,i],geom)
    heading=np.rad2deg(yaw(data['quat'][:n,i]))
    lateral=lateral_components(data['quat'][:n,i],data['velocity'][:n,i])
    windows=[]
    for start in range(0,20,2):
        lo,hi=round(start/DT),min(n,round((start+2)/DT))
        if hi<=lo:continue
        windows.append(dict(env_id=i,feedback=feedback_enabled,start_s=start,end_s=hi*DT,samples=hi-lo,
            x_begin_m=float(root[lo,0]),x_end_m=float(root[hi-1,0]),
            y_end_m=float(root[hi-1,1]),heading_mean_deg=float(heading[lo:hi].mean()),
            heading_end_deg=float(heading[hi-1]),vx_mean_m_s=float(data['velocity'][lo:hi,i,0].mean()),
            vy_mean_m_s=float(data['velocity'][lo:hi,i,1].mean()),
            forward_contribution_y_m=float(lateral[lo:hi,0].sum()*DT),
            body_lateral_contribution_y_m=float(lateral[lo:hi,1].sum()*DT),
            vertical_contribution_y_m=float(lateral[lo:hi,2].sum()*DT),
            collision_foot_seconds=float(features['collision'][lo:hi].sum()*DT),
            wall_foot_seconds=float(features['wall'][lo:hi].sum()*DT)))
    final=dict(env_id=i,feedback=feedback_enabled,outcome=record['outcome'],duration_s=n*DT,
        root_success_x=geom['root_success_x'],last_edge_x=geom['fronts'][-1],last_top_z=geom['tops'][-1],
        final_root=root[-1].tolist(),final_feet=data['feet'][n-1,i].tolist(),best_risers=record['best_risers'],
        final_heading_deg=float(heading[-1]),final_yaw_command=float(data['command'][n-1,i,2]),
        final_velocity=data['velocity'][n-1,i].tolist(),
        last5_progress_m=float(root[-1,0]-root[max(0,n-251),0]))
    series=dict(time=(np.arange(n)+1)*DT,root=root.copy(),heading=heading,
        vx=data['velocity'][:n,i,0].copy(),collision=features['collision'].sum(-1))
    return windows,final,series


def common_timeout_contacts(old, new, cfg):
    """碰阶比较采用双方首次倾斜 20 度前共同时间及共同到阶区域。"""
    rows=[]
    for i in (196,199):
        cutoffs=[]
        for data in (old,new):
            n=int(data['active'][:,i].sum())
            tilt=np.rad2deg(np.arccos(np.clip(-data['gravity'][:n,i,2],-1.,1.)))
            first=np.flatnonzero(tilt>20.)
            cutoffs.append(int(first[0]) if len(first) else n)
        end=min(cutoffs)
        mask=(old['root'][:end,i,0]>.85)&(new['root'][:end,i,0]>.85)
        row=dict(env_id=i,common_cutoff_s=end*DT,common_exposure_s=float(mask.sum()*DT))
        for side,data in (('before',old),('after',new)):
            features=stair_features(data['feet'][:end,i],data['forces'][:end,i],cfg['geometry'][3])
            for feature in ('collision','wall'):
                row[f'{side}_{feature}_foot_seconds']=float(features[feature][mask].sum()*DT)
        rows.append(row)
    return rows


def failure_lateral(path, data, record, feedback_enabled):
    """剩余失败的横移分解；积分残差显式保留。"""
    i=record['env_id'];n=int(data['active'][:,i].sum())
    with np.load(path/'initial.npz') as initial:
        y0=float(initial['root'][i,1]-initial['origins'][i,1])
    displacement=float(data['root'][n-1,i,1])-y0
    components=lateral_components(data['quat'][:n,i],data['velocity'][:n,i]).sum(0)*DT
    cfg=read(path/'config.json')
    return dict(profile=cfg['profile'],direction=cfg['direction'],feedback=feedback_enabled,
        env_id=i,height=record['height'],speed=record['speed'],outcome=record['outcome'],
        duration_s=n*DT,displacement_y_m=displacement,forward_y_m=float(components[0]),
        body_lateral_y_m=float(components[1]),vertical_y_m=float(components[2]),
        integration_error_m=float(displacement-components.sum()))


def timeout_plot(series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(11,10),sharex=True)
    for col,i in enumerate((196,199)):
        for enabled,label in ((False,'Original command'),(True,'Heading feedback')):
            s=series[i,enabled];t=s['time']
            for row,value in enumerate((s['root'][:,0],s['root'][:,1],s['heading'],s['vx'])):
                axes[row,col].plot(t,value,label=label,linewidth=1.)
        axes[0,col].axhline(series[i,True]['target'],color='gray',linestyle=':',label='Success x threshold')
        axes[0,col].set_title(f'Nominal up / 5 cm / 0.32 m/s / start {i}')
        axes[3,col].set_xlabel('Time after PD preparation (s)')
        for row,label in enumerate(('Forward x (m)','Lateral y (m)','Heading (deg)','Body forward velocity (m/s)')):
            axes[row,col].set_ylabel(label);axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(DOCS/'timeout_trajectories.png',dpi=150)
    fig.savefig(DOCS/'timeout_trajectories.pdf');plt.close(fig)


def main():
    complete=read(OUTPUT/'complete.json')
    assert complete['status']=='passed' and complete['new_episodes']==720 and complete['optimizer_updates']==0
    pairs,episodes,windows,timeout_windows,timeout_finals,checks=[],[],[],[],[],[]
    timeout_series={};raw={};historical=[];common_contacts=[];failure_components=[]
    for entry in complete['matrix']:
        after,before=Path(entry['candidate']),Path(entry['baseline'])
        checks.append(verify_pair(after,entry['profile'],entry['direction'],reused=entry['reused']))
        reports=[read(p/'report.json') for p in (before,after)]
        current=transition_rows(*reports,entry['profile'],entry['direction']);pairs+=current
        old,new=load_trace(before),load_trace(after)
        windows+=common_windows(current,old,new)
        if entry['profile']=='nominal' and entry['direction']=='up':
            common_contacts=common_timeout_contacts(old,new,read(before/'config.json'))
        for enabled,path,report,data in ((False,before,reports[0],old),(True,after,reports[1],new)):
            cfg=read(path/'config.json')
            episodes.extend(episode_metrics(r,data,cfg,enabled) for r in report['records'])
            remaining={r['env_id'] for r in reports[1]['records'] if r['outcome']!='success'}
            failure_components.extend(failure_lateral(path,data,r,enabled)
                for r in report['records'] if r['env_id'] in remaining)
            if entry['profile']=='nominal' and entry['direction']=='up':
                for i in (196,199):
                    record=next(r for r in report['records'] if r['env_id']==i)
                    detail,final,series=timeout_analysis(path,data,record,enabled)
                    timeout_windows+=detail;timeout_finals.append(final)
                    series['target']=final['root_success_x'];timeout_series[i,enabled]=series
            if path==before or entry['reused']:historical.append(path)
            for f in path.iterdir():
                if f.is_file():raw[str(f)]=dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f))
        del old,new
    assert len(pairs)==960 and len(episodes)==1920
    verified_historical=historical_check(historical)
    assert sorted(verified_historical,key=lambda r:r['path'])==sorted(
        read(OUTPUT/'historical_manifest.json'),key=lambda r:r['path'])
    tables=dict(pairs=pairs,changed_outcomes=[r for r in pairs if r['before']!=r['after']],episodes=episodes,
        remaining_failures=[r for r in episodes if r['feedback'] and r['outcome']!='success'],
        groups=grouped(pairs,('profile','direction','height','speed')),
        heights=grouped(pairs,('profile','direction','height')),
        totals=grouped(pairs,('profile','direction')),common_windows=windows,timeout_windows=timeout_windows,
        timeout_common_contacts=common_contacts,failure_lateral_components=failure_components)
    window_groups=[]
    keys=('profile','direction','height','speed','window_start_s','window_end_s')
    for values in sorted({tuple(r[k] for k in keys) for r in windows}):
        selected=[r for r in windows if tuple(r[k] for k in keys)==values];valid=[r for r in selected if r['complete']]
        row=dict(zip(keys,values),starts=len(selected),common_survivors=len(valid),excluded=len(selected)-len(valid))
        for side in ('before','after'):
            for metric in ('yaw_abs_mean_deg','lateral_abs_mean_m','vx_mean_m_s'):
                k=f'{side}_{metric}';row[k]=float(np.mean([r[k] for r in valid])) if valid else None
        window_groups.append(row)
    tables['common_window_groups']=window_groups
    for name,rows in tables.items():write_csv(DOCS/f'{name}.csv',rows)
    timeout_plot(timeout_series)
    save(DOCS/'timeout_details.json',timeout_finals)
    for f in OUTPUT.rglob('*.json'):
        target=DOCS/'runs'/f.relative_to(OUTPUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    for f in OUTPUT.rglob('*'):
        if f.is_file():raw[str(f)]=dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f))
    assert sha(CHECKPOINT)==DIGEST
    raw[str(CHECKPOINT)]=dict(path=str(CHECKPOINT),bytes=CHECKPOINT.stat().st_size,sha256=DIGEST)
    suites=ET.parse(DOCS/'tests.xml').getroot().findall('testsuite')
    assert all(int(s.get(k))==0 for s in suites for k in ('errors','failures','skipped'))
    sources=[Path(__file__),PROBE,Path(__file__).with_name('reference_low_stairs_heading_evaluate.py'),PROTOCOL,ORIGINAL]
    save(DOCS/'summary.json',dict(status='passed',status_scope='完整性核验通过，不表示部署验收通过',
        new_episodes=720,reused_heading_episodes=240,reused_baseline_episodes=960,optimizer_updates=0,
        checkpoint_sha256=DIGEST,totals=tables['totals'],heights=tables['heights'],
        gained_success=sum(r['gained_success'] for r in pairs),lost_success=sum(r['lost_success'] for r in pairs),
        new_orientation_failures=sum(r['new_orientation_failure'] for r in pairs),
        recovery57='本阶段未重跑且未修复，零命令 0.82 秒姿态失败仍保留',
        checks=checks,tests_passed=sum(int(s.get('tests')) for s in suites),
        elapsed_simulation_s=sum(read(Path(r['candidate'])/'execution.json')['elapsed_s'] for r in complete['matrix'] if not r['reused']),
        implementation_sources=[dict(path=str(p),sha256=sha(p)) for p in sources]))
    save(DOCS/'raw_manifest.json',list(raw.values()))
    for row in tables['totals']:print(row,flush=True)
    print('HEADING ANALYSIS COMPLETE',flush=True)


if __name__=='__main__':
    main()
