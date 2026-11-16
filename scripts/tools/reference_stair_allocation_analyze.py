"""Analyze only recorded allocation-trial trajectories; never launch simulation."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from reference_stair_allocation import OUTPUT, DOCS, PARENT, DIGEST, GROUPS
from reference_stair_allocation_train import directory
from reference_stair_allocation_evaluate import EVALUATION, read, checked_training
from reference_stairs_eval import save, sha, MODELS, load_actor
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_low_stairs_failure_audit import stair_features, DT, LEGS
from reference_low_stairs_heading_analyze import episode_metrics, transition_rows, grouped
from reference_low_stairs_equal100_analyze import validate_trace, heading_error
from reference_low_stairs_regression_analyze import yaw
from reference_core500_analyze import write_csv, paired_turn_rows
from reference_turn_analyze import analyze as analyze_turn


def load(path):
    with np.load(path/'trace.npz') as f:
        data = {k:f[k] for k in f.files}
    validate_trace(read(path/'report.json'), data)
    return data


def training_scalars(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(path/'training'),size_guidance={'scalars':0})
    events.Reload()
    tags = ('Train/mean_reward','Train/mean_episode_length','Episode_Termination/bad_orientation_2',
            'Episode_Termination/time_out','Episode_Termination/terrain_out_of_bounds',
            'Metrics/base_velocity/error_vel_xy','Metrics/base_velocity/error_vel_yaw',
            'Episode_Reward/stair_progress','Episode_Reward/stair_landing','Episode_Reward/stair_clearance')
    result = {}
    for tag in tags:
        values = events.Scalars(tag)
        assert len(values)==100 and [v.step for v in values]==list(range(10100,10200))
        array = np.asarray([v.value for v in values]); assert np.isfinite(array).all()
        result[tag] = dict(first20_mean=float(array[:20].mean()),last20_mean=float(array[-20:].mean()),
            overall_mean=float(array.mean()),last=float(array[-1]))
    return result


def crossing_events(record, data, cfg):
    """First central +x crossing per foot/edge; 20 ms linear interpolation only."""
    i = record['env_id']; n = int(data['active'][:,i].sum())
    feet = data['feet'][:n,i]; forces = data['forces'][:n,i]
    geometry = cfg['geometry'][record['column']]
    features = stair_features(feet, forces, geometry)
    rows = []
    for foot, leg in enumerate(LEGS):
        for level, (edge, top) in enumerate(zip(geometry['fronts'], geometry['tops']), 1):
            indices = np.flatnonzero((feet[:-1,foot,0] < edge) & (feet[1:,foot,0] >= edge)
                & (feet[1:,foot,0] > np.abs(feet[1:,foot,1])+.04)) + 1
            if not len(indices): continue
            t = int(indices[0])
            fraction = (edge-feet[t-1,foot,0])/(feet[t,foot,0]-feet[t-1,foot,0])
            z = feet[t-1,foot,2] + fraction*(feet[t,foot,2]-feet[t-1,foot,2]) - .022
            support = features['support'][:,foot] & ~features['wall'][:,foot]
            landing = (support & (np.abs(features['ground'][:,foot]-top) < .005)
                & (feet[:,foot,0] >= edge+.022) & (np.arange(n) >= t))
            sustained = np.flatnonzero(np.convolve(landing.astype(int), np.ones(3,dtype=int), mode='valid') == 3) if n >= 3 else []
            rows.append(dict(env_id=i, height=record['height'], speed=record['speed'], outcome=record['outcome'],
                leg=leg, level=level, edge_x_m=edge, upper_surface_m=top,
                crossing_time_s=(t+fraction)*DT, interpolated_bottom_above_top_m=float(z-top),
                adjacent_min_bottom_above_top_m=float(min(feet[t-1,foot,2],feet[t,foot,2])-.022-top),
                stable_landing_end_s=float((sustained[0]+3)*DT) if len(sustained) else None))
    return rows


def foot_metrics(record, data, cfg):
    i = record['env_id']; n = int(data['active'][:,i].sum())
    geometry = cfg['geometry'][record['column']]
    feet = data['feet'][:n,i]; forces = data['forces'][:n,i]
    features = stair_features(feet, forces, geometry)
    levels = (np.abs(feet[...,:2]).max(-1)[...,None] >= np.asarray(geometry['fronts'])).sum(-1)
    support = features['support'] & ~features['wall']
    stable = []
    for foot in range(4):
        best = -1
        for level in range(len(geometry['fronts'])+1):
            mask = support[:,foot] & (levels[:,foot] == level)
            if n >= 3 and (np.convolve(mask.astype(int), np.ones(3,dtype=int), mode='valid') == 3).any(): best = level
        stable.append(best)
    power = np.abs(data['torque'][:n,i]*data['joint_vel'][:n,i]).sum(-1)
    return dict(env_id=i, final_hind_min_x_m=float(feet[-1,2:,0].min()),
        final_front_min_x_m=float(feet[-1,:2,0].min()),
        mean_absolute_joint_power_w=float(power.mean()), absolute_joint_work_j=float(power.sum()*DT),
        maximum_abs_torque_nm=float(np.abs(data['torque'][:n,i]).max()),
        **{f'{leg}_max_stably_supported_level':stable[k] for k,leg in enumerate(LEGS)})


def detail_windows(name, paths, traces, ids):
    """Separate stopped stepping, rear-foot delay and lateral loss of posture."""
    rows, common = [], []
    for i in ids:
        series = {}
        for policy,path in paths.items():
            cfg = read(path/'config.json'); r = read(path/'report.json')['records'][i]
            d = traces[policy]; n = int(d['active'][:,i].sum())
            f = stair_features(d['feet'][:n,i],d['forces'][:n,i],cfg['geometry'][r['column']])
            support = f['support'] & ~f['wall']
            angle = np.rad2deg(np.arccos(np.clip(-d['gravity'][:n,i,2],-1,1)))
            cutoff = np.flatnonzero(angle>20.)
            series[policy] = dict(n=n,cutoff=int(cutoff[0]) if len(cutoff) else n,features=f,data=d)
            foot_speed = np.linalg.norm(np.diff(d['feet'][:n,i],axis=0),axis=-1)/DT
            for label,lo,hi in [('first_2s',0,min(n,100)),('last_5s',max(0,n-250),n)]:
                speed = foot_speed[lo:max(lo,hi-1)]
                row = dict(name=name,policy=policy,env_id=i,outcome=r['outcome'],window=label,window_start_s=lo*DT,
                    window_end_s=hi*DT,root_x_progress_m=float(d['root'][hi-1,i,0]-d['root'][lo,i,0]),
                    root_y_progress_m=float(d['root'][hi-1,i,1]-d['root'][lo,i,1]),
                    mean_body_vx_m_s=float(d['velocity'][lo:hi,i,0].mean()),
                    mean_abs_body_vy_m_s=float(np.abs(d['velocity'][lo:hi,i,1]).mean()),
                    mean_command_vx=float(d['command'][lo:hi,i,0].mean()),
                    mean_action_temporal_std=float(d['action'][lo:hi,i].std(axis=0).mean()),
                    four_feet_supported_fraction=float(support[lo:hi].all(-1).mean()),
                    max_tilt_deg=float(angle[lo:hi].max()),
                    first_tilt20_s=(int(cutoff[0])+1)*DT if len(cutoff) else None)
                for foot,leg in enumerate(LEGS):
                    row[f'{leg}_mean_foot_speed_m_s'] = float(speed[:,foot].mean()) if len(speed) else None
                    row[f'{leg}_support_fraction'] = float(support[lo:hi,foot].mean())
                    row[f'{leg}_wall_seconds'] = float(f['wall'][lo:hi,foot].sum()*DT)
                rows.append(row)
        for comparator in ('zero10100','A'):
            a,b = series[comparator],series['B']; end = min(a['cutoff'],b['cutoff'])
            mask = (a['data']['root'][:end,i,0]>.85) & (b['data']['root'][:end,i,0]>.85)
            row = dict(name=name,env_id=i,comparator=comparator,common_end_s=end*DT,
                common_exposure_s=float(mask.sum()*DT))
            for policy,s in [('before',a),('B',b)]:
                for key in ('wall','collision'):
                    row[f'{policy}_{key}_foot_seconds'] = float(s['features'][key][:end][mask].sum()*DT)
            common.append(row)
    return rows, common


def actor_adapter(checkpoint, initial):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from reference_low_stairs_analyze import read_agent
    torch.set_num_threads(1)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with np.load(initial/'initial.npz') as f:
        inputs = torch.tensor(f['observation'])
    obs = TensorDict({'policy':inputs}, batch_size=[len(inputs)])
    native = MLPModel(obs, {'actor':['policy']}, 'actor', 12, **read_agent()).eval()
    native.load_state_dict(state['actor_state_dict'], strict=True)
    MODELS['allocation_adapter'] = checkpoint
    adapter = load_actor('allocation_adapter')
    with torch.inference_mode(): error = float((native(obs)-adapter(inputs)).abs().max())
    assert error == 0.
    return dict(status='passed', max_output_error=error, samples=len(inputs))


def retention_metrics(complete, out):
    rows = []
    for item in complete.get('retention',{}).get('matrix',[]):
        path = Path(item['path']); report = read(path/'report.json')
        with np.load(path/'trace.npz') as f:
            data = {k:f[k] for k in f.files}
        for record in report['records']:
            i = record['env_id']; n = int(data['active'][:,i].sum())
            assert n > 0 and data['active'][:n,i].all() and not data['active'][n:,i].any()
            assert abs(n*DT-record['duration_s']) < 1e-5
            valid = np.arange(n)*DT >= 1.
            vx_error = np.abs(data['velocity'][:n,i,0]-data['command'][:n,i,0])
            tilt = np.rad2deg(np.arccos(np.clip(-data['gravity'][:n,i,2],-1,1)))
            rows.append(dict(model=item['model'],profile=item['profile'],**record,
                vx_mae_after_ramp=float(vx_error[valid].mean()) if valid.any() else None,
                max_tilt_deg=float(tilt.max())))
    if rows: write_csv(out/'retention_episodes.csv',rows)
    groups = []
    for model,profile,terrain,level in sorted({(r['model'],r['profile'],r['terrain'],r['row']) for r in rows}):
        selected = [r for r in rows if (r['model'],r['profile'],r['terrain'],r['row']) == (model,profile,terrain,level)]
        errors = [r['vx_mae_after_ramp'] for r in selected if r['vx_mae_after_ramp'] is not None]
        groups.append(dict(model=model,profile=profile,terrain=terrain,row=level,episodes=len(selected),
            success=sum(r['outcome']=='success' for r in selected),
            orientation_failures=sum(r['outcome']=='orientation_failure' for r in selected),
            mean_forward_displacement_m=float(np.mean([r['displacement'][0] for r in selected])),
            vx_mae_after_ramp=float(np.mean(errors)) if errors else None))
    if groups: write_csv(out/'retention_groups.csv',groups)
    return groups


def plot_case(name, env_id, paths, traces, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,3,figsize=(15,10),layout='constrained')
    for col, (policy, path) in enumerate(paths.items()):
        data = traces[policy]; cfg = read(path/'config.json'); r = read(path/'report.json')['records'][env_id]
        n = int(data['active'][:,env_id].sum()); t = (np.arange(n)+1)*DT
        geom = cfg['geometry'][r['column']]; feet = data['feet'][:n,env_id]
        edges, tops = geom['fronts'], geom['tops']
        xx = [0.] + edges + [3.95]; yy = [0.,0.] + tops
        axes[0,col].step(xx,yy,where='pre',color='black',label='Stair surface')
        for foot, leg in enumerate(LEGS):
            axes[0,col].plot(feet[:,foot,0],feet[:,foot,2]-.022,lw=.7,alpha=.8,label=leg+' bottom')
            axes[0,col].scatter(feet[-1,foot,0],feet[-1,foot,2]-.022,s=24)
        axes[0,col].plot(data['root'][:n,env_id,0],data['root'][:n,env_id,2],color='gray',ls='--',label='Root')
        axes[0,col].set(title=f'{policy}: {r["outcome"]}, {r["duration_s"]:.2f} s',xlabel='Forward x (m)',ylabel='Height relative to tile origin (m)',xlim=(0,4.))
        axes[0,col].legend(fontsize=7,ncol=2)
        axes[1,col].plot(t,data['root'][:n,env_id,0],label='Root x')
        axes[1,col].plot(t,feet[:,2:,0].min(-1),label='Rear-foot minimum x')
        axes[1,col].plot(t,data['root'][:n,env_id,1],label='Root y')
        axes[1,col].axhline(geom['root_success_x'],ls=':',color='gray',label='Root success threshold')
        axes[1,col].set(xlabel='Time (s)',ylabel='Position (m)',xlim=(0,20)); axes[1,col].legend(fontsize=8)
        tilt = np.rad2deg(np.arccos(np.clip(-data['gravity'][:n,env_id,2],-1,1)))
        axes[2,col].plot(t,tilt,label='Body tilt')
        axes[2,col].plot(t,np.rad2deg(yaw(data['quat'][:n,env_id])),label='Heading')
        axes[2,col].set(xlabel='Time (s)',ylabel='Angle (deg)',xlim=(0,20)); axes[2,col].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle(f'{name}, fixed start {env_id}: recorded foot/root trajectories; not simulator rendering\n20 ms samples; endpoint dots mark each first-episode outcome')
    fig.savefig(output/f'{name}_start{env_id}.png',dpi=140)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = read(EVALUATION/'complete.json')
    matrix = complete['matrix']
    out = DOCS/'results'; out.mkdir(parents=True,exist_ok=True)
    checkpoints = {v:checked_training(v) for v in ('A','B')}
    training = {}
    for variant in ('A','B'):
        path = directory(variant,1); r = read(path/'report.json'); e = read(path/'execution.json')
        coverage = np.asarray(r['coverage_by_column_level'])
        assert coverage.sum() == r['interactions'] == 9830400
        by_family = {k:int(coverage[lo:hi].sum()) for k,(lo,hi) in GROUPS.items()}
        assert all(by_family[k] == r['initial_counts'][k]*2400 for k in by_family)
        training[variant] = dict(updates=r['optimizer_updates'], steps=r['environment_steps'],
            interactions=r['interactions'], elapsed_s=e['elapsed_s'], peak_gpu_used_mib=e['peak_total_gpu_used_mib'],
            initial_counts=r['initial_counts'], final_counts=r['final_counts'], coverage=by_family,
            initial_learning_rate=r['initial_learning_rate'], final_learning_rate=r['final_learning_rate'],
            terminated=r['terminated'], truncated=r['truncated'],
            termination_fraction=r['terminated']/max(1,r['terminated']+r['truncated']),
            initial_level_histogram=r['initial_level_histogram'],final_level_histogram=r['final_level_histogram'],
            checkpoint=r['output_checkpoint'],sha256=r['output_sha256'],logged_scalars=training_scalars(path))
        for filename in ('report.json','execution.json','restore.json','configuration.json','allocation_migration.json','geometry.json','simulation_audit.json'):
            shutil.copy2(path/filename,out/f'{variant}_{filename}')
    episodes, crossings, transitions, summaries, turns, turn_windows, turn_pairs = [], [], [], [], [], [], []
    checks, visuals, raw = [], [], set()
    focus_windows, common_contacts = [], []
    by_name = {}
    for entry in matrix: by_name.setdefault(entry['name'],{})[entry['variant']] = entry
    for name, entries in by_name.items():
        assert set(entries) == {'A','B'}
        entry = entries['A']
        paths = {'zero10100':Path(entry['baseline']), **{v:Path(entries[v]['path']) for v in ('A','B')}}
        reports = {v:read(p/'report.json') for v,p in paths.items()}
        for v,p in paths.items():
            assert same_initial(p,paths['zero10100'])
            assert same_configuration(read(p/'config.json'),read(paths['zero10100']/'config.json'))
            assert read(p/'execution.json')['status'] == 'completed' and reports[v]['actor_unchanged']
            raw.update(f for f in p.iterdir() if f.is_file())
            checks.append(dict(name=name,policy=v,initial_exact=True,configuration_exact=True,episodes=reports[v]['episodes']))
            for filename in ('report.json','execution.json','config.json'):
                shutil.copy2(p/filename,out/f'{name}_{v}_{filename}')
        if entry['kind'] == 'turn':
            windows = {}
            for v,p in paths.items():
                _, _, rows, groups = analyze_turn(p)
                windows[v] = rows
                turns.extend(dict(policy=v,name=name,**r) for r in groups)
                turn_windows.extend(dict(policy=v,name=name,**r) for r in rows)
            for comparator in ('zero10100','A'):
                turn_pairs.extend(dict(comparator=comparator,**r) for r in paired_turn_rows(windows[comparator],windows['B'],entry['profile']))
            continue
        traces = {v:load(p) for v,p in paths.items()}
        for v,p in paths.items():
            cfg = read(p/'config.json')
            if entry['heading']: heading_error(p,traces[v])
            for record in reports[v]['records']:
                row = episode_metrics(record,traces[v],cfg,entry['heading'])
                row.update(foot_metrics(record,traces[v],cfg))
                episodes.append(dict(name=name,policy=v,**row))
                if record['height'] and entry['direction'] == 'up':
                    crossings.extend(dict(name=name,policy=v,**r) for r in crossing_events(record,traces[v],cfg))
            counts = Counter(r['outcome'] for r in reports[v]['records'])
            positive = [r for r in reports[v]['records'] if r['height'] > 0]
            summaries.append(dict(name=name,policy=v,starts=240,success=counts['success'],
                orientation_failure=counts['orientation_failure'],lateral_exit=counts['lateral_exit'],
                timeout=counts['timeout'],stuck=counts['stuck'],positive_stair_starts=len(positive),
                positive_stair_success=sum(r['outcome']=='success' for r in positive)))
        for comparator in ('zero10100','A'):
            paired = transition_rows(reports[comparator],reports['B'],entry['profile'],entry['direction'])
            transitions.extend(dict(name=name,comparator=comparator,**r) for r in paired)
        if entry['direction'] == 'up':
            focus = [196,199] if entry['profile'] == 'nominal' else [121,231]
            lost = sorted({r['env_id'] for r in transitions if r['name']==name and r['lost_success']})
            ids = sorted(set(focus+lost[:4]))
            for i in ids:
                plot_case(name,i,paths,traces,out)
                visuals.append(dict(name=name,env_id=i,path=str(out/f'{name}_start{i}.png'),
                    selection='preselected old failures plus first four sorted lost-success IDs', simulator_rendering=False))
            if entry['profile']=='nominal':
                detail, common = detail_windows(name,paths,traces,ids)
                focus_windows.extend(detail); common_contacts.extend(common)
        del traces
    for name, rows in [('stairs_episodes',episodes),('crossings',crossings),('transitions',transitions),
        ('stairs_summary',summaries),('turn_groups',turns),('turn_windows',turn_windows),('turn_pairs',turn_pairs),
        ('focus_windows',focus_windows),('common_contacts',common_contacts)]:
        if rows: write_csv(out/f'{name}.csv',rows)
    groups = []
    for name in by_name:
        for comparator in ('zero10100','A'):
            selected = [r for r in transitions if r['name']==name and r['comparator']==comparator]
            if selected: groups.extend(dict(name=name,comparator=comparator,**g) for g in grouped(selected,('height','speed')))
    if groups: write_csv(out/'stairs_groups.csv',groups)
    first = next(iter(by_name))
    adapters = {v:actor_adapter(p,EVALUATION/v/first) for v,(p,h) in checkpoints.items()}
    integration_interactions = sum(read(directory(v,0)/'report.json')['interactions'] for v in ('A','B'))
    evaluated = sum(r['episodes'] for r in matrix) + complete['retention_episodes']
    assert evaluated <= 6080 and integration_interactions == 524288 and sha(PARENT) == DIGEST
    retention = retention_metrics(complete,out)
    save(out/'summary.json',dict(status=complete['status'], promoted=False, training=training,
        new_optimizer_updates=200, integration_interactions=integration_interactions,
        total_training_and_integration_interactions=20185088, new_evaluation_episodes=evaluated,
        core_episode_limit=5120, retention_episode_limit=960, evaluation_limit=6080,
        gate=complete.get('gate'), regressions=complete.get('regressions',[]),
        not_run=complete.get('not_run',[]), retention_not_run=complete.get('retention_not_run',False),
        stairs=summaries, groups=groups, retention=retention, adapters=adapters, pair_checks=checks, visuals=visuals,
        limitations=['Single training seed, 100-update screen, reused development starts.',
            'Collision and support metrics are geometry/force proxies, not contact manifold measurements.',
            'Crossing clearance is interpolated from 20 ms frames, not a continuous minimum.',
            'Images reconstruct recorded foot/root paths; they are not simulator screenshots.']))
    raw.update(p for v in ('A','B') for p in directory(v,1).rglob('*') if p.is_file())
    raw.update(p for v in ('A','B') for p in directory(v,0).rglob('*') if p.is_file())
    raw.update(p for p in EVALUATION.iterdir() if p.is_file())
    save(out/'raw_manifest.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(raw)])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for v in ('A','B'):
        updates=read(directory(v,1)/'report.json')['updates']
        axes[0].plot([r['update'] for r in updates],[r['mean_terrain_level'] for r in updates],label=v)
        axes[1].plot([r['update'] for r in updates],[r['learning_rate'] for r in updates],label=v)
    for ax in axes: ax.set_xlabel('Additional PPO updates'); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean terrain level'); axes[1].set_ylabel('Learning rate')
    fig.savefig(out/'training.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(status=complete['status'],episodes=evaluated,stairs=summaries),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
