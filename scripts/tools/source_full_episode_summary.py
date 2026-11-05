"""Export the completed full-first-episode comparison without simulator access."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
import numpy as np
from source_adaptation_runtime import write_json
from rl_training.source_scan.first_episode_comparison import first_episodes, compare_episodes

REPORT = ROOT/'docs/lite3-source-full-episode-comparison-20261105'


def csv_rows(path,rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def failure_analysis(data, episodes, changed, state_names, out):
    """Compare failures at the same control, using only archived first episodes."""
    fields = {name:i for i,name in enumerate(state_names)}
    failures = [p for p in changed if p['original_completed'] and not p['candidate_completed']]
    rows = []

    def snapshot(state):
        x,y,yaw = (float(state[fields[k]]) for k in ('progress_m','lateral_m','yaw_rad'))
        ex = .87*abs(np.cos(yaw))+.57*abs(np.sin(yaw))
        ey = .87*abs(np.sin(yaw))+.57*abs(np.cos(yaw))
        # Reconstruct only the current yaw-aligned rectangle. Full body poses
        # and world-frame prediction inputs were not archived for this stage.
        clearance = min(x-ex+1.4,10.4-x-ex,y-ey+1.9,1.9-y-ey)
        return dict(progress_m=x,lateral_m=y,yaw_deg=float(np.degrees(yaw)),
            body_vx_mps=float(state[fields['vx_mps']]),
            current_scanner_clearance_m=float(clearance))

    for pair in failures:
        i,step = pair['env_id'],pair['candidate_steps']
        if step > pair['original_steps']:
            raise ValueError('original first episode does not cover candidate failure time')
        row = dict(env_id=i,column=pair['column'],terrain=pair['terrain'],
            comparison_control=step,comparison_time_s=step*.02,
            original_completion_control=pair['original_steps'],
            candidate_terminal_hold=episodes['candidate'][i]['hold'],
            candidate_reasons=pair['candidate_reasons'])
        values = {}
        for label in ('original','candidate'):
            values[label] = snapshot(data[label]['state'][step-1,i])
            row.update({label+'_'+key:value for key,value in values[label].items()})
        row['candidate_minus_original_abs_lateral_m'] = (
            abs(values['candidate']['lateral_m'])-abs(values['original']['lateral_m']))
        row['candidate_minus_original_abs_yaw_deg'] = (
            abs(values['candidate']['yaw_deg'])-abs(values['original']['yaw_deg']))
        final = snapshot(data['original']['state'][pair['original_steps']-1,i])
        row.update({'original_completion_'+key:value for key,value in final.items()})
        rows.append(row)
    if not rows:
        return dict(new_failure_count=0,new_failures=[])
    csv_rows(out/'new_failures.csv',rows)
    report = dict(new_failure_count=len(rows),new_failures=rows,
        recovered_environment_ids=[p['env_id'] for p in changed if p['candidate_completed'] and not p['original_completed']],
        matched_time_method='Original and candidate sampled at candidate first-terminal control; original completion endpoint recorded separately.',
        scanner_reconstruction=dict(source='source/rl_training/rl_training/source_scan/scan_footprint.py',
            half_extents_m=[.87,.57],allowed_x_m=[-1.4,10.4],allowed_y_m=[-1.9,1.9],
            uses_current_saved_root_xy_yaw_only=True,
            limitation='Geometric reconstruction, not recorded branch provenance. Body-link guard and one-step prediction cannot be reconstructed exactly from these arrays.'),
        interpretation='All new failures have larger lateral displacement and heading deviation at the same time; body-frame speed tracking does not enforce lane-relative position or heading.',
        causal_limitation='One deterministic registered L1 allocation. This does not establish a unique optimizer, reward, or friction-physics cause.')
    write_json(out/'failure_analysis.json',report)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(len(rows),2,figsize=(12,12),layout='constrained',squeeze=False)
    terrain_titles = {12:'Left/right friction bands',13:'Front/back friction bands',14:'Alternating friction regions'}
    for row,(path_ax,yaw_ax) in zip(rows,axes):
        i,step = row['env_id'],row['comparison_control']
        for label,title,color in [('original','Original','#4776a8'),('candidate','Update 2','#d17b36')]:
            end = episodes[label][i]['steps']
            state = data[label]['state'][:end,i]
            path_ax.plot(state[:,fields['progress_m']],state[:,fields['lateral_m']],color=color,label=title)
            path_ax.scatter(state[-1,fields['progress_m']],state[-1,fields['lateral_m']],color=color,
                marker='o' if label=='original' else 'x',s=35,zorder=3)
            yaw_ax.plot(np.arange(1,end+1)*.02,np.degrees(state[:,fields['yaw_rad']]),color=color,label=title)
        old = data['original']['state'][step-1,i]
        path_ax.scatter(old[fields['progress_m']],old[fields['lateral_m']],facecolors='none',
            edgecolors='#4776a8',s=65,zorder=4)
        path_ax.axvline(9,color='gray',ls=':',alpha=.6)
        path_ax.set(title=f"Env {i}: {terrain_titles[row['column']]}",xlabel='World forward progress (m)',
            ylabel='World lateral offset (m)',xlim=(0,9.6),ylim=(-1.18,.12))
        yaw_ax.axvline(step*.02,color='gray',ls=':',alpha=.6)
        yaw_ax.set(title=f"Candidate bounds ending: {step*.02:.2f} s",xlabel='Time (s)',
            ylabel='Yaw (degrees)',xlim=(0,32),ylim=(-23,3))
        for ax in (path_ax,yaw_ax):
            ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,loc='lower left')
    axes[0,1].legend(frameon=False,loc='lower left')
    fig.suptitle('Five new bounds failures: first-episode trajectories\n'
        'Circle = original completion; cross = candidate ending; open circle = original at candidate ending\n'
        'Left axes scaled independently; x = 9 m also requires 50 stable controls',fontsize=11)
    fig.savefig(out/'new_failure_trajectories.png',dpi=160)
    plt.close(fig)
    return report


def main():
    raw,out = REPORT/'raw/runtime',REPORT/'deliverables'
    out.mkdir(parents=True,exist_ok=True)
    result = json.loads((raw/'result.json').read_text())
    manifest = json.loads((raw/'manifest.json').read_text())
    data = {}
    for label in ('original','candidate'):
        with np.load(raw/(label+'.npz')) as archive:
            data[label] = {key:archive[key] for key in ('done','state','rewards','actions','columns')}
    episodes = {}
    for case in result['cases']:
        label = case['label']
        episodes[label] = first_episodes(data[label],json.loads((raw/(label+'_events.json')).read_text()),
            case['initial']['assignments'])
        if episodes[label] != case['first_episodes']:
            raise ValueError('offline first-episode summary differs from runtime')
    comparison = compare_episodes(episodes['original'],episodes['candidate'],data['original'],data['candidate'],
        manifest['plan']['comparison_gates'])
    if 'comparison' in result and comparison != result['comparison']:
        raise ValueError('offline comparison differs from runtime')
    write_json(out/'comparison.json',comparison)
    for label,rows in episodes.items():
        csv_rows(out/(label+'_episodes.csv'),rows)
    csv_rows(out/'paired_episodes.csv',comparison['pairs'])
    conditions = []
    for row in comparison['conditions']:
        col = row['column']
        record = dict(row)
        pairs = [p for p in comparison['pairs'] if p['column']==col and not p['excluded']]
        n = sum(p['paired_controls'] for p in pairs)
        record['paired_controls'] = n
        for label in ('original','candidate'):
            grouped = [e for e in episodes[label] if e['column']==col]
            record[label+'_completion_fraction'] = record[label+'_completed']/len(grouped)
            record[label+'_paired_error_mps'] = float(np.sqrt(sum(p[label+'_velocity_rms_error_mps']**2*p['paired_controls'] for p in pairs)/n)) if n else None
            record[label+'_paired_reward'] = sum(p[label+'_reward_per_control']*p['paired_controls'] for p in pairs)/n if n else None
            record[label+'_mean_lateral_peak_m'] = float(np.mean([e['lateral_peak_m'] for e in grouped]))
        conditions.append(record)
    csv_rows(out/'conditions.csv',conditions)
    pairs = comparison['pairs']
    changed = [p for p in pairs if p['original_completed']!=p['candidate_completed']]
    failures = failure_analysis(data,episodes,changed,result['state_names'],out)
    summary = dict(status='passed' if comparison['passed'] else 'behavior_regressed',
        runtime_status=result['status'],comparison_computed_offline='comparison' not in result,
        execution=json.loads((raw/'execution.json').read_text()),
        checks=comparison['checks'],original=comparison['original'],candidate=comparison['candidate'],
        paired_metrics=comparison['paired_metrics'],paired_controls=comparison['paired_controls'],
        excluded_environment_ids=comparison['excluded_environment_ids'],conditions=conditions,
        candidate_first_episode_clipped_fraction=comparison['candidate_first_episode_clipped_fraction'],
        changed_outcomes=changed,prefix_checks={c['label']:c['prefix_matches_short_trial'] for c in result['cases']},
        original_historical_completed=87,original_historical_completed_reproduced=comparison['original']['completed']==87,
        controls_by_actor={c['label']:c['controls'] for c in result['cases']},
        new_failure_environment_ids=[r['env_id'] for r in failures['new_failures']],
        failure_analysis='failure_analysis.json',
        limitation='Fixed registered L1 geometries and deterministic initial states; no new training or formal selection qualification.')
    write_json(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','checks','original','candidate','paired_metrics','controls_by_actor']},indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(conditions))
    fig,axes = plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for offset,label,title,color in [(-.18,'original','Original','#4776a8'),(.18,'candidate','Fixed LR, update 2','#d17b36')]:
        axes[0].bar(x+offset,[r[label+'_completion_fraction'] for r in conditions],width=.36,label=title,color=color)
        axes[1].plot(x,[r[label+'_paired_error_mps'] for r in conditions],marker='o',label=title,color=color)
    axes[0].set(ylabel='Completed first episodes / assigned episodes',ylim=(0,1.12),title='Full first-episode completion')
    axes[1].set(ylabel='Velocity RMS error (m/s)',title='Matched valid windows: control 101 through earlier first ending')
    for ax in axes:
        ax.set_xticks(x,[str(r['column'])+'\nn='+str(r['pairs']) for r in conditions])
        ax.set_xlabel('Terrain condition / number of paired environments')
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle('Frozen-policy comparison on the same 15 L1 conditions')
    fig.savefig(out/'condition_comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
