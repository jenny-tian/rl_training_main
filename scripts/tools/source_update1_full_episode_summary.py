"""Recompute and export the approved update1 evaluation from frozen recordings."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
import numpy as np
from source_adaptation_runtime import write_json
from source_full_episode_summary import csv_rows
from rl_training.source_scan.first_episode_comparison import first_episodes,compare_episodes,episode_totals
from rl_training.source_scan.update1_episode_comparison import LABELS,three_actor_windows

REPORT = ROOT/'docs/lite3-source-update1-full-episode-20261105'
BASELINE = ROOT/'docs/lite3-source-full-episode-comparison-20261105'
GATES = dict(max_paired_velocity_rms_error_ratio=1.5,min_paired_reward_ratio=.8,max_clipped_joint_fraction=.01)


def main():
    out,raw = REPORT/'deliverables',REPORT/'raw/runtime'
    live = json.loads((raw/'result.json').read_text())
    old = json.loads((BASELINE/'raw/runtime/result.json').read_text())
    cases = {'original':old['cases'][0],'update1':live['cases'][0],'update2':old['cases'][1]}
    data,episodes = {},{}
    for label in LABELS:
        folder,name = (raw,'update1') if label=='update1' else (BASELINE/'raw/runtime','original' if label=='original' else 'candidate')
        with np.load(folder/(name+'.npz')) as archive:
            data[label] = {key:archive[key] for key in ('done','state','rewards','actions','columns')}
        episodes[label] = first_episodes(data[label],json.loads((folder/(name+'_events.json')).read_text()),
            cases[label]['initial']['assignments'])
        if episodes[label]!=cases[label]['first_episodes']:
            raise ValueError('offline episode accounting differs from capture: '+label)
        csv_rows(out/(label+'_episodes.csv'),episodes[label])
    paired = compare_episodes(episodes['original'],episodes['update1'],data['original'],data['update1'],GATES)
    three = three_actor_windows(episodes,data)
    if paired!=live['comparison'] or three!=live['three_actor_comparison']:
        raise ValueError('offline metrics differ from online finalization')
    write_json(out/'original_update1_comparison.json',paired)
    write_json(out/'three_actor_comparison.json',three)
    csv_rows(out/'original_update1_pairs.csv',paired['pairs'])
    csv_rows(out/'three_actor_pairs.csv',three['pairs'])
    actors = [dict(actor=label,**episode_totals(episodes[label]),**{
        'common_'+key:value for key,value in three['metrics'][label].items()}) for label in LABELS]
    csv_rows(out/'actors.csv',actors)
    conditions = []
    for row in three['conditions']:
        item = dict(column=row['column'],terrain=row['terrain'],environments=row['environments'])
        for label in LABELS:
            item[label+'_completed'] = row['completed'][label]
            item.update({label+'_'+key:value for key,value in row['metrics'][label].items()})
        conditions.append(item)
    csv_rows(out/'conditions.csv',conditions)
    priority_ids = json.loads((REPORT/'authorization.json').read_text())['plan']['priority_environment_ids']
    priorities = []
    all_outcomes = []
    for i,pair in enumerate(three['pairs']):
        item = dict(env_id=i,column=pair['column'],terrain=pair['terrain'],
            common_end_control=pair['end_control'],common_end_time_s=.02*pair['end_control'])
        for label in LABELS:
            ep = episodes[label][i]
            end = data[label]['state'][ep['steps']-1,i]
            item.update({label+'_completed':ep['completed'],label+'_end_control':ep['steps'],
                label+'_end_time_s':ep['seconds'],label+'_reasons':ep['termination_reasons'],
                label+'_terminal_progress_m':float(end[5]),label+'_terminal_lateral_m':float(end[9]),
                label+'_terminal_yaw_deg':float(np.degrees(end[12]))})
            item.update({label+'_common_'+key:value for key,value in pair['endpoints'][label].items()})
        all_outcomes.append(item)
        if i in priority_ids:
            priorities.append(item)
    priorities.sort(key=lambda row:priority_ids.index(row['env_id']))
    csv_rows(out/'episode_outcomes.csv',all_outcomes)
    csv_rows(out/'priority_environments.csv',priorities)
    changed01 = [p['env_id'] for p in paired['pairs'] if p['original_completed']!=p['candidate_completed']]
    changed12 = [p['env_id'] for p in three['pairs'] if p['completed']['update1']!=p['completed']['update2']]
    summary = dict(status=live['status'],offline_equals_online=True,
        execution=json.loads((raw/'execution.json').read_text()),
        controls=cases['update1']['controls'],training_charge=271104,
        checks=paired['checks'],actors={label:episode_totals(episodes[label]) for label in LABELS},
        paired_original_update1_metrics=paired['paired_metrics'],
        paired_original_update1_transitions=paired['paired_controls'],
        paired_original_update1_excluded=paired['excluded_environment_ids'],
        common_three_actor_metrics=three['metrics'],common_three_actor_transitions=three['paired_transitions_per_actor'],
        common_three_actor_excluded=three['excluded_environment_ids'],
        conditions=conditions,priority_environments=priorities,
        changed_original_update1_outcome_environment_ids=changed01,changed_update1_update2_outcome_environment_ids=changed12,
        update1_first_episode_clipped_fraction=paired['candidate_first_episode_clipped_fraction'],
        prefix_checks=cases['update1']['prefix_matches_short_trial'],
        update1_common_lateral_rms_ratio_to_original=three['metrics']['update1']['lateral_rms_m']/three['metrics']['original']['lateral_rms_m'],
        conclusion='Update1 preserves all 120 binary episode outcomes and passes registered gates; outcome regression appears by update2 on this fixed allocation. Update1 already has larger aggregate lateral RMS and earlier endings for three pre-existing stair failures.',
        limitation='One deterministic registered L1 allocation; no claim of all-terrain improvement, cross-seed robustness, unique causal attribution, or permission for further training.')
    write_json(out/'summary.json',summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'original':'#4776a8','update1':'#299665','update2':'#d17b36'}
    titles = {'original':'Original','update1':'Update 1','update2':'Update 2'}
    x = np.arange(len(conditions))
    fig,axes = plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for offset,label in zip((-.25,0,.25),LABELS):
        axes[0].bar(x+offset,[r[label+'_completed']/r['environments'] for r in conditions],width=.25,color=colors[label],label=titles[label])
        axes[1].plot(x,[r[label+'_lateral_rms_m'] for r in conditions],marker='o',color=colors[label],label=titles[label])
    axes[0].set(ylabel='First-episode completion fraction',ylim=(0,1.12),title='Completion on 15 registered L1 conditions')
    axes[1].set(ylabel='World lateral RMS (m)',title='Identical time windows shared by all three actors')
    for ax in axes:
        ax.set_xticks(x,[str(r['column'])+'\nn='+str(r['environments']) for r in conditions])
        ax.set_xlabel('Condition index / assigned environments')
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False)
    fig.savefig(out/'three_actor_conditions.png',dpi=160)
    plt.close(fig)

    names = {6:'Ascending stairs',12:'Left/right friction bands',13:'Front/back friction bands',14:'Alternating friction regions'}
    fig,axes = plt.subplots(4,2,figsize=(12,12),layout='constrained')
    for i,ax in zip(priority_ids,axes.flat):
        for label in LABELS:
            ep = episodes[label][i]
            state = data[label]['state'][:ep['steps'],i]
            ax.plot(state[:,5],state[:,9],color=colors[label],label=titles[label])
            ax.scatter(state[-1,5],state[-1,9],color=colors[label],marker='o' if ep['completed'] else 'x',s=40,zorder=4)
        ax.axvline(9,color='gray',ls=':',alpha=.6)
        ax.set(title=f'Env {i}: {names[episodes["original"][i]["column"]]}',
            xlabel='World forward progress (m)',ylabel='World lateral offset (m)')
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle('First-episode paths: original, update 1 and update 2\n'
        'Circle = completion; cross = bounds ending; x = 9 m also requires 50 stable controls\n'
        'Forward and lateral axes scaled independently',fontsize=11)
    fig.savefig(out/'priority_trajectories.png',dpi=160)
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('status','execution','checks','actors',
        'paired_original_update1_transitions','common_three_actor_transitions',
        'changed_original_update1_outcome_environment_ids','changed_update1_update2_outcome_environment_ids',
        'update1_common_lateral_rms_ratio_to_original')},indent=2))


if __name__=='__main__':
    main()
