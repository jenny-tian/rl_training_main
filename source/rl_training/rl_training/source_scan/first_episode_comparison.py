"""First-episode accounting and matched time windows for frozen Source actors."""
from collections import Counter
import numpy as np

SAFETY_TERMS = {'source_training_safety', 'illegal_contact', 'bad_orientation_2'}
BOUNDS_TERM = 'source_terrain_tile_bounds'
TIMEOUT_TERM = 'source_training_timeout'


def first_episodes(arrays, events, assignments, *, control_dt=0.02):
    """Terminal samples are included; all samples after the first reset are excluded."""
    done = np.asarray(arrays['done'],dtype=bool)
    state = np.asarray(arrays['state'])
    if done.ndim != 2 or state.shape[:2] != done.shape or len(assignments) != done.shape[1]:
        raise ValueError('episode arrays or assignments have inconsistent dimensions')
    if not all(np.isfinite(arrays[key]).all() for key in ('state','rewards','actions')):
        raise ValueError('episode metrics contain nonfinite data')
    if not done.any(axis=0).all():
        raise ValueError('not all first episodes ended within the bounded horizon')
    ends = done.argmax(axis=0)+1
    terminal = {(row['env_id'],row['control_step']):row for row in events}
    if len(terminal) != len(events):
        raise ValueError('duplicate terminal events')
    result = []
    for env_id,assignment in enumerate(assignments):
        end = int(ends[env_id])
        event = terminal.get((env_id,end))
        if event is None or event['episode_step'] != end:
            raise ValueError('first-episode terminal snapshot is missing or shifted')
        if assignment['env_id'] != env_id or not (arrays['columns'][:end,env_id] == assignment['column']).all():
            raise ValueError('first-episode terrain assignment changed')
        values = state[:end,env_id].astype(np.float64)
        if event['completed'] and (event['hold'] < 50 or values[-1,5] < 9.0):
            raise ValueError('completion lacks progress and stable-hold evidence')
        reasons = event['terms']
        result.append(dict(env_id=env_id,column=assignment['column'],terrain=assignment['subterrain'],
            steps=end,seconds=end*control_dt,completed=bool(event['completed']),hold=event['hold'],
            termination_reasons=reasons,safety_ending=bool(SAFETY_TERMS.intersection(reasons)),
            bounds_ending=BOUNDS_TERM in reasons,timeout=TIMEOUT_TERM in reasons,
            forward_progress_m=float(values[-1,5]),lateral_peak_m=float(np.abs(values[:,9]).max()),
            velocity_rms_error_mps=float(np.sqrt(values[:,3].mean())),vx_mps=float(values[:,0].mean()),
            reward_per_control=float(arrays['rewards'][:end,env_id].astype(np.float64).sum(-1).mean()*control_dt),
            clipped_joint_fraction=float(arrays['actions'][:end,env_id,1].mean())))
    return result


def episode_totals(episodes):
    completed = [row for row in episodes if row['completed']]
    return dict(episodes=len(episodes),completed=len(completed),
        completion_fraction=len(completed)/len(episodes),
        safety_endings=sum(row['safety_ending'] for row in episodes),
        bounds_endings=sum(row['bounds_ending'] for row in episodes),
        timeouts=sum(row['timeout'] for row in episodes),
        termination_counts=dict(Counter(term for row in episodes for term in row['termination_reasons'])),
        completed_time_mean_s=float(np.mean([row['seconds'] for row in completed])) if completed else None)


def compare_episodes(old, new, old_arrays, new_arrays, gates, *, control_dt=0.02, warmup=100):
    if len(old) != len(new) or not old:
        raise ValueError('first-episode pairs are incomplete')
    pairs = []
    windows = {'original':[], 'candidate':[]}
    for a,b in zip(old,new):
        if (a['env_id'],a['column']) != (b['env_id'],b['column']):
            raise ValueError('first-episode pairs use different environments or conditions')
        end = min(a['steps'],b['steps'])
        row = dict(env_id=a['env_id'],column=a['column'],terrain=a['terrain'],
            original_steps=a['steps'],candidate_steps=b['steps'],
            original_completed=a['completed'],candidate_completed=b['completed'],
            original_reasons=a['termination_reasons'],candidate_reasons=b['termination_reasons'],
            paired_controls=max(0,end-warmup),excluded=end<=warmup)
        for label,data in [('original',old_arrays),('candidate',new_arrays)]:
            if end > warmup:
                i = a['env_id']
                error2 = data['state'][warmup:end,i,3].astype(np.float64)
                vx = data['state'][warmup:end,i,0].astype(np.float64)
                reward = data['rewards'][warmup:end,i].astype(np.float64).sum(-1)*control_dt
                clipping = data['actions'][warmup:end,i,1].astype(np.float64)
                value = np.stack((error2,vx,reward,clipping),axis=-1)
                windows[label].append(value)
                row.update({label+'_velocity_rms_error_mps':float(np.sqrt(error2.mean())),
                    label+'_reward_per_control':float(reward.mean()),label+'_vx_mps':float(vx.mean())})
            else:
                row.update({label+'_velocity_rms_error_mps':None,label+'_reward_per_control':None,label+'_vx_mps':None})
        pairs.append(row)
    if not windows['original']:
        raise ValueError('no paired samples remain after warmup')
    metrics = {}
    for label,chunks in windows.items():
        value = np.concatenate(chunks)
        metrics[label] = dict(transitions=len(value),velocity_rms_error_mps=float(np.sqrt(value[:,0].mean())),
            vx_mps=float(value[:,1].mean()),reward_per_control=float(value[:,2].mean()),
            clipped_joint_fraction=float(value[:,3].mean()))
    conditions = []
    for col in sorted({row['column'] for row in old}):
        a = [row for row in old if row['column']==col]
        b = [row for row in new if row['column']==col]
        conditions.append(dict(column=col,terrain=a[0]['terrain'],pairs=len(a),
            original_completed=sum(row['completed'] for row in a),candidate_completed=sum(row['completed'] for row in b),
            original_safety=sum(row['safety_ending'] for row in a),candidate_safety=sum(row['safety_ending'] for row in b),
            original_bounds=sum(row['bounds_ending'] for row in a),candidate_bounds=sum(row['bounds_ending'] for row in b),
            original_timeouts=sum(row['timeout'] for row in a),candidate_timeouts=sum(row['timeout'] for row in b)))
    original,candidate = episode_totals(old),episode_totals(new)
    m0,m1 = metrics['original'],metrics['candidate']
    # Clip over every first-episode sample, including startup and the unpaired tail.
    clip = float(np.concatenate([new_arrays['actions'][:row['steps'],row['env_id'],1] for row in new]).mean())
    checks = dict(total_completion=candidate['completed']>=original['completed'],
        condition_completion=all(row['candidate_completed']>=row['original_completed'] for row in conditions),
        safety=candidate['safety_endings']<=original['safety_endings'],
        bounds=candidate['bounds_endings']<=original['bounds_endings'],
        paired_velocity_error=m1['velocity_rms_error_mps']<=gates['max_paired_velocity_rms_error_ratio']*m0['velocity_rms_error_mps'],
        paired_reward=m1['reward_per_control']>=gates['min_paired_reward_ratio']*m0['reward_per_control'],
        clipping=clip<=gates['max_clipped_joint_fraction'])
    return dict(passed=all(checks.values()),checks=checks,original=original,candidate=candidate,
        paired_metrics=metrics,paired_controls=sum(row['paired_controls'] for row in pairs),
        excluded_environment_ids=[row['env_id'] for row in pairs if row['excluded']],
        candidate_first_episode_clipped_fraction=clip,pairs=pairs,conditions=conditions)
