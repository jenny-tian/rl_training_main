"""Prefix checks and common first-episode windows for three frozen actors."""
import numpy as np

LABELS = ('original', 'update1', 'update2')


def short_prefix_checks(arrays, reference, controls=400):
    result = {}
    for key in ('state', 'columns', 'done', 'rewards'):
        actual, expected = np.asarray(arrays[key]), np.asarray(reference[key])
        if actual.shape[0] < controls or expected.shape[0] != controls:
            raise ValueError('prefix comparison requires a complete registered short recording')
        actual = actual[:controls, :, :9] if key == 'state' else actual[:controls]
        result[key] = np.array_equal(actual, expected)
    return result


def _metrics(values):
    return dict(transitions=len(values), velocity_rms_error_mps=float(np.sqrt(values[:, 0].mean())),
        body_vx_mps=float(values[:, 1].mean()), reward_per_control=float(values[:, 2].mean()),
        lateral_rms_m=float(np.sqrt(values[:, 3].mean())),
        yaw_rms_deg=float(np.degrees(np.sqrt(values[:, 4].mean()))),
        mean_abs_lateral_m=float(values[:, 5].mean()),
        mean_abs_yaw_deg=float(np.degrees(values[:, 6].mean())))


def three_actor_windows(episodes, arrays, *, warmup=100, control_dt=.02):
    """Use exactly the same pre-reset controls for all actors, including terminal samples."""
    if set(episodes) != set(LABELS) or set(arrays) != set(LABELS):
        raise ValueError('three registered actors are required')
    n = len(episodes['original'])
    if not n or any(len(episodes[label]) != n for label in LABELS):
        raise ValueError('three-actor episode tables have different lengths')
    data = {label: {key: np.asarray(arrays[label][key]) for key in ('state', 'rewards')} for label in LABELS}
    chunks = {label: [] for label in LABELS}
    grouped = {}
    pairs = []
    for i in range(n):
        rows = {label: episodes[label][i] for label in LABELS}
        original = rows['original']
        identity = (i, original['column'], original['terrain'])
        if any((row['env_id'], row['column'], row['terrain']) != identity for row in rows.values()):
            raise ValueError('three-actor environments or terrain assignments differ')
        end = min(row['steps'] for row in rows.values())
        pair = dict(env_id=i, column=original['column'], terrain=original['terrain'],
            end_control=end, controls=max(0, end-warmup), excluded=end <= warmup,
            completed={label: row['completed'] for label, row in rows.items()},
            first_ending_controls={label: row['steps'] for label, row in rows.items()})
        if end > warmup:
            pair['metrics'] = {}
            pair['endpoints'] = {}
            group = grouped.setdefault(original['column'], {label: [] for label in LABELS})
            for label in LABELS:
                state = data[label]['state']
                rewards = data[label]['rewards']
                if state.shape[0] < end or state.shape[1] != n or rewards.shape[:2] != state.shape[:2]:
                    raise ValueError('recording does not cover the common first-episode window')
                s = state[warmup:end, i].astype(np.float64)
                r = rewards[warmup:end, i].astype(np.float64).sum(-1)*control_dt
                values = np.stack((s[:, 3], s[:, 0], r, s[:, 9]**2, s[:, 12]**2,
                                   np.abs(s[:, 9]), np.abs(s[:, 12])), axis=-1)
                if not np.isfinite(values).all():
                    raise ValueError('common-window metrics contain nonfinite values')
                chunks[label].append(values)
                group[label].append(values)
                pair['metrics'][label] = _metrics(values)
                pair['endpoints'][label] = dict(progress_m=float(s[-1, 5]), lateral_m=float(s[-1, 9]),
                    yaw_deg=float(np.degrees(s[-1, 12])))
        pairs.append(pair)
    if not chunks['original']:
        raise ValueError('no common samples remain after warmup')
    metrics = {label: _metrics(np.concatenate(chunks[label])) for label in LABELS}
    conditions = []
    for column in sorted({row['column'] for row in episodes['original']}):
        rows = [row for row in episodes['original'] if row['column'] == column]
        condition = dict(column=column, terrain=rows[0]['terrain'], environments=len(rows),
            completed={label: sum(row['completed'] for row in episodes[label] if row['column'] == column)
                       for label in LABELS},
            metrics={label: _metrics(np.concatenate(grouped[column][label])) for label in LABELS}
                    if column in grouped else {})
        conditions.append(condition)
    return dict(window='control 101 through earliest first ending among all three actors' if warmup == 100
                else f'control {warmup+1} through earliest first ending among all three actors',
        paired_transitions_per_actor=sum(pair['controls'] for pair in pairs),
        excluded_environment_ids=[pair['env_id'] for pair in pairs if pair['excluded']],
        metrics=metrics, conditions=conditions, pairs=pairs)
