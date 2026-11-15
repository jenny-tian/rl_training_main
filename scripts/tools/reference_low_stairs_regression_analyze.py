"""按首次回合分析站立退化和台阶侧偏，保留竞争终止与时间窗限制。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_regression_probe import OUTPUT
from reference_low_stairs_evaluate import OUTPUT as BASELINE
from reference_low_stairs_clearance_evaluate import OUTPUT as CANDIDATE
from reference_low_stairs_failure_audit import stair_features
from reference_core500_analyze import write_csv

OUT = ROOT / 'docs/lite3-low-stairs-regression-preparation-20261115'
DT = .02


def read(path):
    return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


def first_time(mask):
    indices = np.flatnonzero(mask)
    return float((indices[0] + 1) * DT) if len(indices) else None


def yaw(quaternion):
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def lateral_components(quaternion, velocity):
    """世界 y 速度由机身 x、y、z 分量分别贡献；包含滚转和俯仰。"""
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    rotation_y = np.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), -1)
    return rotation_y * velocity[..., :3]


def verify_replay(probe, historical):
    assert read(probe / 'execution.json')['status'] == 'completed'
    assert read(probe / 'diagnostic_report.json')['optimizer_updates'] == 0
    a, b = npz(probe / 'initial.npz'), npz(historical / 'initial.npz')
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    a, b = npz(probe / 'trace.npz'), npz(historical / 'trace.npz')
    # 固定步数探针最后一帧会将存活记作 survived；物理和已有终止标记不变。
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key][:len(a[key])]), key
    return dict(status='passed', run=str(probe), original=str(historical),
                steps=len(a['root']), initial_exact=True, trace_exact=True)


def stair_rows(directory, policy):
    report, cfg, data = read(directory / 'report.json'), read(directory / 'config.json'), npz(directory / 'trace.npz')
    initial = npz(directory / 'initial.npz')
    rows = []
    for record in report['records']:
        if record['height'] != .05:
            continue
        i = record['env_id']; n = int(data['active'][:, i].sum())
        q, vel = data['quat'][:n, i], data['velocity'][:n, i]
        heading = yaw(q); heading0 = yaw(initial['root'][i, 3:7])
        delta = np.remainder(heading-heading0+np.pi, 2*np.pi)-np.pi
        root = data['root'][:n, i]; y0 = initial['root'][i, 1] - initial['origins'][i, 1]
        features = stair_features(data['feet'][:n, i], data['forces'][:n, i], cfg['geometry'][3])
        parts = lateral_components(q, vel).sum(0) * DT
        row = dict(policy=policy, profile=report['profile'], env_id=i, speed=record['speed'],
            outcome=record['outcome'], duration_s=n*DT, best_risers=record['best_risers'],
            first_contact_s=first_time(features['collision'].any(-1)),
            first_yaw10_s=first_time(np.abs(delta) > np.deg2rad(10)),
            first_y20cm_s=first_time(np.abs(root[:, 1]-y0) > .2),
            final_heading_deg=float(np.rad2deg(heading[-1])),
            displacement_y=float(root[-1, 1]-y0),
            forward_contribution_y=float(parts[0]), slip_contribution_y=float(parts[1]),
            vertical_contribution_y=float(parts[2]), integration_error_y=float(root[-1, 1]-y0-parts.sum()))
        for seconds in (2, 4, 6, 8):
            k = int(seconds / DT)
            row[f'yaw_abs_{seconds}s'] = float(np.rad2deg(np.abs(delta[k-1]))) if n >= k else None
            row[f'drift_abs_{seconds}s'] = float(abs(root[k-1, 1]-y0)) if n >= k else None
        rows.append(row)
    return rows


def stand_rows(directory, policy):
    report, data, initial = read(directory / 'report.json'), npz(directory / 'diagnostic.npz'), npz(directory / 'initial.npz')
    physics = npz(directory / 'physics_initial.npz')
    names = read(directory / 'names.json')
    rows = []
    for record in report['records']:
        i = record['env_id']; n = min(100, round(record['duration_s'] / DT))
        tilt = np.rad2deg(np.arccos(np.clip(-data['gravity'][:n, i, 2], -1, 1)))
        support = data['force'][:n, i, :, 2] > 1
        q0 = initial['root'][i, 3:7]
        tilt0 = np.rad2deg(np.arccos(np.clip(1-2*(q0[1]**2+q0[2]**2), -1, 1)))
        action_delta = data['candidate_action'][0, i] - data['parent_action'][0, i]
        scales = np.tile([.125, .25, .25], 4)
        target_delta = action_delta * scales
        changed = int(np.argmax(np.abs(target_delta)))
        pre_failure = record['outcome'] == 'orientation_failure' and record['duration_s'] <= 2.
        row = dict(policy=policy, **record, initial_tilt_deg=float(tilt0),
            before_command_failure=pre_failure,
            first_tilt10_s=first_time(tilt > 10), first_tilt20_s=first_time(tilt > 20),
            first_single_support_s=first_time(support.sum(-1) <= 1),
            max_initial_target_change_rad=float(np.abs(target_delta).max()),
            most_changed_joint=names['joint_names'][names['action_joint_ids'][changed]],
            first_halfsecond_train_stand_rate=float(data['train_stand_rate'][:min(n, 25), i].mean()),
            first_halfsecond_train_height_rate=float(data['train_height_rate'][:min(n, 25), i].mean()),
            initial_total_mass=float(physics['masses'][i].sum()),
            initial_torso_com_x=float(physics['coms'][i, 0, 0]),
            initial_torso_com_y=float(physics['coms'][i, 0, 1]))
        for name in names['actuator_names']:
            row[f'delay_{name}_physics_steps'] = int(physics[f'delay_{name}'][i])
            if f'pd_stiffness_{name}' in physics:
                row[f'pd_stiffness_{name}_mean'] = float(physics[f'pd_stiffness_{name}'][i].mean())
        for j, leg in enumerate(('FL', 'FR', 'HL', 'HR')):
            row[f'{leg}_support_first_halfsecond'] = float(support[:25, j].mean())
        rows.append(row)
    return rows


def standing_summary(parent_path, candidate_path):
    old, new = npz(parent_path / 'diagnostic.npz'), npz(candidate_path / 'diagnostic.npz')
    parent_report, report = read(parent_path / 'report.json'), read(candidate_path / 'report.json')
    failed = np.array([r['outcome'] == 'orientation_failure' and r['duration_s'] <= 2 for r in report['records']])
    assert np.array_equal(old['observation'][0], new['observation'][0])
    assert np.array_equal(old['candidate_action'][0], new['candidate_action'][0])
    assert np.array_equal(old['parent_action'][0], new['parent_action'][0])
    assert all(r['duration_s'] >= .24 for r in report['records'])
    physics = npz(parent_path / 'physics_initial.npz')
    candidate_physics = npz(candidate_path / 'physics_initial.npz')
    assert all(np.array_equal(physics[k], candidate_physics[k]) for k in candidate_physics)
    initial = npz(parent_path / 'initial.npz'); q = initial['root'][:, 3:7]
    tilt0 = np.rad2deg(np.arccos(np.clip(1-2*(q[:, 1]**2+q[:, 2]**2), -1, 1)))
    delta = (old['candidate_action'][0]-old['parent_action'][0]) * np.tile([.125, .25, .25], 4)
    changed = np.argmax(np.abs(delta), axis=1)
    quantiles = lambda a: np.quantile(a, [0, .5, 1]).tolist()
    result = dict(paired_initial_physics_exact=True, first_observation_exact=True,
        counterfactual_actions_exact=True, failures=int(failed.sum()),
        initial_tilt_failed_min_median_max_deg=quantiles(tilt0[failed]),
        initial_tilt_other_min_median_max_deg=quantiles(tilt0[~failed]),
        max_initial_joint_target_difference_failed_min_median_max_rad=quantiles(np.abs(delta[failed]).max(-1)),
        hr_hipy_largest_difference_failed_count=int((changed[failed] == 10).sum()),
        hr_hipy_target_difference_failed_min_median_max_rad=quantiles(delta[failed, 10]),
        parent_survived_first2_of_candidate_failed=int(sum(failed[i] and not
            (r['outcome'] == 'orientation_failure' and r['duration_s'] <= 2) for i, r in enumerate(parent_report['records']))),
        parent_survived_first8_of_candidate_failed=int(sum(failed[i] and r['outcome'] == 'survived' for i, r in enumerate(parent_report['records']))))
    for policy, data in (('parent', old), ('candidate', new)):
        result[policy] = dict(mean_tilt_at_024s_deg=float(np.rad2deg(np.arccos(np.clip(-data['gravity'][11, failed, 2], -1, 1))).mean()),
            foot_support_fraction_at_024s=(data['force'][11, failed, :, 2] > 1).mean(0).tolist(),
            first_024s_stand_rate=float(data['train_stand_rate'][:12, failed].mean()),
            first_024s_height_rate=float(data['train_height_rate'][:12, failed].mean()))
    preparation = npz(parent_path / 'preparation.npz')
    result['pd_start_tilt_failed_min_median_max_deg'] = quantiles(np.rad2deg(np.arccos(np.clip(-preparation['gravity'][0, failed, 2], -1, 1))))
    result['actual_pd_parameter_ranges'] = {key: [float(value.min()), float(value.max())]
        for key, value in physics.items() if key.startswith('pd_')}
    return result


def plot_diagnosis(parent_path, candidate_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old, new = npz(parent_path / 'diagnostic.npz'), npz(candidate_path / 'diagnostic.npz')
    failed = np.array([r['outcome'] == 'orientation_failure' for r in read(candidate_path / 'report.json')['records']])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout='constrained')
    for label, data in (('Parent 10000', old), ('Clearance 10500', new)):
        t = np.arange(1, 13) * DT
        axes[0].plot(t, np.rad2deg(np.arccos(np.clip(-data['gravity'][:12, failed, 2], -1, 1))).mean(-1), label=label)
        axes[1].plot(t, data['joint_target'][:12, failed, 7].mean(-1), label=label)
    axes[0].set(title='Same 23 starts, before first failure', xlabel='Time (s)', ylabel='Mean body tilt (deg)')
    axes[1].set(title='Right hind hip target, same starts', xlabel='Time (s)', ylabel='Joint target (rad)')
    for label, root in (('Parent 10000', BASELINE), ('Clearance 10500', CANDIDATE)):
        data = npz(root / 'up_nominal/trace.npz'); init = npz(root / 'up_nominal/initial.npz')
        assert data['active'][:200, 220:240].all()
        heading = yaw(data['quat'][:200, 220:240]); initial_yaw = yaw(init['root'][220:240, 3:7])
        difference = np.remainder(heading-initial_yaw+np.pi, 2*np.pi)-np.pi
        axes[2].plot(np.arange(1, 201)*DT, np.rad2deg(np.abs(difference)).mean(-1), label=label)
    axes[2].set(title='5 cm stairs, 0.6 m/s, all 20 starts', xlabel='Time (s)', ylabel='Mean absolute yaw drift (deg)')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(OUT / 'regression_evidence.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-probe', default='turn_parent_native_8s')
    parser.add_argument('--candidate-probe', default='turn_candidate_native_v2')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stair = []
    for policy, root in (('parent', BASELINE), ('candidate', CANDIDATE)):
        stair.extend(stair_rows(root / 'up_nominal', policy))
    write_csv(OUT / 'stairs_drift.csv', stair)
    paired_windows = []
    for speed in (.32, .4, .6):
        old = {r['env_id']: r for r in stair if r['policy'] == 'parent' and r['speed'] == speed}
        new = {r['env_id']: r for r in stair if r['policy'] == 'candidate' and r['speed'] == speed}
        for seconds in (2, 4, 6, 8):
            keys = [i for i in old if min(old[i]['duration_s'], new[i]['duration_s']) >= seconds]
            paired_windows.append(dict(speed=speed, seconds=seconds, paired_count=len(keys),
                **{f'{policy}_{metric}': float(np.mean([rows[i][f'{metric}_{seconds}s'] for i in keys]))
                   for policy, rows in (('parent', old), ('candidate', new)) for metric in ('yaw_abs', 'drift_abs')}))
    write_csv(OUT / 'stairs_paired_windows.csv', paired_windows)
    checks, stands = [], []
    for policy, root, name in (('parent', BASELINE, args.parent_probe), ('candidate', CANDIDATE, args.candidate_probe)):
        path = OUTPUT / name
        if (path / 'diagnostic_report.json').is_file():
            checks.append(verify_replay(path, root / 'turn_training_randomized'))
            stands.extend(stand_rows(path, policy))
    if stands:
        write_csv(OUT / 'standing_episodes.csv', stands)
    detail = standing_summary(OUTPUT / args.parent_probe, OUTPUT / args.candidate_probe)
    plot_diagnosis(OUTPUT / args.parent_probe, OUTPUT / args.candidate_probe)
    controls = []
    for path in sorted(OUTPUT.iterdir()):
        if not (path / 'diagnostic_report.json').is_file():
            continue
        meta, report = read(path / 'diagnostic_report.json'), read(path / 'report.json')
        if meta['intervention'] == 'native':
            continue
        reference = (BASELINE if meta['policy'] == 'parent' else CANDIDATE) / (
            f'turn_{meta["profile"]}' if meta['kind'] == 'turn' else f'up_{meta["profile"]}')
        before, after = npz(reference / 'initial.npz'), npz(path / 'initial.npz')
        assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
        initial_physics = npz(path / 'physics_initial.npz')
        effective_physics = npz(path / 'physics_effective.npz')
        for key in initial_physics:
            if meta['intervention'] == 'no_delay' and key.startswith('delay_'):
                assert (effective_physics[key] == 0).all()
            else:
                assert np.array_equal(initial_physics[key], effective_physics[key]), key
        row = dict(run=path.name, **meta, paired_initial_exact=True, physical_intervention_scope_passed=True,
            orientation_failures=sum(r['outcome'] == 'orientation_failure' for r in report['records']))
        if meta['kind'] == 'stairs':
            row['five_cm'] = [dict(speed=v, **{outcome: sum(r['height'] == .05 and r['speed'] == v and r['outcome'] == outcome
                for r in report['records']) for outcome in ('success', 'lateral_exit', 'orientation_failure', 'stuck', 'timeout')}) for v in (.32, .4, .6)]
        controls.append(row)
    summary = dict(status='diagnosed', replay_checks=checks, paired_windows=paired_windows, controls=controls, standing=detail,
        initial_failures={p: sum(r['before_command_failure'] for r in stands if r['policy'] == p) for p in ('parent', 'candidate')},
        training=False, optimizer_updates=0, formal_acceptance_unchanged=True,
        source_sha256=sha(__file__), caveats=['20 ms 采样不能排除物理子步尖峰',
            '接触是几何和净力代理；时序关联不能单独证明因果', '单因素对照是诊断，不能替代正式原协议验收'])
    save(OUT / 'analysis.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
