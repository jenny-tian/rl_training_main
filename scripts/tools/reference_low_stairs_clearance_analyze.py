"""净空试训的配对行为验收；运行成功与能力通过分别记录。"""
from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT, MODELS, load_actor, save, sha
from reference_six_watch import read
from reference_low_stairs_clearance_evaluate import TRAINING, OUTPUT, MODEL, verify_training, verify_run, same_configuration
from reference_low_stairs_evaluate import verify_run as verify_baseline
from reference_core500_evaluate import same_initial
from reference_core500_analyze import write_csv, compare_video, paired_turn_rows
from reference_turn_analyze import analyze as analyze_turn, WINDOWS
from reference_turn_diagnose import CASES
from reference_isaac_stairs_analyze import analyze as analyze_stairs
from reference_low_stairs_analyze import stairs_heights, check_log, failure_diagnostics, flat_heading_diagnostics, read_agent
from reference_low_stairs_a_analyze import contact_diagnostics
from reference_low_stairs_failure_audit import stair_features, LEGS, DT

OUT = ROOT / 'docs/lite3-low-stairs-clearance-trial-20261115'
OUTCOMES = ('success', 'stuck', 'orientation_failure', 'lateral_exit', 'timeout')


def screen(heights, groups, pairs, turn_failures):
    checks = []
    def require(name, actual, target, minimum=True):
        checks.append(dict(name=name, actual=actual, target=target, comparison='>=' if minimum else '<=',
                           passed=bool(actual >= target if minimum else actual <= target)))
    for speed, target in ((.32, 1), (.4, 5), (.6, 14)):
        row = next(r for r in groups if (r['policy'], r['profile'], r['direction'], r['height'], r['speed']) ==
                   ('candidate', 'nominal', 'up', .05, speed))
        require(f'nominal_up_5cm_{speed}_success', row['success'], target)
    for row in heights:
        if row['policy'] != 'candidate' or row['profile'] != 'nominal':
            continue
        old = next(r for r in heights if r['policy'] == 'baseline' and
                   (r['profile'], r['direction'], r['height']) == (row['profile'], row['direction'], row['height']))
        if (row['direction'], row['height']) != ('up', .05):
            require(f'nominal_{row["direction"]}_{row["height"]}_success', row['success'], old['success'])
        require(f'nominal_{row["direction"]}_{row["height"]}_orientation', row['orientation_failure'], old['orientation_failure'], False)
    worsening = [r for r in pairs if r['candidate_mae'] > r['baseline_mae'] * 1.1 + 1e-12]
    expected = {(p, c, w) for p in ('nominal', 'training_randomized') for c in CASES for w in WINDOWS}
    missing = sorted(expected - {(r['profile'], r['case'], r['window']) for r in pairs})
    require('turn_windows_worse_than_10_percent', len(worsening), 0, False)
    require('turn_missing_windows', len(missing), 0, False)
    require('turn_new_paired_orientation_failures', sum(r['new_failures'] for r in turn_failures), 0, False)
    random_changes = []
    for row in groups:
        if row['policy'] != 'candidate' or row['profile'] != 'training_randomized':
            continue
        old = next(r for r in groups if r['policy'] == 'baseline' and
            (r['profile'], r['direction'], r['height'], r['speed']) ==
            (row['profile'], row['direction'], row['height'], row['speed']))
        random_changes.append(dict(direction=row['direction'], height=row['height'], speed=row['speed'],
            baseline_success=old['success'], candidate_success=row['success'], success_delta=row['success'] - old['success'],
            baseline_orientation=old['orientation_failure'], candidate_orientation=row['orientation_failure']))
    decreases = [r for r in random_changes if r['success_delta'] < 0 or r['candidate_orientation'] > r['baseline_orientation']]
    passed = all(r['passed'] for r in checks)
    return dict(status='failed' if not passed else ('requires_review' if decreases else 'passed'),
        engineering_checks_passed=passed, checks=checks, turn_error_regressions=worsening, turn_missing_windows=missing,
        random_groups=random_changes, random_decreases_require_review=decreases,
        scope='固定工程筛查，不代替多种子、统计或部署验收；不自动追加训练')


def training_summary():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    report = read(TRAINING / 'report.json')
    events = EventAccumulator(str(TRAINING / 'training'), size_guidance={'scalars': 0}); events.Reload()
    metrics = {}
    for tag in ('Train/mean_reward', 'Train/mean_episode_length', 'Episode_Termination/bad_orientation_2',
                'Episode_Termination/time_out', 'Metrics/base_velocity/error_vel_xy', 'Metrics/base_velocity/error_vel_yaw',
                'Episode_Reward/stair_clearance', 'Episode_Reward/stair_progress', 'Episode_Reward/stair_landing'):
        values = events.Scalars(tag)
        assert len(values) == 500 and [v.step for v in values] == list(range(10000, 10500))
        a = np.array([v.value for v in values]); assert np.isfinite(a).all()
        metrics[tag] = dict(first20=float(a[:20].mean()), first100=float(a[:100].mean()),
                            last100=float(a[-100:].mean()), last=float(a[-1]))
    coverage, moving, scored = (np.array(report[key]) for key in
        ('coverage_by_column_level', 'translation_frames_by_column_level', 'clearance_frames_by_column_level'))
    assert coverage.shape == moving.shape == scored.shape == (20, 10)
    assert coverage.sum() == report['interactions'] and np.all(scored <= moving) and np.all(moving <= coverage)
    geometry = read(TRAINING / 'geometry.json')['tiles']; rows = []
    for family in dict.fromkeys(report['terrain_column_names']):
        columns = [i for i, name in enumerate(report['terrain_column_names']) if name == family]
        for level in range(10):
            tiles = [t for t in geometry if t['direction'] == family and t['row'] == level]
            rows.append(dict(family=family, level=level, environment_frames=int(coverage[columns, level].sum()),
                translation_frames=int(moving[columns, level].sum()), clearance_frames=int(scored[columns, level].sum()),
                min_stair_height_m=min(t['min_height_m'] for t in tiles) if tiles else None,
                max_stair_height_m=max(t['max_height_m'] for t in tiles) if tiles else None))
    return dict(status='passed', updates=500, interactions=report['interactions'], metrics=metrics, coverage=rows,
        initial_level_histogram=report['initial_level_histogram'], final_level_histogram=report['final_terrain_level_histogram'],
        gait_range=[min(r['gait_level'] for r in report['updates']), max(r['gait_level'] for r in report['updates'])],
        initial_learning_rate=report['initial_learning_rate'], final_learning_rate=report['final_learning_rate'],
        execution=read(TRAINING / 'execution.json'), first20=read(TRAINING / 'first20_check.json'),
        wall_scored_frames_per_foot=report['wall_scored_frames_per_foot'])


def adapter_check(checkpoint):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    torch.set_num_threads(1); torch.manual_seed(20261114)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with np.load(OUTPUT / 'turn_nominal/initial.npz') as initial:
        actual = torch.from_numpy(initial['observation'].copy())
    inputs = torch.cat((torch.randn(2048, 45), actual)); obs = TensorDict({'policy': inputs}, batch_size=[len(inputs)])
    native = MLPModel(obs, {'actor': ['policy']}, 'actor', 12, **read_agent()).eval()
    native.load_state_dict(state['actor_state_dict'], strict=True)
    MODELS[MODEL] = checkpoint; adapted = load_actor(MODEL)
    with torch.inference_mode():
        difference = (native(obs) - adapted(inputs)).abs().max().item()
    assert difference == 0.
    return dict(status='passed', samples=len(inputs), max_output_difference=difference, checkpoint=str(checkpoint), sha256=sha(checkpoint))


def rear_diagnostics(report, cfg, data):
    rows = []
    for record in report['records']:
        if not record['height']:
            continue
        i = record['env_id']; n = int(data['active'][:, i].sum())
        features = stair_features(data['feet'][:n, i], data['forces'][:n, i], cfg['geometry'][record['column']])
        tilt = np.degrees(np.arccos(np.clip(-data['gravity'][:n, i, 2], -1., 1.)))
        above = np.flatnonzero(tilt > 20.); first = int(above[0]) if len(above) else n
        start = max(0, first - round(.5 / DT))
        for j, leg in enumerate(LEGS):
            rows.append(dict(profile=report['profile'], env_id=i, height=record['height'], speed=record['speed'],
                outcome=record['outcome'], leg=leg, first_tilt20_s=(first + 1) * DT if len(above) else None,
                window_end='first_tilt20' if len(above) else 'episode_end',
                collision_frames_halfsecond=int(features['collision'][start:first, j].sum()),
                wall_frames_halfsecond=int(features['wall'][start:first, j].sum()),
                support_frames_halfsecond=int(features['support'][start:first, j].sum())))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    complete = read(OUTPUT / 'complete.json'); assert complete['status'] == 'passed' and complete['candidate_episodes'] == 1600
    checkpoint, digest = verify_training(); assert digest == complete['candidate_sha256']
    tables = {k: [] for k in ('stairs_heights', 'stairs_groups', 'stairs_episodes', 'stairs_reward_windows',
        'turn_groups', 'turn_windows', 'turn_paired', 'turn_failures', 'stairs_paired', 'failure_diagnostics',
        'contacts_before_tilt', 'first_front_approaches', 'flat_heading', 'legs_before_tilt')}
    checks, logs, raw = [], [], set()
    for entry in complete['matrix']:
        old, new = Path(entry['baseline']), Path(entry['candidate'])
        assert same_initial(old, new) and same_configuration(read(old / 'config.json'), read(new / 'config.json'))
        reports, pair_windows = {}, {}
        for policy, directory in (('baseline', old), ('candidate', new)):
            report = (verify_baseline(directory, entry['kind'], entry['profile'], entry['direction']) if policy == 'baseline'
                      else verify_run(directory, entry['kind'], entry['profile'], digest, entry['direction']))
            reports[policy] = report; logs.append(check_log(directory))
            def extend(name, rows):
                tables[name].extend(dict(policy=policy, **r) for r in rows)
            if entry['kind'] == 'turn':
                _, data, rows, grouped = analyze_turn(directory)
                extend('turn_groups', grouped); extend('turn_windows', rows); pair_windows[policy] = rows
            else:
                _, cfg, data, ep, rw, _ = analyze_stairs(directory)
                extend('stairs_heights', stairs_heights(report)); extend('stairs_episodes', ep); extend('stairs_reward_windows', rw)
                extend('failure_diagnostics', failure_diagnostics(report, data))
                extend('flat_heading', flat_heading_diagnostics(report, data))
                extend('stairs_groups', [dict(profile=entry['profile'], direction=entry['direction'],
                    **{k: v for k, v in g.items() if k != 'outcomes'},
                    **{k: g['outcomes'].get(k, 0) for k in OUTCOMES}) for g in report['groups']])
                if entry['direction'] == 'up':
                    contacts, approaches = contact_diagnostics(report, cfg, data)
                    extend('contacts_before_tilt', contacts); extend('first_front_approaches', approaches)
                    extend('legs_before_tilt', rear_diagnostics(report, cfg, data))
            assert all(np.isfinite(v).all() for v in data.values())
            checks.append(dict(policy=policy, run=str(directory), episodes=report['episodes'], arrays_finite=True,
                               actor_unchanged=True, initial_equal=True, common_config_equal=True))
            raw.update(p for p in directory.iterdir() if p.is_file())
            for name in ('report.json', 'config.json', 'execution.json'):
                shutil.copy2(directory / name, OUT / f'{policy}_{directory.name}_{name}')
        a, b = ({r['env_id']: r for r in reports[k]['records']} for k in ('baseline', 'candidate'))
        assert a.keys() == b.keys()
        if entry['kind'] == 'turn':
            tables['turn_paired'].extend(paired_turn_rows(pair_windows['baseline'], pair_windows['candidate'], entry['profile']))
            for case in CASES:
                ids = [i for i, r in a.items() if r['case'] == case]
                assert all(a[i]['case'] == b[i]['case'] and a[i]['repeat'] == b[i]['repeat'] for i in ids)
                tables['turn_failures'].append(dict(profile=entry['profile'], case=case, starts=len(ids),
                    baseline_failures=sum(a[i]['outcome'] == 'orientation_failure' for i in ids),
                    candidate_failures=sum(b[i]['outcome'] == 'orientation_failure' for i in ids),
                    new_failures=sum(a[i]['outcome'] != 'orientation_failure' and b[i]['outcome'] == 'orientation_failure' for i in ids),
                    recovered=sum(a[i]['outcome'] == 'orientation_failure' and b[i]['outcome'] != 'orientation_failure' for i in ids)))
        else:
            for i in a:
                assert all(a[i][k] == b[i][k] for k in ('height', 'speed', 'repeat', 'column'))
                tables['stairs_paired'].append(dict(profile=entry['profile'], direction=entry['direction'], env_id=i,
                    height=a[i]['height'], speed=a[i]['speed'], repeat=a[i]['repeat'],
                    baseline_outcome=a[i]['outcome'], candidate_outcome=b[i]['outcome'],
                    gained_success=a[i]['outcome'] != 'success' and b[i]['outcome'] == 'success',
                    lost_success=a[i]['outcome'] == 'success' and b[i]['outcome'] != 'success'))
    assert sum(r['episodes'] for r in checks if r['policy'] == 'candidate') == 1600
    for name, rows in tables.items():
        write_csv(OUT / f'{name}.csv', rows)
    video_checks = {}
    for entry in complete['videos']:
        directory = Path(entry['candidate'])
        verify_run(directory, entry['kind'], entry['profile'], digest, entry['direction'], entry['video_env'])
        logs.append(check_log(directory)); dest = OUT / 'clips' / entry['name']
        compare_video(directory, Path(entry['formal']), dest); video_checks[entry['name']] = read(dest / 'replay_validation.json')
        raw.update(p for p in directory.iterdir() if p.is_file())
    training = training_summary(); adapter = adapter_check(checkpoint)
    test_file = TRAINING.parent / 'tests.xml'
    suites = ET.parse(test_file).getroot().findall('testsuite')
    assert suites and all(int(s.get(k)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    test_count = sum(int(s.get('tests')) for s in suites); assert test_count == 139
    shutil.copy2(test_file, OUT / 'tests.xml'); raw.add(test_file)
    screening = screen(tables['stairs_heights'], tables['stairs_groups'], tables['turn_paired'], tables['turn_failures'])
    save(OUT / 'screening.json', screening); save(OUT / 'training_summary.json', training)
    write_csv(OUT / 'training_terrain_coverage.csv', training['coverage'])
    save(OUT / 'summary.json', dict(status='passed', status_scope='证据核验通过，能力验收单独见 screening',
        candidate_sha256=digest, candidate_episodes=1600, reused_baseline_episodes=1600, training_updates=500,
        stairs_heights=tables['stairs_heights'], stairs_groups=tables['stairs_groups'], turn_paired=tables['turn_paired'],
        turn_failures=tables['turn_failures'], screening=screening, matrix=complete['matrix'], videos=complete['videos'],
        caveats=['单训练种子、固定评估起点，不作为部署资格。', '公共奖励评分不是新净空奖励的回放。',
                 '转向误差使用配对完整时间窗，失败统计保留所有回合。', '碰阶诊断为每 20 ms 采样的几何与合力代理。',
                 '旧课程按已记录总数重建，无法分别归因课程恢复和净空修改。']))
    save(OUT / 'verification.json', dict(status='passed', formal_candidate_episodes=1600, adapter=adapter,
        rollout_checks=checks, runtime_logs=logs, video_checks=video_checks, video_replays_excluded=True,
        additional_training_performed_by_analyzer=False, tests_passed=test_count,
        sources=[dict(path=str(Path(__file__).with_name(name)), sha256=sha(Path(__file__).with_name(name))) for name in
                 ('reference_low_stairs_clearance_train.py', 'reference_low_stairs_clearance_evaluate.py',
                  'reference_low_stairs_clearance_analyze.py')]))
    for name in ('report.json', 'execution.json', 'restore.json', 'first20_check.json', 'terrain_audit.json',
                 'clearance_audit.json', 'implementation_sources.json', 'geometry.json', 'configuration.json'):
        shutil.copy2(TRAINING / name, OUT / f'training_{name}')
    raw.update(p for p in TRAINING.rglob('*') if p.is_file()); raw.update(p for p in OUTPUT.iterdir() if p.is_file())
    save(OUT / 'raw_manifest.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(raw)])
    plot(tables['stairs_heights'], tables['turn_paired'])
    print(f'CLEARANCE ANALYSIS COMPLETE: {screening["status"]}', flush=True)


def plot(heights, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, profile in enumerate(('nominal', 'training_randomized')):
        for col, direction in enumerate(('up', 'down')):
            ax = axes[row, col]
            for policy, label, offset in (('baseline', 'Parent 10000', -.18), ('candidate', 'Clearance 10500', .18)):
                rows = [next(r for r in heights if (r['policy'], r['profile'], r['direction'], r['height']) ==
                        (policy, profile, direction, h)) for h in (0., .02, .03, .05)]
                ax.bar(np.arange(4) + offset, [r['success'] for r in rows], .36, label=label)
            ax.set_xticks(range(4), ['Flat', '2 cm', '3 cm', '5 cm']); ax.set_ylim(0, 65)
            ax.set_ylabel('Course completed / 60'); ax.set_title(f'{profile} / {direction}')
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(OUT / 'stairs_comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, profile in zip(axes, ('nominal', 'training_randomized')):
        rows = [next((r for r in pairs if (r['profile'], r['case'], r['window']) == (profile, case, 'steady')), None) for case in CASES]
        for policy, label in (('baseline', 'Parent 10000'), ('candidate', 'Clearance 10500')):
            ax.plot(range(len(CASES)), [r[f'{policy}_mae'] if r else np.nan for r in rows], marker='o', label=label)
        ax.set_xticks(range(len(CASES)), [f'{case}\nn={r["n"] if r else 0}' for case, r in zip(CASES, rows)], rotation=35, ha='right')
        ax.set_ylabel('Paired yaw MAE (rad/s)'); ax.set_title(profile); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(OUT / 'turn_comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
