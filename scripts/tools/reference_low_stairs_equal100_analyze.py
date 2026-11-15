"""相同更新数的三方冻结对照；历史 500 次结果单列为背景。"""
from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_equal100_evaluate import (
    OUTPUT, DOCS, EXPECTED, GUARDED_SHA, PARENT_RUNS, GUARDED_RUNS, preflight, read, verify_run)
from reference_low_stairs_evaluate import EXPECTED_SHA
from reference_low_stairs_clearance_evaluate import OUTPUT as CLEARANCE500
from reference_low_stairs_regression_probe import OUTPUT as DIAGNOSIS
from reference_low_stairs_guarded_analyze import adapter_check
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_core500_analyze import write_csv, paired_turn_rows
from reference_turn_analyze import analyze as analyze_turn, WINDOWS
from reference_turn_diagnose import CASES
from reference_isaac_stairs_analyze import analyze as analyze_stairs
from reference_low_stairs_analyze import stairs_heights, check_log
from reference_low_stairs_failure_audit import stair_features, DT
from reference_low_stairs_regression_analyze import yaw, first_time

PRIMARY = ('parent10000', 'clearance10100', 'guarded10100')
OUTCOMES = ('success', 'stuck', 'orientation_failure', 'lateral_exit', 'timeout')


def common_turns(windows, profile):
    """同一起点必须在三方均完整存活；缺失时间窗也显式保留。"""
    rows = []
    for case in CASES:
        for window in WINDOWS:
            maps = {p: {r['repeat']: r for r in windows[p]
                        if r['case'] == case and r['window'] == window} for p in PRIMARY}
            ids = sorted(set.intersection(*(set(m) for m in maps.values())))
            ids = [i for i in ids if all(m[i]['complete_window'] for m in maps.values())]
            row = dict(profile=profile, case=case, window=window, starts=20, n=len(ids))
            for policy, mapping in maps.items():
                for metric in ('yaw_command', 'yaw_actual', 'yaw_mae'):
                    row[f'{policy}_{metric}'] = float(np.mean([mapping[i][metric] for i in ids])) if ids else None
            rows.append(row)
    return rows


def stagnation(x, before):
    """过去一秒前进不足 5 cm 且连续满足 0.5 s；返回首个慢窗末时刻。"""
    slow = np.zeros(len(x), dtype=bool)
    lag, persistence = round(1. / DT), round(.5 / DT)
    if len(x) > lag:
        slow[lag:] = x[lag:] - x[:-lag] < .05
    slow &= before
    edges = np.diff(np.r_[False, slow, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    sustained = starts[(ends - starts) >= persistence]
    return dict(first_stagnation_window_end_s=float((sustained[0] + 1) * DT) if len(sustained) else None,
                slow_window_seconds=float(slow.sum() * DT),
                longest_slow_window_run_s=float((ends - starts).max(initial=0) * DT))


def stair_diagnostics(report, cfg, data, policy, heading):
    rows, series = [], {}
    for record in report['records']:
        if record['height'] != .05:
            continue
        i = record['env_id']; n = int(data['active'][:, i].sum())
        root = data['root'][:n, i]
        tilt = np.rad2deg(np.arccos(np.clip(-data['gravity'][:n, i, 2], -1., 1.)))
        ids = np.flatnonzero(tilt > 20.)
        cutoff = int(ids[0]) if len(ids) else n
        before = (np.arange(n) < cutoff) & (root[:, 0] > .85)
        features = stair_features(data['feet'][:n, i], data['forces'][:n, i], cfg['geometry'][record['column']])
        collision, wall = features['collision'], features['wall']
        row = dict(policy=policy, profile=report['profile'], heading=heading,
            env_id=i, speed=record['speed'], repeat=record['repeat'], outcome=record['outcome'],
            duration_s=n*DT, best_risers=record['best_risers'], final_x_m=float(root[-1, 0]),
            final_y_m=float(root[-1, 1]), final_yaw_deg=float(np.rad2deg(yaw(data['quat'][n-1, i]))),
            first_tilt20_s=(cutoff+1)*DT if cutoff<n else None,
            before_tilt_exposure_s=float(before.sum()*DT),
            first_collision_before_tilt_s=first_time(collision.any(-1) & before),
            collision_foot_seconds=float(collision[before].sum()*DT),
            wall_foot_seconds=float(wall[before].sum()*DT),
            collision_foot_fraction=float(collision[before].mean()) if before.any() else None,
            **stagnation(root[:, 0], before))
        for seconds in (2, 4, 6, 8):
            index = round(seconds/DT)-1
            row[f'root_x_{seconds}s'] = float(root[index, 0]) if index<n else None
        rows.append(row)
        series[i] = dict(root=root, cutoff=cutoff, collision=collision, wall=wall, speed=record['speed'])
    return rows, series


def paired_contact(series, profile, heading):
    """三方首次倾斜/终止前的共同时间和共同到阶区域，减少暴露差异。"""
    rows = []
    for i in sorted(set.intersection(*(set(series[p]) for p in PRIMARY))):
        selected = {p: series[p][i] for p in PRIMARY}
        end = min(s['cutoff'] for s in selected.values())
        mask = np.ones(end, dtype=bool)
        for s in selected.values():
            mask &= s['root'][:end, 0] > .85
        row = dict(profile=profile, heading=heading, env_id=i, speed=selected[PRIMARY[0]]['speed'],
                   exposure_s=float(mask.sum()*DT))
        for policy, s in selected.items():
            for feature in ('collision', 'wall'):
                row[f'{policy}_{feature}_foot_seconds'] = float(s[feature][:end][mask].sum()*DT)
        rows.append(row)
    return rows


def heading_error(path, data):
    with np.load(path/'initial.npz') as initial:
        previous = np.concatenate((yaw(initial['root'][:, 3:7])[None], yaw(data['quat'][:-1])))
    expected = np.where(data['command'][..., 0] > 0,
        np.clip(.5*((-previous+np.pi)%(2*np.pi)-np.pi), -.8, .8), 0.)
    error = float(np.abs(expected-data['command'][..., 2])[data['active']].max())
    assert error < 1e-5
    return error


def historical_heading(path, digest):
    """仅接受完整 1000 步旧航向诊断；不修改其 probe 标志或历史文件。"""
    report, cfg, diagnostic, execution = (read(path/f'{n}.json') for n in
                                        ('report', 'config', 'diagnostic_report', 'execution'))
    assert diagnostic['status'] == 'passed' and diagnostic['diagnostic_only']
    assert not diagnostic['training'] and diagnostic['optimizer_updates'] == 0
    assert diagnostic['intervention'] == 'heading' and diagnostic['kind'] == 'stairs'
    assert diagnostic['steps'] == report['steps'] == 1000
    assert diagnostic['initial_preparation_steps'] == 50
    # 旧诊断运行后、首次提交前脚本发生过变化；按已提交的旧报告核验来源，
    # 不声称当前文件就是当时源码。实际反馈公式另从完整轨迹逐帧核验。
    archived = read(ROOT/'docs/lite3-low-stairs-regression-preparation-20261115/verification.json')
    saved = next(r for r in archived['runs'] if r['name'] == path.name)
    assert saved['diagnostic'] == diagnostic and saved['execution'] == execution
    assert diagnostic['source_sha256'] == 'b95eb76fcae969aabd99272908e343abe9b907a59a7f77efb89450afadeea501'
    assert diagnostic['original_protocol_sha256'] == sha(ROOT/'scripts/tools/reference_isaac_stairs_diagnose.py')
    assert execution['status'] == 'completed' and execution['exit_code'] == 0
    assert report['status'] == 'passed' and report['actor_unchanged'] and report['probe']
    assert report['checkpoint_sha256'] == digest and report['episodes'] == diagnostic['episodes'] == 240
    assert report['profile'] == 'nominal' and report['direction'] == 'up' and report['step_dt'] == DT
    assert report['action_mode'] == 'mean' and report['height_set'] == 'low_stairs'
    assert report['first_episodes_only'] and report['pre_reset_capture']
    with np.load(path/'trace.npz') as archive:
        data = {k: archive[k] for k in archive.files}
    return report, cfg, data


def validate_trace(report, data):
    assert data['active'].shape == (report['steps'], report['episodes'])
    assert all(np.isfinite(v).all() for v in data.values())
    for record in report['records']:
        active = data['active'][:, record['env_id']]
        n = int(active.sum())
        assert n > 0 and active[:n].all() and not active[n:].any()
        assert abs(n*DT-record['duration_s']) < 1e-5


def group_diagnostics(rows):
    result = []
    for policy, profile, heading, speed in sorted({(r['policy'], r['profile'], r['heading'], r['speed']) for r in rows}):
        selected = [r for r in rows if (r['policy'], r['profile'], r['heading'], r['speed']) ==
                    (policy, profile, heading, speed)]
        exposure = sum(r['before_tilt_exposure_s'] for r in selected)
        row = dict(policy=policy, profile=profile, heading=heading, speed=speed, starts=len(selected),
            successes=sum(r['outcome']=='success' for r in selected),
            stagnation_events=sum(r['first_stagnation_window_end_s'] is not None for r in selected),
            tilt20_events=sum(r['first_tilt20_s'] is not None for r in selected),
            collision_events=sum(r['first_collision_before_tilt_s'] is not None for r in selected),
            exposure_seconds=exposure,
            exposure_weighted_collision_fraction=sum(r['collision_foot_seconds'] for r in selected)/(4*exposure) if exposure else None)
        for key in ('first_stagnation_window_end_s', 'first_tilt20_s', 'first_collision_before_tilt_s',
                    'collision_foot_seconds', 'wall_foot_seconds', 'final_x_m', 'best_risers'):
            values = [r[key] for r in selected if r[key] is not None]
            row[f'mean_{key}'] = float(np.mean(values)) if values else None
        result.append(row)
    return result


def main():
    checkpoint, lineage = preflight()
    complete = read(OUTPUT/'complete.json')
    assert complete['status'] == 'passed' and complete['new_episodes'] == 1360 and complete['training_updates'] == 0
    roots = dict(parent10000=PARENT_RUNS, clearance10100=OUTPUT, guarded10100=GUARDED_RUNS, clearance10500=CLEARANCE500)
    hashes = dict(parent10000=EXPECTED_SHA, clearance10100=EXPECTED, guarded10100=GUARDED_SHA,
                  clearance10500=lineage['clearance500_sha256'])
    labels = dict(parent10000='lowstairs10000', clearance10100='lowstairs_clearance10100',
                  guarded10100='lowstairs_guarded10100', clearance10500='lowstairs_clearance10500')
    tables = {k: [] for k in ('turn_groups', 'turn_windows', 'turn_pairs', 'turn_common_three', 'turn_failures',
        'stairs_heights', 'stairs_groups', 'stairs_pairs', 'diagnostics', 'paired_contact')}
    checks, runs = [], []
    for profile in ('nominal', 'training_randomized'):
        windows, reports = {}, {}
        for policy, root in roots.items():
            path = root/f'turn_{profile}'
            verify_run(path, 'turn', profile, hashes[policy], labels[policy])
            assert same_initial(path, PARENT_RUNS/f'turn_{profile}')
            assert same_configuration(read(path/'config.json'), read(PARENT_RUNS/f'turn_{profile}/config.json'))
            report, data, rows, groups = analyze_turn(path)
            validate_trace(report, data)
            checks.append(check_log(path)); runs.append((policy, path))
            tables['turn_groups'].extend(dict(policy=policy, **r) for r in groups)
            tables['turn_windows'].extend(dict(policy=policy, **r) for r in rows)
            windows[policy], reports[policy] = rows, report
            del data
        tables['turn_common_three'].extend(common_turns(windows, profile))
        for before, after in (('parent10000','clearance10100'), ('parent10000','guarded10100'),
                              ('clearance10100','guarded10100'), ('clearance10100','clearance10500')):
            tables['turn_pairs'].extend(dict(baseline_policy=before, candidate_policy=after, **r)
                                       for r in paired_turn_rows(windows[before], windows[after], profile))
        parent = {r['env_id']:r for r in reports['parent10000']['records']}
        for policy, report in reports.items():
            for case in CASES:
                records = [r for r in report['records'] if r['case']==case]
                tables['turn_failures'].append(dict(policy=policy, profile=profile, case=case, starts=len(records),
                    before_command=sum(r['outcome']=='orientation_failure' and r['duration_s']<=2. for r in records),
                    after_command=sum(r['outcome']=='orientation_failure' and r['duration_s']>2. for r in records),
                    new_vs_parent=sum(r['outcome']=='orientation_failure' and parent[r['env_id']]['outcome']!='orientation_failure' for r in records),
                    recovered_vs_parent=sum(r['outcome']!='orientation_failure' and parent[r['env_id']]['outcome']=='orientation_failure' for r in records)))

    for profile, heading in (('nominal', False), ('training_randomized', False), ('nominal', True)):
        series, reports = {}, {}
        for policy, root in roots.items():
            historical = heading and policy in ('parent10000','clearance10500')
            path = (DIAGNOSIS/('stairs_parent_heading' if policy=='parent10000' else 'stairs_candidate_heading')
                    if historical else root/('heading_up_nominal' if heading else f'up_{profile}'))
            if historical:
                report, cfg, data = historical_heading(path, hashes[policy])
            else:
                verify_run(path, 'stairs', profile, hashes[policy], labels[policy], heading)
                report, cfg, data, _, _, _ = analyze_stairs(path)
            assert same_initial(path, PARENT_RUNS/f'up_{profile}')
            assert same_configuration(cfg, read(PARENT_RUNS/f'up_{profile}/config.json'))
            validate_trace(report, data)
            check = check_log(path)
            if heading:
                check['max_heading_feedback_error'] = heading_error(path, data)
            if historical:
                check['legacy_diagnostic_source_sha256'] = read(path/'diagnostic_report.json')['source_sha256']
                check['current_diagnostic_source_sha256'] = sha(ROOT/'scripts/tools/reference_low_stairs_regression_probe.py')
                check['legacy_source_matches_current'] = False
                check['provenance_note'] = '已提交旧报告和原始文件哈希核验；运行时旧探针源码未单独保存，另核验逐帧反馈公式。'
            checks.append(check); runs.append((policy, path)); reports[policy] = report
            tables['stairs_heights'].extend(dict(policy=policy, heading=heading, **r) for r in stairs_heights(report))
            tables['stairs_groups'].extend(dict(policy=policy, profile=profile, heading=heading,
                height=g['height'], speed=g['speed'], starts=g['episodes'], **{k:g['outcomes'].get(k,0) for k in OUTCOMES})
                for g in report['groups'])
            rows, series[policy] = stair_diagnostics(report, cfg, data, policy, heading)
            tables['diagnostics'].extend(rows)
            del data
        tables['paired_contact'].extend(paired_contact(series, profile, heading))
        for before, after in (('parent10000','clearance10100'), ('clearance10100','guarded10100'),
                              ('clearance10100','clearance10500')):
            a, b = ({r['env_id']:r for r in reports[p]['records']} for p in (before, after))
            for height in (0., .02, .03, .05):
                for speed in (.32, .4, .6):
                    ids = [i for i,r in a.items() if (r['height'],r['speed'])==(height,speed)]
                    tables['stairs_pairs'].append(dict(profile=profile, heading=heading,
                        baseline_policy=before, candidate_policy=after, height=height, speed=speed, starts=len(ids),
                        gained_success=sum(a[i]['outcome']!='success' and b[i]['outcome']=='success' for i in ids),
                        lost_success=sum(a[i]['outcome']=='success' and b[i]['outcome']!='success' for i in ids),
                        new_orientation_failure=sum(a[i]['outcome']!='orientation_failure' and b[i]['outcome']=='orientation_failure' for i in ids)))
    tables['diagnostic_groups'] = group_diagnostics(tables['diagnostics'])
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(DOCS/f'{name}.csv', rows)
    # 对每个复用文件核验当时已归档的哈希，不以当前重新算出的哈希代替历史来源。
    historical_hashes = {}
    for folder in ('results-20261114','clearance-trial-20261115','guarded-trial-20261115','regression-preparation-20261115'):
        for entry in read(ROOT/f'docs/lite3-low-stairs-{folder}/raw_manifest.json'):
            if entry['path'] in historical_hashes:
                assert historical_hashes[entry['path']] == entry['sha256']
            historical_hashes[entry['path']] = entry['sha256']
    raw, reused = {}, 0
    for policy, path in runs:
        for file in sorted(path.iterdir()):
            if not file.is_file():
                continue
            digest = sha(file)
            if policy != 'clearance10100':
                assert historical_hashes[str(file)] == digest, file
                reused += 1
            raw[str(file)] = dict(path=str(file), bytes=file.stat().st_size, sha256=digest)
            if file.suffix == '.json':
                dest = DOCS/'runs'/policy/path.name/file.name
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file,dest)
        cfg = read(path/'config.json')
        assert all(sha(s['path'])==s['sha256'] for s in cfg['sources'])
    for file in OUTPUT.glob('*.json'):
        shutil.copy2(file, DOCS/file.name)
        raw[str(file)] = dict(path=str(file), bytes=file.stat().st_size, sha256=sha(file))
    for file in (checkpoint, Path(lineage['guarded_checkpoint']), Path(lineage['clearance500_checkpoint'])):
        raw[str(file)] = dict(path=str(file), bytes=file.stat().st_size, sha256=sha(file))
    adapter = adapter_check(checkpoint, OUTPUT)
    suites = ET.parse(DOCS/'tests.xml').getroot().findall('testsuite')
    assert all(int(s.get(k))==0 for s in suites for k in ('errors','failures','skipped'))
    assert sha(checkpoint) == EXPECTED
    save(DOCS/'summary.json', dict(status='passed', status_scope='证据核验通过，不代表能力验收通过',
        new_episodes=1360, new_training_updates=0, lineage=lineage,
        primary_policies=list(PRIMARY), context_policy='clearance10500', **tables))
    save(DOCS/'verification.json', dict(status='passed', adapter=adapter, runs=checks,
        historical_files_verified=reused, tests_passed=sum(int(s.get('tests')) for s in suites),
        checkpoint_unchanged=True, automatic_training_extension=False,
        new_run_elapsed_s=sum(read(p/'execution.json')['elapsed_s'] for policy,p in runs if policy=='clearance10100'),
        implementation_sources=[dict(path=str(p), sha256=sha(p)) for p in sorted((ROOT/'scripts/tools').glob('reference_low_stairs_equal100*.py'))]))
    save(DOCS/'raw_manifest.json', list(raw.values()))
    print('EQUAL100 ANALYSIS COMPLETE; no policy updates', flush=True)


if __name__ == '__main__':
    main()
