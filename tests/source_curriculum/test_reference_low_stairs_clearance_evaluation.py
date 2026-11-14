"""净空试训验收必须保留低速改进、高速保持和转向退化检查。"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
from reference_low_stairs_clearance_analyze import CASES, WINDOWS, OUTCOMES, screen, stairs_heights, rear_diagnostics
from reference_low_stairs_clearance_evaluate import BASELINE, MODEL, verify_run


def inputs(improved=False):
    heights, groups = [], []
    for profile in ('nominal', 'training_randomized'):
        for direction in ('up', 'down'):
            report = json.loads((BASELINE / f'{direction}_{profile}' / 'report.json').read_text())
            for policy in ('baseline', 'candidate'):
                heights.extend(dict(policy=policy, **r) for r in stairs_heights(report))
                groups.extend(dict(policy=policy, profile=profile, direction=direction,
                    **{k: v for k, v in g.items() if k != 'outcomes'},
                    **{k: g['outcomes'].get(k, 0) for k in OUTCOMES}) for g in report['groups'])
    if improved:
        for row in groups:
            if (row['policy'], row['profile'], row['direction'], row['height']) == ('candidate', 'nominal', 'up', .05):
                row['success'] = {.32: 1, .4: 5, .6: 14}[row['speed']]
    pairs = [dict(profile=p, case=c, window=w, n=20, baseline_mae=.2, candidate_mae=.2)
             for p in ('nominal', 'training_randomized') for c in CASES for w in WINDOWS]
    failures = [dict(profile='nominal', case='turn_left', new_failures=0)]
    return heights, groups, pairs, failures


def test_unchanged_parent_cannot_pass_the_improvement_requirement():
    result = screen(*inputs())
    assert result['status'] == 'failed'
    assert {r['name'] for r in result['checks'] if not r['passed']} == {
        'nominal_up_5cm_0.32_success', 'nominal_up_5cm_0.4_success'}


def test_agreed_minimum_improvement_and_preserved_abilities_pass():
    assert screen(*inputs(True))['status'] == 'passed'


def test_high_speed_loss_cannot_be_hidden_by_low_speed_gains():
    values = inputs(True)
    row = next(r for r in values[1] if (r['policy'], r['profile'], r['direction'], r['height'], r['speed']) ==
               ('candidate', 'nominal', 'up', .05, .6))
    row['success'] = 13
    assert screen(*values)['status'] == 'failed'


@pytest.mark.parametrize('change', ['late_turn_error', 'new_failure', 'missing_window', 'downstairs_loss'])
def test_other_ability_regressions_block_acceptance(change):
    values = inputs(True)
    if change == 'late_turn_error': values[2][-1]['candidate_mae'] = .221
    elif change == 'new_failure': values[3][0]['new_failures'] = 1
    elif change == 'missing_window': values[2].pop()
    else:
        row = next(r for r in values[0] if (r['policy'], r['profile'], r['direction'], r['height']) ==
                   ('candidate', 'nominal', 'down', .05))
        row['success'] -= 1
    assert screen(*values)['status'] == 'failed'


def test_randomized_regression_requires_review_even_when_nominal_passes():
    values = inputs(True)
    row = next(r for r in values[1] if r['policy'] == 'candidate' and r['profile'] == 'training_randomized')
    row['success'] -= 1
    result = screen(*values)
    assert result['status'] == 'requires_review' and result['engineering_checks_passed']
    assert len(result['random_decreases_require_review']) == 1


@pytest.mark.parametrize('field,value', [('model', 'wrong'), ('checkpoint_sha256', 'b' * 64),
    ('actor_unchanged', False), ('probe', True), ('action_mode', 'sampled'),
    ('first_episodes_only', False), ('height_set', 'reference'), ('video_env', 0)])
def test_wrong_checkpoint_or_evaluation_protocol_is_rejected(field, value, tmp_path):
    report = json.loads((BASELINE / 'up_nominal/report.json').read_text())
    report.update(model=MODEL, checkpoint_sha256='a' * 64)
    # 当前上下阶脚本使用 checkpoint_sha256；避免兼容字段优先级隐藏篡改。
    report.pop('model_sha256', None)
    for name, data in (('report', report), ('execution', dict(status='completed', exit_code=0, physical_gpu=1)),
                       ('config', dict(reward_scoring='reference_frozen_common'))):
        (tmp_path / f'{name}.json').write_text(json.dumps(data))
    verify_run(tmp_path, 'stairs', 'nominal', 'a' * 64, 'up')
    report[field] = value; (tmp_path / 'report.json').write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        verify_run(tmp_path, 'stairs', 'nominal', 'a' * 64, 'up')


def test_hind_contact_diagnostics_stop_before_first_tilt():
    feet = np.tile(np.array([[1.178, .1, .03], [1.178, -.1, .03], [1.178, .1, .03], [1.178, -.1, .03]]), (60, 1, 1, 1))
    forces = np.zeros_like(feet); forces[:, :, 3, 0] = -30.
    gravity = np.tile([0., 0., -1.], (60, 1, 1)); gravity[30:, :, 2] = -np.cos(.4)
    data = dict(active=np.ones((60, 1), bool), feet=feet, forces=forces, gravity=gravity)
    report = dict(profile='nominal', records=[dict(env_id=0, height=.05, speed=.6, column=0, outcome='orientation_failure')])
    rows = rear_diagnostics(report, dict(geometry=[dict(fronts=[1.2], tops=[.05])]), data)
    assert len(rows) == 4 and rows[3]['collision_frames_halfsecond'] == 25
    assert all(r['collision_frames_halfsecond'] == 0 for r in rows[:3])
    assert rows[3]['first_tilt20_s'] == pytest.approx(.62)


def test_evaluation_refuses_an_unfinished_training_run(monkeypatch):
    import reference_low_stairs_clearance_evaluate as module
    monkeypatch.setattr(module, 'read', lambda _: {'status': 'running'})
    with pytest.raises(AssertionError):
        module.verify_training()


@pytest.mark.parametrize('execution', [None, {'status': 'failed'}])
def test_waiting_never_restarts_missing_or_failed_training(execution, monkeypatch):
    import reference_low_stairs_clearance_evaluate as module
    monkeypatch.setattr(module, 'read', lambda _: execution)
    monkeypatch.setattr(module.time, 'sleep', lambda _: pytest.fail('失败任务不能继续等待'))
    with pytest.raises(RuntimeError):
        module.wait_training()


def test_waiting_starts_evaluation_only_after_supervisor_completion(monkeypatch):
    import reference_low_stairs_clearance_evaluate as module
    states = iter([{'status': 'running'}, {'status': 'completed'}]); waits = []
    monkeypatch.setattr(module, 'read', lambda _: next(states))
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: waits.append(seconds))
    module.wait_training()
    assert waits == [30]
