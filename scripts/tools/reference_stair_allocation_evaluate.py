"""Bounded frozen A/B evaluation, with pre-registered early regression gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from reference_stairs_eval import ROOT, save, sha
from reference_stair_allocation import OUTPUT, DIGEST, PARENT, parent_state, allocate, allocation_contract
from reference_stair_allocation_train import directory, sources, preflight
from reference_low_stairs_resume import STATE_KEY
from reference_low_stairs_guarded_ppo import KEY
from reference_low_stairs_clearance_train import check_algorithm
from reference_reward_resume import equal_state
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_low_stairs_zero_preservation_evaluate import verify_run

BASELINE = PARENT.parent.parent / 'evaluation_10100'
HEADING = ROOT / 'logs/reference_six/low_stairs_heading_stability_20261116'
EVALUATION = OUTPUT / 'evaluation'


def read(path): return json.loads(Path(path).read_text())


def checked_training(variant):
    import torch
    preflight(variant, 1)
    run = directory(variant, 1)
    r, e = read(run/'report.json'), read(run/'execution.json')
    assert r['status'] == 'passed' and e['status'] == 'completed' and e['physical_gpu'] == 1
    assert r['variant'] == variant and r['optimizer_updates'] == r['max_updates'] == 100
    assert r['environment_steps'] == 2400 and r['interactions'] == 9830400
    assert not r['automatic_extension'] and r['parent_sha256'] == DIGEST and r['sources'] == sources()
    assert [x['update'] for x in r['updates']] == list(range(1,101))
    assert all(x['losses']['capability_preservation'] == 0. for x in r['updates'])
    assert [x['phase_updates'] for x in r['checkpoints']] == [0,50,100]
    parent = parent_state(); old = parent['infos'][STATE_KEY]
    types, migration = allocate(old['terrain_types'], old['terrain_levels'], variant)
    contract = allocation_contract(old['contract'], migration)
    for row in r['checkpoints']:
        n, path = row['phase_updates'], Path(row['path'])
        assert sha(path) == row['sha256']
        s = torch.load(path, map_location='cpu', weights_only=True)
        check_algorithm(s, 100+n)
        assert s['iter'] == 10099+n and s['infos']['completed_updates'] == 10100+n
        assert s['infos'][KEY]['calls'] == 2000+20*n and s['infos'][KEY]['coefficient'] == 0.
        c = s['infos'][STATE_KEY]
        assert c['contract'] == contract and equal_state(c['terrain_types'], types)
        assert equal_state(c['terrain_origins'], old['terrain_origins'])
        assert equal_state(c['env_origins'], c['terrain_origins'][c['terrain_levels'], types])
        if n == 0:
            assert equal_state(c['terrain_levels'], old['terrain_levels'])
            assert all(equal_state(s[k], parent[k]) for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'))
    assert str(path) == r['output_checkpoint'] and sha(path) == r['output_sha256']
    assert read(run/'configuration.json') == read(PARENT.parent/'configuration.json')
    return path, r['output_sha256']


def jobs():
    result = [(f'heading_{d}_{p}', 'stairs', p, d, True)
              for d in ('up', 'down') for p in ('nominal', 'training_randomized')]
    result += [(f'turn_{p}', 'turn', p, None, False) for p in ('nominal', 'training_randomized')]
    result += [(f'{d}_{p}', 'stairs', p, d, False)
               for d in ('up', 'down') for p in ('nominal', 'training_randomized')]
    assert sum(320 if row[1] == 'turn' else 240 for row in result)*2 == 5120
    return result


def baseline_path(name):
    return HEADING/name if name.startswith('heading_') and name != 'heading_up_nominal' else BASELINE/name


def record_regressions(old, new, kind):
    a = {r['env_id']: r for r in old['records']}
    b = {r['env_id']: r for r in new['records']}
    if a.keys() != b.keys(): raise ValueError('unpaired record IDs')
    result = []
    for i, left in a.items():
        right = b[i]
        success = 'success' if kind == 'stairs' else 'survived'
        if left['outcome'] == success and right['outcome'] != success:
            result.append(dict(env_id=i, reason='lost_success', before=left, after=right))
        if right['outcome'] == 'orientation_failure' and (
            left['outcome'] != 'orientation_failure' or right['duration_s'] < left['duration_s']-1e-8):
            result.append(dict(env_id=i, reason='new_or_earlier_orientation_failure', before=left, after=right))
    return result


def turn_regressions(old_path, new_path):
    from reference_turn_analyze import analyze
    from reference_core500_analyze import paired_turn_rows
    old, _, a, _ = analyze(old_path)
    _, _, b, _ = analyze(new_path)
    pairs = paired_turn_rows(a, b, old['profile'])
    applicable = [r for r in pairs if r['case'] not in ('standing', 'forward', 'low_left', 'low_right')]
    bad = [dict(reason='yaw_mae_regression', **r) for r in applicable
           if r['mae_delta'] > .01 and (r['baseline_mae'] == 0. or r['mae_delta'] > .1*r['baseline_mae'])]
    return bad


def main():
    checkpoints = {v: checked_training(v) for v in ('A', 'B')}
    EVALUATION.mkdir(parents=True, exist_ok=True)
    plan = dict(core_episode_limit=5120, retention_episode_limit=960, total_episode_limit=6080,
        checkpoints={v: dict(path=str(p), sha256=h) for v,(p,h) in checkpoints.items()},
        jobs=jobs(), early_stop='B loses a prior success, gains/advances a posture failure, or fails paired yaw MAE gate',
        source_sha256=sha(__file__))
    if (EVALUATION/'complete.json').exists():
        raise RuntimeError('stage evaluation already ended; no automatic extension')
    if (EVALUATION/'plan.json').exists(): assert read(EVALUATION/'plan.json') == plan
    save(EVALUATION/'plan.json', plan)
    matrix = []
    for name, kind, profile, direction, heading in jobs():
        baseline = baseline_path(name)
        baseline_report = read(baseline/'report.json')
        assert read(baseline/'execution.json')['status'] == 'completed'
        assert baseline_report.get('model_sha256', baseline_report.get('checkpoint_sha256')) == DIGEST
        results = {}
        for variant in ('A', 'B'):
            path, digest = checkpoints[variant]
            model = f'allocation_{variant.lower()}10200'
            dest = EVALUATION/variant/name
            flags = ['--model', model, '--checkpoint', str(path), '--checkpoint-sha256', digest,
                     '--profile', profile, '--output', str(dest)]
            if heading:
                script = 'reference_stair_allocation_heading.py'; flags += ['--direction', direction]
            else:
                script = 'reference_turn_diagnose.py' if kind == 'turn' else 'reference_isaac_stairs_diagnose.py'
                flags += ['--repeats','20','--gpu','1','--minimum-free-mib','10240']
                if kind == 'stairs': flags += ['--direction',direction,'--height-set','low_stairs','--action-mode','mean']
            if not dest.exists():
                print('EVALUATE', variant, name, flush=True)
                subprocess.run([sys.executable, str(ROOT/'scripts/tools'/script), *flags], cwd=ROOT, check=True)
            results[variant] = verify_run(dest, kind, profile, digest, model, direction, heading=False)
            assert same_initial(dest, baseline)
            assert same_configuration(read(dest/'config.json'), read(baseline/'config.json'))
            if heading:
                meta = read(dest/'heading_intervention.json')
                assert meta['source_sha256'] == sha(ROOT/'scripts/tools/reference_stair_allocation_heading.py')
                assert meta['checkpoint_sha256'] == digest and meta['gain'] == .5 and meta['limit_rad_s'] == .8
                assert not meta['lateral_position_feedback'] and meta['optimizer_updates'] == 0
            matrix.append(dict(variant=variant, name=name, kind=kind, profile=profile, direction=direction,
                heading=heading, path=str(dest), baseline=str(baseline), episodes=results[variant]['episodes'],
                initial_exact=True, configuration_exact=True))
            save(EVALUATION/'matrix.json', matrix)
        regressions = []
        for comparator, old in (('zero10100',baseline_report), ('A',results['A'])):
            bad = record_regressions(old, results['B'], kind)
            if kind == 'turn':
                bad += turn_regressions(baseline if comparator == 'zero10100' else EVALUATION/'A'/name, EVALUATION/'B'/name)
            regressions += [dict(comparator=comparator, **r) for r in bad]
        save(EVALUATION/f'gate_{name}.json', dict(name=name, passed=not regressions, regressions=regressions))
        if regressions:
            save(EVALUATION/'complete.json', dict(status='stopped_regression', gate=name,
                core_episodes=sum(r['episodes'] for r in matrix), retention_episodes=0,
                not_run=[r[0] for r in jobs() if r[0] not in {x['name'] for x in matrix}],
                retention_not_run=True, regressions=regressions, matrix=matrix, promoted=False))
            print('REGRESSION GATE STOP:', name, flush=True)
            return
    # The separate protocol is only admitted after the core matrix passes.
    from reference_stair_allocation_retention import run_matrix
    retention = run_matrix(checkpoints)
    save(EVALUATION/'complete.json', dict(status='completed', core_episodes=5120,
        retention_episodes=960, retention=retention, matrix=matrix,
        promoted=False, note='Behavior acceptance requires the final analysis; no deployment.'))


if __name__ == '__main__': main()
