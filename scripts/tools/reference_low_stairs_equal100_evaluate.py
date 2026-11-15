"""对已有净空 10100 检查点执行 1360 回合冻结对照，不启动训练。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_clearance_evaluate import verify_training as clearance_training
from reference_low_stairs_clearance_train import TRAINING, check_algorithm
from reference_low_stairs_guarded_evaluate import checked_training, read
from reference_low_stairs_evaluate import OUTPUT as PARENT_RUNS, EXPECTED_SHA, verify_run as parent_run
from reference_low_stairs_guarded_bank import OUTPUT as GUARDED
from reference_low_stairs_a_evaluate import same_configuration
from reference_core500_evaluate import same_initial

OUTPUT = ROOT / 'logs/reference_six/low_stairs_equal100_20261115'
DOCS = ROOT / 'docs/lite3-low-stairs-equal100-20261115'
MODEL = 'lowstairs_clearance10100'
EXPECTED = 'd6bf0f7bf696e88e57ebc91983491de780058ea36cedc8dfa780b0dcafa4b982'
GUARDED_SHA = '603a3ea4f8937cdd312fa74267f6e0e30a70fb1913aaf8da9733bf4ec0afd859'
GUARDED_RUNS = GUARDED / 'evaluation_10100'


def preflight():
    import torch
    final, final_sha = clearance_training()
    report = read(TRAINING / 'report.json')
    row = next(r for r in report['checkpoints'] if r['phase_updates'] == 100)
    checkpoint = Path(row['path'])
    assert sha(checkpoint) == row['sha256'] == EXPECTED
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    check_algorithm(state, 100)
    info = state['infos']
    assert state['iter'] == 10099 and info['completed_updates'] == 10100
    assert info['phase_updates'] == 100 and info['parent_sha256'] == EXPECTED_SHA
    assert not info['validation_only'] and not info['from_scratch']
    assert info['reward_variant'] == 'low_stairs_persistent_clearance'
    assert state['actor_state_dict']['mlp.0.weight'].shape == (512, 45)
    assert state['critic_state_dict']['mlp.0.weight'].shape == (512, 393)
    guarded, guarded_sha = checked_training(100)
    assert guarded_sha == GUARDED_SHA
    return checkpoint, dict(checkpoint=str(checkpoint), checkpoint_sha256=EXPECTED,
        parent_sha256=EXPECTED_SHA, phase_updates=100, cumulative_updates=10100,
        adam_steps=202000, guarded_checkpoint=str(guarded), guarded_sha256=guarded_sha,
        clearance500_checkpoint=str(final), clearance500_sha256=final_sha,
        historical_training_report=str(TRAINING / 'report.json'),
        historical_training_report_sha256=sha(TRAINING / 'report.json'))


def verify_run(directory, kind, profile, digest, model, heading=False):
    report, execution, config = (read(directory / f'{n}.json') for n in ('report', 'execution', 'config'))
    assert execution['status'] == 'completed' and execution['exit_code'] == 0
    assert execution['physical_gpu'] == 1
    assert report['status'] == 'passed' and report['actor_unchanged'] and not report['probe']
    assert report.get('model_sha256', report.get('checkpoint_sha256')) == digest
    assert report['model'] == model and report['profile'] == profile and report['video_env'] is None
    expected = 320 if kind == 'turn' else 240
    assert report['episodes'] == len(report['records']) == expected
    assert {r['env_id'] for r in report['records']} == set(range(expected))
    assert report['step_dt'] == .02
    assert config['reward_scoring'] == 'reference_frozen_common'
    if kind == 'stairs':
        assert report['height_set'] == 'low_stairs' and report['direction'] == 'up'
        assert report['action_mode'] == 'mean' and report['first_episodes_only'] and report['pre_reset_capture']
        assert {r['height'] for r in report['records']} == {0., .02, .03, .05}
        assert {r['speed'] for r in report['records']} == {.32, .4, .6}
    if heading:
        intervention = read(directory / 'heading_intervention.json')
        assert intervention['checkpoint_sha256'] == digest
        assert intervention['diagnostic_only'] and intervention['optimizer_updates'] == 0
        assert intervention['gain'] == .5 and intervention['limit_rad_s'] == .8
        assert intervention['target'] == 'world +x' and not intervention['lateral_position_feedback']
        assert intervention['source_sha256'] == sha(ROOT / 'scripts/tools/reference_low_stairs_guarded_heading.py')
    return report


def main(check_only=False):
    checkpoint, lineage = preflight()
    if check_only:
        print(lineage, flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT / 'plan.json', dict(**lineage, model=MODEL, new_episodes=1360,
        formal_episodes=1120, heading_episodes=240, training_updates=0, physical_gpu=1,
        minimum_free_mib=10240, repeats_per_case=20, heights_m=[0., .02, .03, .05],
        speeds_m_s=[.32, .4, .6], profiles=['nominal', 'training_randomized'],
        automatic_training_extension=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources=[dict(path=str(p), sha256=sha(p)) for p in [Path(__file__), DOCS / 'PLAN.md']]))
    matrix = []
    jobs = [(f'turn_{p}', 'turn', p, False) for p in ('nominal', 'training_randomized')]
    jobs += [(f'up_{p}', 'stairs', p, False) for p in ('nominal', 'training_randomized')]
    jobs += [('heading_up_nominal', 'stairs', 'nominal', True)]
    for name, kind, profile, heading in jobs:
        dest = OUTPUT / name
        baseline = PARENT_RUNS / ('up_nominal' if heading else name)
        guarded = GUARDED_RUNS / name
        parent_run(baseline, kind, profile, 'up' if kind == 'stairs' else None)
        verify_run(guarded, kind, profile, GUARDED_SHA, 'lowstairs_guarded10100', heading)
        script = ('reference_low_stairs_guarded_heading.py' if heading else
                  'reference_turn_diagnose.py' if kind == 'turn' else 'reference_isaac_stairs_diagnose.py')
        flags = ['--model', MODEL, '--checkpoint', str(checkpoint), '--checkpoint-sha256', EXPECTED,
                 '--output', str(dest)]
        if not heading:
            flags += ['--profile', profile, '--repeats', '20', '--gpu', '1', '--minimum-free-mib', '10240']
            if kind == 'stairs':
                flags += ['--direction', 'up', '--height-set', 'low_stairs', '--action-mode', 'mean']
        if not dest.exists():
            print('EVALUATE', name, flush=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts/tools' / script), *flags], cwd=ROOT, check=True)
        result = verify_run(dest, kind, profile, EXPECTED, MODEL, heading)
        for old in (baseline, guarded):
            assert same_initial(dest, old), (name, old)
            assert same_configuration(read(dest / 'config.json'), read(old / 'config.json')), (name, old)
        assert sha(checkpoint) == EXPECTED
        matrix.append(dict(name=name, kind=kind, profile=profile, heading=heading,
            candidate=str(dest), parent=str(baseline), guarded=str(guarded), episodes=result['episodes'],
            initial_equal=True, configuration_equal=True,
            parent_path_scope='initial/config only' if heading else 'formal comparison'))
        save(OUTPUT / 'matrix.json', matrix)
    assert sum(r['episodes'] for r in matrix) == 1360
    save(OUTPUT / 'complete.json', dict(status='passed', status_scope='冻结评估完成，能力结论另行分析',
        **lineage, new_episodes=1360, formal_episodes=1120, heading_episodes=240,
        training_updates=0, automatic_training_extension=False, matrix=matrix))
    print('EQUAL100 COMPLETE: 1360 frozen episodes; zero training updates', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true')
    main(p.parse_args().check_only)
