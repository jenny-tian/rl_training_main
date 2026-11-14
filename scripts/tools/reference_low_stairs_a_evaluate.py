"""Evaluate only the approved completed A500 trial, with no further training."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from reference_stairs_eval import ROOT, sha
from reference_six_watch import read, save, snapshot
from reference_low_stairs_evaluate import OUTPUT as BASELINE, EXPECTED_SHA, verify_run as verify_baseline
from reference_low_stairs_trial import source_check
from reference_core500_evaluate import same_initial

TRAINING = ROOT / 'logs/reference_six/low_stairs_a_trial_20261114/riser_contact_seed42_500'
OUTPUT = ROOT / 'logs/reference_six/low_stairs_a_evaluation_20261114'
MODEL = 'lowstairs_a10500'


def same_configuration(left, right):
    ignored = {'model', 'checkpoint', 'model_sha256', 'checkpoint_sha256'}
    # This learned distribution parameter is recorded for provenance, but is
    # never sampled by the approved deterministic-mean stair protocol.
    if left.get('action_mode') == right.get('action_mode') == 'mean':
        ignored.add('action_std')
    return {k:v for k,v in left.items() if k not in ignored} == {k:v for k,v in right.items() if k not in ignored}


def verify_training():
    import torch
    execution, report = read(TRAINING / 'execution.json'), read(TRAINING / 'report.json')
    assert execution['status'] == 'completed' and execution['exit_code'] == 0
    assert execution['physical_gpu'] == 1
    assert report['status'] == 'passed' and report['completed_updates'] == 500
    assert report['cumulative_updates'] == 10500 and report['interactions'] == 49152000
    assert report['environment_steps'] == 12000 and not report['from_scratch']
    assert report['initial_sha256'] == EXPECTED_SHA and not report['automatic_extension']
    assert report['reward_variant'] == 'low_stairs_riser_contact' and report['repair_variant'] == 'riser_contact'
    assert read(TRAINING / 'first20_check.json')['status'] == 'passed'
    assert [r['phase_updates'] for r in report['checkpoints']] == [100, 250, 500]
    for row in report['checkpoints']:
        assert sha(row['path']) == row['sha256']
    checkpoint = Path(report['output_checkpoint'])
    digest = sha(checkpoint)
    assert digest == report['output_sha256'] == report['checkpoints'][-1]['sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    info = state['infos']
    assert state['iter'] == 10499 and info['completed_updates'] == 10500 and info['phase_updates'] == 500
    assert info['trial_budget'] == 500 and info['parent_sha256'] == EXPECTED_SHA
    assert not info['from_scratch'] and not info['validation_only']
    assert info['terrain_profile'] == 'low_stairs' and info['repair_variant'] == 'riser_contact'
    assert info['reward_variant'] == 'low_stairs_riser_contact'
    assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()} == {210000}
    assert all(torch.isfinite(v).all() for k in ('actor_state_dict', 'critic_state_dict') for v in state[k].values())
    assert state['actor_state_dict']['mlp.0.weight'].shape == (512, 45)
    assert state['critic_state_dict']['mlp.0.weight'].shape == (512, 393)
    assert {g['lr'] for g in state['optimizer_state_dict']['param_groups']} == {report['final_learning_rate']}
    source_check()
    for row in read(TRAINING / 'implementation_sources.json'):
        assert sha(row['path']) == row['sha256']
    return checkpoint, digest


def verify_run(directory, kind, profile, digest, direction=None, video_env=None):
    result, execution, config = (read(directory / f'{name}.json') for name in ('report', 'execution', 'config'))
    assert execution['status'] == 'completed' and execution['exit_code'] == 0 and execution['physical_gpu'] == 1
    assert result['status'] == 'passed' and result['actor_unchanged'] and not result['probe']
    assert result.get('model_sha256', result.get('checkpoint_sha256')) == digest
    assert result['model'] == MODEL and result['profile'] == profile and result['video_env'] == video_env
    expected = 320 if kind == 'turn' else 240
    assert result['episodes'] == len(result['records']) == expected
    assert len({r['env_id'] for r in result['records']}) == expected
    assert config['reward_scoring'] == 'reference_frozen_common'
    if kind == 'stairs':
        assert result['height_set'] == 'low_stairs' and result['direction'] == direction
        assert result['action_mode'] == 'mean' and result['first_episodes_only'] and result['pre_reset_capture']
        assert {r['height'] for r in result['records']} == {0., .02, .03, .05}
        assert {r['speed'] for r in result['records']} == {.32, .4, .6}
    return result


def wait_training(wait):
    deadline = time.monotonic() + 21600
    while True:
        execution = read(TRAINING / 'execution.json')
        if execution and execution['status'] == 'failed':
            raise RuntimeError('A trial failed; no restart or evaluation of an incomplete checkpoint')
        if execution and execution['status'] == 'completed':
            return
        if not wait or time.monotonic() >= deadline:
            raise RuntimeError('A trial has not completed within the waiting budget')
        progress = snapshot(TRAINING)
        if progress:
            save(OUTPUT / 'training_progress.json', progress)
            if progress['status'] == 'awaiting_supervisor_result':
                time.sleep(5)
                if read(TRAINING / 'execution.json')['status'] == 'running':
                    raise RuntimeError('A worker exited without supervisor completion')
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(1,), default=1)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wait_training(args.wait_for_training)
    checkpoint, digest = verify_training()
    save(OUTPUT / 'training_progress.json', snapshot(TRAINING))
    save(OUTPUT / 'plan.json', dict(model=MODEL, checkpoint=str(checkpoint), sha256=digest,
        baseline=str(BASELINE), baseline_sha256=EXPECTED_SHA, candidate_episodes=1600,
        physical_gpu=1, repeats_per_case=20, minimum_free_mib=10240,
        heights_m=[0., .02, .03, .05], speeds_m_s=[.32, .4, .6],
        profiles=['nominal', 'training_randomized'], training=False,
        video_selection='fixed repeat 0: up 5 cm at 0.32 m/s and right-to-left turn',
        video_replays_excluded_from_formal_count=True,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    matrix, videos = [], []

    def run(name, kind, profile, direction=None, video_env=None):
        directory = OUTPUT / name
        script = 'reference_turn_diagnose.py' if kind == 'turn' else 'reference_isaac_stairs_diagnose.py'
        flags = ['--model', MODEL, '--checkpoint', str(checkpoint), '--checkpoint-sha256', digest,
                 '--profile', profile, '--repeats', '20']
        if kind == 'stairs':
            flags += ['--direction', direction, '--height-set', 'low_stairs', '--action-mode', 'mean']
        if video_env is not None:
            flags += ['--video-env', str(video_env)]
        if not directory.exists():
            print('EVALUATE', name, flush=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts/tools' / script), *flags,
                '--gpu', '1', '--minimum-free-mib', '10240', '--output', str(directory)], cwd=ROOT, check=True)
        result = verify_run(directory, kind, profile, digest, direction, video_env)
        assert sha(checkpoint) == digest
        return dict(name=name, kind=kind, profile=profile, direction=direction,
                    candidate=str(directory), episodes=result['episodes'], video_env=video_env)

    for profile in ('nominal', 'training_randomized'):
        for kind, direction, name in (('stairs', 'up', f'up_{profile}'),
                                      ('stairs', 'down', f'down_{profile}'),
                                      ('turn', None, f'turn_{profile}')):
            baseline = BASELINE / name
            verify_baseline(baseline, kind, profile, direction)
            entry = run(name, kind, profile, direction)
            assert same_initial(Path(entry['candidate']), baseline), f'initial state mismatch: {name}'
            assert same_configuration(read(Path(entry['candidate'])/'config.json'),read(baseline/'config.json'))
            entry.update(baseline=str(baseline), initial_equal=True)
            matrix.append(entry)
            save(OUTPUT / 'matrix.json', matrix)
    assert sum(r['episodes'] for r in matrix) == 1600
    save(OUTPUT / 'formal_complete.json', dict(status='passed', candidate_episodes=1600,
        candidate_sha256=digest, matrix=matrix))
    for name, kind, direction, env_id, formal in (
        ('video_up_5cm_slow', 'stairs', 'up', 180, 'up_nominal'),
        ('video_turn', 'turn', None, 220, 'turn_nominal'),
    ):
        entry = run(name, kind, 'nominal', direction, env_id)
        entry['formal'] = str(OUTPUT / formal)
        videos.append(entry)
        save(OUTPUT / 'videos.json', videos)
    save(OUTPUT / 'complete.json', dict(status='passed', candidate_episodes=1600,
        status_scope='execution complete; behavior acceptance requires analysis',
        candidate_sha256=digest, matrix=matrix, videos=videos, automatic_training_extension=False))
    print('LOW STAIRS A MATRIX COMPLETE: 1600 paired formal episodes and two separate video replays', flush=True)


if __name__ == '__main__':
    main()
