"""Run the approved 1600 frozen low-stair episodes and fixed-start video replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from reference_stairs_eval import ROOT, save, sha

TRAINING = ROOT / 'logs/reference_six/low_stairs_training/scratch_seed42_10000'
OUTPUT = ROOT / 'logs/reference_six/low_stairs_evaluation_20261114'
MODEL = 'lowstairs10000'
EXPECTED_SHA = '419921855e24b40af058ac6642083eca8538f639e596606c51c2150ae7549a8f'


def read(path):
    return json.loads(Path(path).read_text())


def verify_training():
    import torch
    report = read(TRAINING / 'report.json')
    execution = read(TRAINING / 'execution.json')
    assert execution['status'] == 'completed' and execution['exit_code'] == 0
    assert report['status'] == 'passed' and report['completed_updates'] == 10000
    assert report['interactions'] == 983040000 and report['checkpoint_loaded'] is None
    checkpoint = Path(report['output_checkpoint'])
    assert sha(checkpoint) == report['output_sha256'] == EXPECTED_SHA
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    info = state['infos']
    assert state['iter'] == 9999
    assert info['completed_updates'] == info['phase_updates'] == 10000
    assert info['from_scratch'] and not info['validation_only']
    assert info['terrain_profile'] == info['reward_variant'] == 'low_stairs'
    assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()} == {200000}
    assert all(torch.isfinite(v).all() for k in ('actor_state_dict', 'critic_state_dict') for v in state[k].values())
    return checkpoint


def verify_run(directory, kind, profile, direction=None, video_env=None):
    result = read(directory / 'report.json')
    execution = read(directory / 'execution.json')
    config = read(directory / 'config.json')
    assert execution['status'] == 'completed' and execution['exit_code'] == 0
    assert execution['physical_gpu'] == 1
    assert result['status'] == 'passed' and result['actor_unchanged'] and not result['probe']
    assert result.get('model_sha256', result.get('checkpoint_sha256')) == EXPECTED_SHA
    assert result['model'] == MODEL and result['profile'] == profile
    assert result['video_env'] == video_env
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(1,), default=1)
    args = parser.parse_args()
    checkpoint = verify_training()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT / 'plan.json', dict(model=MODEL, checkpoint=str(checkpoint), sha256=EXPECTED_SHA,
        candidate_episodes=1600, physical_gpu=1, repeats_per_case=20,
        minimum_free_mib=10240,
        memory_admission_basis='Historical 240/320-env runs used about 2.8 GiB extra; camera replays about 5.5 GiB. This run requires 10 GiB free; training defaults remain 12 GiB.',
        heights_m=[0., .02, .03, .05], speeds_m_s=[.32, .4, .6],
        profiles=['nominal', 'training_randomized'], training=False,
        video_selection='fixed repeat 0, 0.4 m/s; up 3/5 cm, down 5 cm, right-to-left turn',
        video_replays_excluded_from_formal_count=True,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    matrix, videos = [], []

    def run(name, kind, profile, direction=None, video_env=None):
        directory = OUTPUT / name
        script = 'reference_turn_diagnose.py' if kind == 'turn' else 'reference_isaac_stairs_diagnose.py'
        flags = ['--model', MODEL, '--checkpoint', str(checkpoint), '--checkpoint-sha256', EXPECTED_SHA,
                 '--profile', profile, '--repeats', '20']
        if kind == 'stairs':
            flags += ['--direction', direction, '--height-set', 'low_stairs', '--action-mode', 'mean']
        if video_env is not None:
            flags += ['--video-env', str(video_env)]
        command = [sys.executable, str(ROOT / 'scripts/tools' / script), *flags,
                   '--gpu', str(args.gpu), '--minimum-free-mib', '10240', '--output', str(directory)]
        if not directory.exists():
            print('EVALUATE', name, flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        result = verify_run(directory, kind, profile, direction, video_env)
        assert sha(checkpoint) == EXPECTED_SHA
        entry = dict(name=name, kind=kind, profile=profile, direction=direction,
                     candidate=str(directory), episodes=result['episodes'])
        if video_env is not None:
            entry['video_env'] = video_env
        return entry

    for profile in ('nominal', 'training_randomized'):
        for direction in ('up', 'down'):
            matrix.append(run(f'{direction}_{profile}', 'stairs', profile, direction))
            save(OUTPUT / 'matrix.json', matrix)
        matrix.append(run(f'turn_{profile}', 'turn', profile))
        save(OUTPUT / 'matrix.json', matrix)
    assert sum(r['episodes'] for r in matrix) == 1600
    save(OUTPUT / 'formal_complete.json', dict(status='passed', candidate_episodes=1600,
        candidate_sha256=EXPECTED_SHA, matrix=matrix))
    for name, kind, direction, env_id, formal in (
        ('video_up_3cm', 'stairs', 'up', 140, 'up_nominal'),
        ('video_up_5cm', 'stairs', 'up', 200, 'up_nominal'),
        ('video_down_5cm', 'stairs', 'down', 200, 'down_nominal'),
        ('video_turn', 'turn', None, 220, 'turn_nominal'),
    ):
        entry = run(name, kind, 'nominal', direction, env_id)
        entry['formal'] = str(OUTPUT / formal)
        videos.append(entry)
        save(OUTPUT / 'videos.json', videos)
    save(OUTPUT / 'complete.json', dict(status='passed', candidate_episodes=1600,
        status_scope='execution complete; behavior acceptance requires analysis',
        candidate_sha256=EXPECTED_SHA, matrix=matrix, videos=videos,
        automatic_training_extension=False))
    print('LOW STAIRS MATRIX COMPLETE: 1600 formal episodes, four separate video replays', flush=True)


if __name__ == '__main__':
    main()
