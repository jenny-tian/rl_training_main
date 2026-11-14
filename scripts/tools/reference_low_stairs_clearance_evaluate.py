"""仅评估完成 500 次的新净空模型；沿用父模型的 1600 回合和两段录像协议。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from reference_stairs_eval import ROOT, save, sha
from reference_six_watch import read
from reference_low_stairs_clearance_train import TRAINING, VARIANT, source_preflight, check_algorithm
from reference_low_stairs_resume import STATE_KEY
from reference_low_stairs_evaluate import OUTPUT as BASELINE, EXPECTED_SHA, verify_run as verify_baseline
from reference_low_stairs_a_evaluate import same_configuration
from reference_core500_evaluate import same_initial

OUTPUT = ROOT / 'logs/reference_six/low_stairs_clearance_evaluation_20261115'
MODEL = 'lowstairs_clearance10500'


def wait_training():
    """只等待本次已启动任务，失败后停止，不重启训练。"""
    deadline = time.monotonic() + 21600
    while True:
        execution = read(TRAINING / 'execution.json')
        if not execution or execution['status'] not in ('running', 'completed'):
            raise RuntimeError('本次训练未启动或已失败；不会重启或评估未完成模型')
        if execution['status'] == 'completed':
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('等待训练完成超时')
        time.sleep(30)


def verify_training():
    import torch
    from reference_low_stairs_trial import checked_state
    from reference_reward_resume import equal_state
    execution, report = read(TRAINING / 'execution.json'), read(TRAINING / 'report.json')
    assert execution['status'] == 'completed' and execution['exit_code'] == 0 and execution['physical_gpu'] == 1
    assert report['status'] == 'passed' and report['completed_updates'] == report['budget'] == 500
    assert report['cumulative_updates'] == 10500 and report['interactions'] == 49152000
    assert report['environment_steps'] == 12000 and not report['from_scratch']
    assert report['initial_sha256'] == EXPECTED_SHA and not report['automatic_extension']
    assert report['repair_variant'] == VARIANT and report['reward_variant'] == 'low_stairs_persistent_clearance'
    assert read(TRAINING / 'first20_check.json')['status'] == 'passed'
    assert [r['phase_updates'] for r in report['checkpoints']] == [0, 100, 250, 500]
    parent, _ = checked_state()
    final = None
    for row in report['checkpoints']:
        assert sha(row['path']) == row['sha256']
        state = torch.load(row['path'], map_location='cpu', weights_only=True)
        n = row['phase_updates']; info = state['infos']; curriculum = info[STATE_KEY]
        check_algorithm(state, n)
        assert state['iter'] == 9999 + n and info['completed_updates'] == 10000 + n and info['phase_updates'] == n
        assert info['trial_budget'] == 500 and info['parent_sha256'] == EXPECTED_SHA
        assert info['validation_only'] == (n == 0) and not info['from_scratch']
        assert info['repair_variant'] == VARIANT and info['reward_variant'] == 'low_stairs_persistent_clearance'
        assert info['terrain_profile'] == 'low_stairs'
        assert curriculum['mode'] == 'exact_curriculum' and curriculum['contract'] == report['curriculum_contract']
        levels, types = curriculum['terrain_levels'], curriculum['terrain_types']
        assert levels.shape == types.shape == (4096,) and levels.min() >= 0 and levels.max() < 10
        assert torch.equal(curriculum['env_origins'], curriculum['terrain_origins'][levels, types])
        if n == 0:
            assert all(equal_state(state[k], parent[k]) for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'))
            assert torch.bincount(levels, minlength=10).tolist() == report['initial_level_histogram']
            assert curriculum['gait_level'] == 1.
        if n == 500:
            final = state
            assert torch.bincount(levels, minlength=10).tolist() == report['final_terrain_level_histogram']
    checkpoint = Path(report['output_checkpoint']); digest = sha(checkpoint)
    assert digest == report['output_sha256'] == report['checkpoints'][-1]['sha256']
    assert checkpoint == Path(report['checkpoints'][-1]['path'])
    assert final['actor_state_dict']['mlp.0.weight'].shape == (512, 45)
    assert final['critic_state_dict']['mlp.0.weight'].shape == (512, 393)
    assert {g['lr'] for g in final['optimizer_state_dict']['param_groups']} == {report['final_learning_rate']}
    source_preflight()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    if args.wait_for_training:
        print('WAITING FOR APPROVED CLEARANCE TRAINING; no restart or extension', flush=True)
        wait_training()
    checkpoint, digest = verify_training()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT / 'plan.json', dict(model=MODEL, checkpoint=str(checkpoint), sha256=digest,
        baseline=str(BASELINE), baseline_sha256=EXPECTED_SHA, candidate_episodes=1600, physical_gpu=1,
        repeats_per_case=20, minimum_free_mib=10240, heights_m=[0., .02, .03, .05],
        speeds_m_s=[.32, .4, .6], profiles=['nominal', 'training_randomized'], training=False,
        video_selection='固定 repeat 0：5 cm 上阶 0.32 m/s 和先右后左转向',
        video_replays_excluded_from_formal_count=True,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=sha(Path(__file__))))
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
                                      ('stairs', 'down', f'down_{profile}'), ('turn', None, f'turn_{profile}')):
            baseline = BASELINE / name
            verify_baseline(baseline, kind, profile, direction)
            entry = run(name, kind, profile, direction)
            assert same_initial(Path(entry['candidate']), baseline), f'初始状态不一致: {name}'
            assert same_configuration(read(Path(entry['candidate']) / 'config.json'), read(baseline / 'config.json'))
            entry.update(baseline=str(baseline), initial_equal=True)
            matrix.append(entry); save(OUTPUT / 'matrix.json', matrix)
    assert sum(r['episodes'] for r in matrix) == 1600
    save(OUTPUT / 'formal_complete.json', dict(status='passed', candidate_episodes=1600, candidate_sha256=digest, matrix=matrix))
    for name, kind, direction, env_id, formal in (
        ('video_up_5cm_slow', 'stairs', 'up', 180, 'up_nominal'),
        ('video_turn', 'turn', None, 220, 'turn_nominal')):
        entry = run(name, kind, 'nominal', direction, env_id)
        entry['formal'] = str(OUTPUT / formal); videos.append(entry); save(OUTPUT / 'videos.json', videos)
    save(OUTPUT / 'complete.json', dict(status='passed', candidate_episodes=1600,
        status_scope='评估运行完成；能力验收另行分析', candidate_sha256=digest,
        matrix=matrix, videos=videos, automatic_training_extension=False))
    print('CLEARANCE MATRIX COMPLETE: 1600 paired episodes and two video replays', flush=True)


if __name__ == '__main__':
    main()
