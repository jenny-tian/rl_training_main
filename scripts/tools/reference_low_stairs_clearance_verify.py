"""核验配对冻结证据，并归档本阶段结果。不会启动仿真或训练。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import torch

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_trial import source_check
from reference_low_stairs_clearance import height_cost
from reference_low_stairs_rewards import clearance_cost
from source_block_runtime import fatal_line


def read(path):
    return json.loads(path.read_text())


def verify(raw, output):
    output.mkdir(parents=True, exist_ok=True)
    reports = []; traces = []; configurations = []
    for variant, directory in (('baseline', 'baseline_final'), ('candidate', 'candidate_v2')):
        path = raw / directory
        report = read(path / 'report.json'); execution = read(path / 'execution.json')
        config = read(path / 'configuration.json')
        assert report['status'] == 'passed' and report['variant'] == variant
        assert report['optimizer_updates'] == 0 and report['actor_critic_optimizer_unchanged']
        assert report['environments'] == 4096 and report['steps'] == 360
        assert report['selective_reset_verified'] and report['same_step_cache_verified']
        assert report['gait_range'] == [1., 1.]
        assert report['exact_reward_max_error'] < 1e-6
        assert report['configuration_audit']['changed_terms'] == ['stair_clearance']
        assert execution['status'] == 'completed' and execution['exit_code'] == 0 and execution['physical_gpu'] == 1
        assert execution['failure'] is None and not execution['fatal_messages']
        assert not any(fatal_line(line) for line in (path / 'runtime.log').read_text().splitlines())
        for source in report['sources']:
            assert sha(source['path']) == source['sha256'], source['path']
        assert sha(report['checkpoint']) == report['checkpoint_sha256']
        with np.load(path / 'trace.npz') as trace:
            traces.append(dict(trace))
        reports.append(report); configurations.append(config)
        for name in ('report.json', 'execution.json', 'configuration.json'):
            shutil.copy2(path / name, output / f'{variant}_{name}')
    a, b = reports
    assert a['hashes'] == b['hashes'], '物理轨迹、其他奖励或同轨迹计分存在差异'
    assert a['counts_first_episode_per_foot'] == b['counts_first_episode_per_foot']
    assert a['scope_counts'] == b['scope_counts']
    assert a['initial_histogram'] == b['initial_histogram']
    assert a['sources'] == b['sources']
    assert a['checkpoint_sha256'] == b['checkpoint_sha256']
    assert traces[0].keys() == traces[1].keys()
    assert all(np.isfinite(value).all() for trace in traces for value in trace.values())
    assert all(np.array_equal(traces[0][k], traces[1][k]) for k in traces[0])
    ca, cb = (copy.deepcopy(c) for c in configurations)
    ca['environment']['rewards']['stair_clearance'] = cb['environment']['rewards']['stair_clearance']
    assert ca == cb, '净空项以外的配置发生变化'
    assert all(n > 0 for n in a['counts_first_episode_per_foot']['wall_old_zero_new_positive'])
    suites = ET.parse(raw / 'tests.xml').getroot().findall('testsuite')
    assert suites and all(int(s.get(k)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    tests = sum(int(s.get('tests')) for s in suites)
    assert tests >= 89
    shutil.copy2(raw / 'tests.xml', output / 'tests.xml')
    # 验证历史训练与上一阶段恢复源码均未修改。
    historical = source_check()
    previous = read(ROOT / 'docs/lite3-low-stairs-resume-repair-20261115/capture_report.json')
    for row in previous['sources']:
        assert sha(row['path']) == row['sha256']
    rows = []
    for bottom in (0., .01, .02, .025, .03, .05, .075, .10):
        bottoms = torch.full((1, 4), bottom); tops = torch.full((1, 4), .05)
        force = torch.zeros(1, 4, 3); velocity = force.clone(); velocity[..., 0] = .5
        old = clearance_cost(bottoms, tops, force, velocity, torch.tensor([[1., 0.]]), torch.ones(1, 4, dtype=torch.bool))
        rows.append(dict(foot_bottom_cm=bottom * 100., old_cost=float(old),
                         new_active_swing_cost=float(height_cost(bottoms, tops).mean())))
    analysis = dict(status='passed', foot_order=['FL', 'FR', 'HL', 'HR'], height_examples=rows,
        exact_paired_hashes=a['hashes'], all_selected_trace_arrays_exact=True,
        same_terrain_commands_physics_ppo=True, only_clearance_changed=True,
        counts_first_episode_per_foot=a['counts_first_episode_per_foot'],
        scope_counts=a['scope_counts'], mean_cost_old_new=a['mean_cost_old_new'],
        note='样本数量是逐足控制帧，不能当成独立回合或成功次数。新的高度曲线和触发时长均已改变，不声称惩罚总量保持。')
    save(output / 'integration_analysis.json', analysis)
    files = [ROOT / 'scripts/tools' / name for name in (
        'reference_low_stairs_clearance.py', 'reference_low_stairs_clearance_cfg.py',
        'reference_low_stairs_clearance_validate.py', 'reference_low_stairs_clearance_verify.py')]
    files.append(ROOT / 'tests/source_curriculum/test_reference_low_stairs_clearance.py')
    save(output / 'verification.json', dict(status='passed', tests_passed=tests, optimizer_updates=0,
        frozen_control_steps=2 * 4096 * 360, paired_hashes_equal=True, checkpoint_sha256=a['checkpoint_sha256'],
        actor_critic_optimizer_unchanged=True, historical_sources=historical,
        stage1_sources_unchanged=True, next_training_started=False,
        sources=[dict(path=str(p), sha256=sha(p)) for p in files]))
    save(output / 'raw_manifest.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(raw.rglob('*')) if p.is_file()])
    print(json.dumps(dict(status='passed', tests=tests, analysis=analysis), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'logs/reference_six/low_stairs_clearance_validation_20261115')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/lite3-low-stairs-clearance-repair-20261115')
    args = parser.parse_args()
    verify(args.raw, args.output)


if __name__ == '__main__':
    main()
