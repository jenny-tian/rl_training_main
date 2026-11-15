"""归档已完成的相位关闭对照；只读取现有轨迹，不启动仿真或训练。"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/lite3-low-stairs-yaw-phase-off-20261115'
OUTPUT = ROOT / 'logs/reference_six/low_stairs_yaw_phase_off_20261115'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(name):
    with (DOCS / f'{name}.csv').open() as stream:
        return list(csv.DictReader(stream))


def main():
    summary, verification = read(DOCS / 'summary.json'), read(DOCS / 'verification.json')
    assert summary['status'] == verification['status'] == 'passed'
    assert summary['screening']['status'] == 'failed'
    assert summary['updates'] == 100 and summary['formal_episodes'] == 1600
    complete = read(OUTPUT / 'evaluation_10100/complete.json')
    assert complete['status'] == 'passed' and complete['heading_episodes'] == 0
    assert complete['formal_episodes'] == 1600 and complete['optimizer_updates'] == 0
    executions = [read(OUTPUT / name / 'execution.json') for name in ('integration', 'train_001_100')]
    executions += [read(Path(r['candidate']) / 'execution.json') for r in complete['matrix']]
    assert len(executions) == 8
    for execution in executions:
        assert execution['status'] == 'completed' and execution['exit_code'] == 0
        assert execution['physical_gpu'] == 1
        assert not Path(f'/proc/{execution["pid"]}').exists(), execution
    checked = {}
    for entry in read(DOCS / 'raw_manifest.json'):
        path = Path(entry['path'])
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], path
        checked[str(path)] = entry['sha256']
    report = read(OUTPUT / 'train_001_100/report.json')
    sources = report['reference_sources'] + report['implementation_sources']
    for entry in sources:
        path = Path(entry['path'])
        assert sha(path) == entry['sha256'], path
        checked[str(path)] = entry['sha256']
    checkpoints = {
        'low_stairs_training/scratch_seed42_10000/model_updates_10000.pt':
            '419921855e24b40af058ac6642083eca8538f639e596606c51c2150ae7549a8f',
        'low_stairs_zero_preservation_20261115/train_001_100/model_updates_10100.pt':
            '290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497',
        'low_stairs_guarded_trial_20261115/train_001_100/model_updates_10100.pt':
            '603a3ea4f8937cdd312fa74267f6e0e30a70fb1913aaf8da9733bf4ec0afd859',
        'low_stairs_yaw_phase_off_20261115/train_001_100/model_updates_10100.pt':
            '2353502964ed2cdb3c9b95692e0b3af8b4daacaa37e643ce3a65f88d1adce8eb',
    }
    for name, digest in checkpoints.items():
        path = ROOT / 'logs/reference_six' / name
        assert sha(path) == digest, path
        checked[str(path)] = digest
    contacts = []
    for profile in ('nominal', 'training_randomized'):
        for speed in (.32, .4, .6):
            selected = [r for r in rows('up_matched_contacts')
                        if r['profile'] == profile and float(r['speed']) == speed]
            assert len(selected) == 20
            exposure = sum(float(r['exposure_s']) for r in selected)
            contacts.append(dict(profile=profile, speed=speed, starts=20,
                exposed_pairs=sum(float(r['exposure_s']) > 0 for r in selected),
                common_exposure_s=exposure,
                **{policy + '_collision_foot_fraction':
                   sum(float(r[policy + '_collision_foot_seconds']) for r in selected) / (4 * exposure)
                   if exposure else None for policy in ('zero', 'phaseoff')}))
    failures = {}
    for policy in ('parent', 'zero', 'phaseoff'):
        failures[policy] = [dict(env_id=int(r['env_id']), duration_s=float(r['duration_s']))
            for r in rows('turn_records') if r['policy'] == policy
            and r['profile'] == 'training_randomized' and r['outcome'] == 'orientation_failure']
    pair_failures = {}
    candidate = {r['env_id'] for r in failures['phaseoff']}
    for baseline in ('parent', 'zero'):
        old = {r['env_id'] for r in failures[baseline]}
        pair_failures[baseline] = dict(new=sorted(candidate-old), recovered=sorted(old-candidate),
                                      shared=sorted(old&candidate))
    save(DOCS / 'detail_summary.json', dict(turn_randomized_failures=failures,
        paired_turn_failures=pair_failures, up_5cm_matched_contacts=contacts))
    for name in ('README.md', 'NEXT_STAGE_PROPOSAL.md'):
        content = (DOCS / name).read_text()
        assert '\u201c' not in content and '\u201d' not in content
    save(DOCS / 'closeout.json', dict(
        completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        integrity_status='passed', capability_status='failed',
        all_eight_gpu_processes_exited=True, pids=[r['pid'] for r in executions],
        reference_and_implementation_sources_verified=len(sources),
        raw_files_verified=len(read(DOCS / 'raw_manifest.json')),
        checked_file_sha256=checked, tests_passed=verification['tests_passed'],
        training_updates=100, formal_frozen_episodes=1600,
        automatic_training_extension=False, deployed=False,
        next_stage_status='proposal_only_awaiting_user_approval'))
    files = [p for p in DOCS.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    files += list((ROOT / 'scripts/tools').glob('reference_low_stairs_yaw_phase_off_*.py'))
    files += list((ROOT / 'tests/source_curriculum').glob('test_reference_yaw_phase_off*.py'))
    save(DOCS / 'artifact_manifest.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                                         for p in sorted(files)])
    print(f'ARCHIVED: {len(files)} artifacts; 1600 episodes; capability failed; all GPU runs exited')


if __name__ == '__main__':
    main()
