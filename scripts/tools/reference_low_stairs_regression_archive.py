"""归档本阶段冻结诊断、验证和原始文件哈希，不重新运行仿真。"""
from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from reference_stairs_eval import save, sha
from reference_low_stairs_regression_probe import OUTPUT, PARENT, CANDIDATE, HASHES
from reference_low_stairs_regression_analyze import OUT, read, verify_replay, npz
from reference_low_stairs_evaluate import OUTPUT as BASELINE
from reference_low_stairs_clearance_evaluate import OUTPUT as NEW_EVALUATION
from reference_low_stairs_clearance_train import source_preflight


def main():
    source_preflight()
    assert sha(PARENT) == HASHES['parent'] and sha(CANDIDATE) == HASHES['candidate']
    analysis, preservation = read(OUT / 'analysis.json'), read(OUT / 'preservation_validation.json')
    assert analysis['initial_failures'] == {'parent': 3, 'candidate': 23}
    assert preservation['status'] == 'passed' and not preservation['integrated_into_ppo']
    tests = ET.parse(OUT / 'tests.xml').getroot().find('testsuite')
    assert int(tests.attrib['tests']) == 85 and int(tests.attrib['errors']) == int(tests.attrib['failures']) == 0
    checks = []
    for name, historical in (
        ('turn_candidate_native_v2', NEW_EVALUATION / 'turn_training_randomized'),
        ('turn_parent_native', BASELINE / 'turn_training_randomized'),
        ('turn_parent_native_8s', BASELINE / 'turn_training_randomized'),
        ('turn_parent_nominal_8s', BASELINE / 'turn_nominal')):
        checks.append(verify_replay(OUTPUT / name, historical))
    # 补录的实际 PD 参数也使用同一次物理随机化；PhysX 中的零刚度不是实际执行器增益。
    physics = npz(OUTPUT / 'turn_parent_native_8s/physics_initial.npz')
    for name in ('turn_candidate_ramp', 'turn_candidate_no_delay'):
        other = npz(OUTPUT / name / 'physics_initial.npz')
        assert physics.keys() == other.keys() and all(np.array_equal(physics[k], other[k]) for k in physics)
    runs, raw_files = [], set()
    for directory in sorted(OUTPUT.iterdir()):
        if not (directory / 'execution.json').is_file():
            continue
        execution = read(directory / 'execution.json')
        assert execution['status'] in ('completed', 'failed')
        entry = dict(name=directory.name, execution=execution)
        if execution['status'] == 'completed':
            meta = read(directory / 'diagnostic_report.json')
            assert meta['status'] == 'passed' and meta['optimizer_updates'] == 0
            assert execution['exit_code'] == 0 and execution['physical_gpu'] == 1
            entry['diagnostic'] = meta
        else:
            assert directory.name == 'turn_candidate_native' and not (directory / 'report.json').exists()
            assert 'unsupported operand type' in read(directory / 'error.json')['error']
            entry['explanation'] = '开发期记录器用函数替换环境类，类型注解导入失败；改为子类，未进入仿真控制。'
        runs.append(entry)
        target = OUT / 'runs' / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ('execution.json', 'report.json', 'diagnostic_report.json', 'error.json', 'names.json'):
            if (directory / name).is_file():
                shutil.copy2(directory / name, target / name)
        raw_files.update(p for p in directory.iterdir() if p.is_file())
    assert len(runs) == 9 and sum('diagnostic' in r for r in runs) == 8
    successful = [r for r in runs if 'diagnostic' in r]
    interactions = sum(r['diagnostic']['episodes'] * r['diagnostic']['steps'] for r in successful)
    assert interactions == 864000
    raw_files.update((PARENT, CANDIDATE))
    for root in (BASELINE, NEW_EVALUATION):
        for name in ('up_nominal', 'turn_nominal', 'turn_training_randomized'):
            raw_files.update(p for p in (root / name).iterdir() if p.suffix in ('.json', '.npz'))
    raw_files.update(Path(__file__).parent / name for name in (
        'reference_low_stairs_regression_probe.py', 'reference_low_stairs_regression_analyze.py',
        'reference_low_stairs_preservation.py', 'reference_low_stairs_preservation_validate.py',
        'reference_low_stairs_regression_archive.py'))
    save(OUT / 'raw_manifest.json', [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(raw_files)])
    save(OUT / 'verification.json', dict(status='passed', tests=85, native_replays=checks,
        historical_training_sources_intact=True, checkpoint_hashes=HASHES,
        optimizer_updates=0, successful_simulations=8, startup_failures=1,
        probe_episode_slots=sum(r['diagnostic']['episodes'] for r in successful),
        measured_interactions=interactions, preparation_interactions=120000,
        simulation_elapsed_s=sum(r['execution']['elapsed_s'] for r in successful),
        peak_gpu1_total_used_mib=max(r['execution']['peak_total_gpu_used_mib'] for r in successful),
        all_runs_exited=True, original_acceptance='failed, unchanged',
        note='诊断中单因素对照不是新的正式验收；随机化回合重置后噪声流可能分叉。', runs=runs))
    print('REGRESSION PREPARATION VERIFIED: 85 tests, 8 frozen simulations, 0 updates')


if __name__ == '__main__':
    main()
