"""顺序完成批准的 720 回合；已存在但不完整的运行不自动重启。"""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_heading_probe import (OUTPUT, CHECKPOINT, DIGEST, MODEL, CASES,
    PROTOCOL, ORIGINAL, PROTOCOL_SHA, ORIGINAL_SHA)
from reference_low_stairs_zero_preservation_evaluate import checked_training, verify_run, EVALUATION, read
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration
from reference_low_stairs_zero_preservation_analyze import paired_records
from reference_low_stairs_equal100_analyze import heading_error, validate_trace
from reference_low_stairs_analyze import check_log

PROBE = Path(__file__).with_name('reference_low_stairs_heading_probe.py')
HISTORICAL = ROOT/'docs/lite3-low-stairs-zero-preservation-20261115/raw_manifest.json'
WATCH = ROOT/'scripts/tools/reference_six_watch.py'
WATCH_SHA = '21060b75a6b34cb49c29cb868b5fb18cede274c4481e3e13616e72b6ddb049ad'


def configuration_equal(left, right):
    """仅允许来源清单中只读日志观察器的固定已知版本有无差异。"""
    configs = [copy.deepcopy(c) for c in (left, right)]
    for cfg in configs:
        sources = cfg['sources']
        for source in sources:
            assert sha(source['path']) == source['sha256']
            if source['path'] == str(WATCH):
                assert source['sha256'] == WATCH_SHA
        cfg['sources'] = [s for s in sources if s['path'] != str(WATCH)]
    return same_configuration(*configs)


def historical_check(paths):
    historic = {r['path']: r['sha256'] for r in read(HISTORICAL)}
    files = []
    for path in paths:
        for file in sorted(path.iterdir()):
            if file.is_file():
                digest = sha(file)
                assert historic[str(file)] == digest, str(file)
                files.append(dict(path=str(file), bytes=file.stat().st_size, sha256=digest))
    return files


def verify_pair(path, profile, direction, *, reused=False):
    report = verify_run(path, 'stairs', profile, DIGEST, MODEL, direction, heading=reused)
    before = EVALUATION/f'{direction}_{profile}'
    baseline = verify_run(before, 'stairs', profile, DIGEST, MODEL, direction)
    paired_records(baseline, report, 'stairs')
    assert same_initial(path, before), f'initial state differs: {path.name}'
    assert configuration_equal(read(path/'config.json'), read(before/'config.json'))
    if not reused:
        meta = read(path/'heading_intervention.json')
        assert meta['source_sha256'] == sha(PROBE) and meta['checkpoint_sha256'] == DIGEST
        assert meta['case'] == path.name and CASES[path.name] == (profile, direction)
        assert meta['original_protocol_sha256'] == PROTOCOL_SHA and meta['original_probe_sha256'] == ORIGINAL_SHA
        assert meta['gain'] == .5 and meta['limit_rad_s'] == .8 and meta['target'] == 'world +x'
        assert meta['optimizer_updates'] == 0 and meta['diagnostic_only'] and not meta['lateral_position_feedback']
        assert meta['initial_preparation_steps'] == 50 and meta['rollout_pause_s'] == meta['rollout_ramp_s'] == .5
    with np.load(path/'trace.npz') as archive:
        data = {k: archive[k] for k in archive.files}
    validate_trace(report, data)
    error = heading_error(path, data)
    # Forward command timing must remain the original per-start ramp.
    t = np.arange(report['steps'])*.02
    speeds = np.array([r['speed'] for r in sorted(report['records'], key=lambda r:r['env_id'])])
    expected = np.clip((t-.5)/.5, 0., 1.)[:, None]*speeds
    assert np.max(np.abs(data['command'][..., 0]-expected)[data['active']]) < 1e-6
    assert not data['command'][..., 1][data['active']].any()
    assert sha(CHECKPOINT) == DIGEST and sha(PROTOCOL) == PROTOCOL_SHA and sha(ORIGINAL) == ORIGINAL_SHA
    return dict(name=path.name, candidate=str(path), baseline=str(before), profile=profile,
        direction=direction, episodes=report['episodes'], reused=reused, initial_equal=True,
        configuration_equal=True, feedback_max_error=error, log_check=check_log(path))


def main():
    checkpoint, digest = checked_training()
    assert checkpoint == CHECKPOINT and digest == DIGEST
    assert sha(PROTOCOL) == PROTOCOL_SHA and sha(ORIGINAL) == ORIGINAL_SHA
    historical = [EVALUATION/f'{d}_{p}' for p in ('nominal', 'training_randomized') for d in ('up', 'down')]
    historical.append(EVALUATION/'heading_up_nominal')
    files = historical_check(historical)
    reused = verify_pair(historical[-1], 'nominal', 'up', reused=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT/'plan.json', dict(new_episodes=720, reused_heading_episodes=240,
        reused_no_feedback_episodes=960, optimizer_updates=0, physical_gpu=1,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=DIGEST, cases=CASES,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        probe_sha256=sha(PROBE), evaluator_sha256=sha(__file__), automatic_extension=False))
    save(OUTPUT/'historical_manifest.json', files)
    matrix = [reused]
    for name, (profile, direction) in CASES.items():
        path = OUTPUT/name
        if not path.exists():
            print('EVALUATE', name, flush=True)
            subprocess.run([sys.executable, str(PROBE), '--case', name], cwd=ROOT, check=True)
        matrix.append(verify_pair(path, profile, direction))
        save(OUTPUT/'matrix.json', matrix)
        print('VERIFIED', name, flush=True)
    assert sum(r['episodes'] for r in matrix if not r['reused']) == 720
    assert historical_check(historical) == files
    save(OUTPUT/'complete.json', dict(status='passed', status_scope='运行和配对核验通过，行为结果另行分析',
        new_episodes=720, reused_heading_episodes=240, optimizer_updates=0, checkpoint_sha256=DIGEST,
        matrix=matrix, automatic_extension=False))
    print('HEADING MATRIX COMPLETE: 720 new frozen episodes; 240 reused', flush=True)


if __name__ == '__main__':
    main()
