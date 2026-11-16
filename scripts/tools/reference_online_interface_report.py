"""Archive the failed interface attempt and reviewed offline repair; never simulate."""
import hashlib
import os
import subprocess

from reference_online_targets import ROOT, OUTPUT, DOCS, read, provenance, HASHES, MODEL_PATHS
from reference_stairs_eval import save, sha


def main():
    failed_commit = '30b22c9'
    directory = OUTPUT / 'smoke_B'
    execution = read(directory / 'execution.json')
    error = read(directory / 'error.json')
    assert execution['status'] == 'failed' and execution['exit_code'] == 1
    assert "compute_command() got an unexpected keyword argument 'dt'" in error['error']
    assert not (directory / 'initial.npz').exists() and not (directory / 'report.json').exists()
    try:
        os.kill(execution['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('failed worker PID is still live; inspect before finalizing')
    locked = read(OUTPUT / 'lock.json')
    assert locked == read(DOCS / 'LOCK.json')
    repaired = []
    for row in locked['sources']:
        from pathlib import Path
        relative = str(Path(row['path']).relative_to(ROOT))
        original = subprocess.check_output(['git', 'show', f'{failed_commit}:{relative}'], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == row['sha256'], relative
        current = sha(row['path'])
        if current != row['sha256']:
            repaired.append(dict(path=relative, before_sha256=row['sha256'], after_sha256=current))
    assert {r['path'] for r in repaired} == {
        'scripts/tools/reference_online_collect.py', 'scripts/tools/reference_online_interface.py'}
    assert {k: sha(v) for k, v in MODEL_PATHS.items()} == HASHES
    # Repeat only read-only lineage validation; this creates no simulation or training.
    audit = provenance()
    assert audit == locked['provenance']
    attempts = read(OUTPUT / 'attempts.json')
    assert len(attempts) == 1 and attempts[0]['kind'] == 'smoke'
    evidence = dict(status='stopped_at_interface_gate', failed_source_commit=failed_commit,
        runtime=execution, error=error, geometry=read(directory / 'geometry.json'),
        provenance_reverified=True, actor_checkpoint_hashes=HASHES,
        preparation_control_steps_consumed=1, actor_calls=0, formal_episodes=0,
        optimizer_updates=0, smoke_runs_consumed=1, smoke_runs_remaining_max=3,
        remaining_formal_budgets=dict(baseline=384, grid=1200, confirmation=64, total=1648),
        repaired_sources=repaired, offline_tests_after_repair=23,
        repair_simulation_verified=False, current_attempt_lock_intentionally_unchanged=True,
        next_action='User approval and a fresh source registration are required before a repaired smoke run.',
        raw_files=[dict(path=str(p), sha256=sha(p)) for p in
            [directory / 'execution.json', directory / 'error.json', directory / 'runtime.log',
             directory / 'geometry.json', directory / 'config.json', OUTPUT / 'attempts.json']])
    save(DOCS / 'EVIDENCE.json', evidence)
    save(OUTPUT / 'STOPPED.json', evidence)


if __name__ == '__main__':
    main()
