"""Seal text/JSON evidence for completed current-model runs; no new simulation."""
from __future__ import annotations

import os
from pathlib import Path

from reference_online_current import OUTPUT, DOCS, completed, registration
from reference_online_targets import MODEL_PATHS, HASHES, protocol, read
from reference_stairs_eval import ROOT, save, sha


def main():
    registration()
    attempts = read(OUTPUT / 'attempts.json')
    runs, raw = [], []
    for attempt in attempts:
        if attempt.get('inherited'):
            continue
        directory = Path(attempt['output'])
        report = completed(directory)
        execution = read(directory / 'execution.json')
        try:
            os.kill(execution['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'worker PID still exists: {execution["pid"]}')
        runs.append(dict(attempt=attempt, execution=execution,
                         report={k: v for k, v in report.items() if k != 'records'}))
        for name in ('execution.json', 'runtime.log', 'report.json', 'config.json', 'geometry.json', 'initial.npz', 'trace.npz'):
            path = directory / name
            raw.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    formal = {k: sum(a['episodes'] for a in attempts if a['kind'] == k) for k in ('baseline', 'grid', 'confirmation')}
    assert all(v <= protocol()['budgets'][k] for k, v in formal.items())
    assert sum(formal.values()) <= protocol()['budgets']['total']
    assert {k: sha(v) for k, v in MODEL_PATHS.items()} == HASHES
    sources = [Path(__file__), Path(__file__).with_name('reference_online_current_analysis.py'),
               Path(__file__).with_name('reference_online_current_replay.py')]
    confirmation = {p.stem: read(p) for p in sorted(DOCS.glob('CONFIRMATION_*.json'))}
    confirmed = [r['condition'] for r in confirmation.values() if r['confirmed']]
    analyses = [DOCS / 'SELECTION.json', *sorted(DOCS.glob('GRID_*.json')),
                *sorted(DOCS.glob('CONFIRMATION_*.json')), *sorted(DOCS.glob('REPLAY_*.json'))]
    value = dict(status='completed_bounded_frozen_validation', optimizer_updates=0, checkpoint_hashes=HASHES,
        smoke_attempts_total=sum(a['kind'] == 'smoke' for a in attempts), inherited_failures=2,
        formal_episodes=formal, formal_total=sum(formal.values()), formal_ceiling=protocol()['budgets']['total'],
        selected=read(OUTPUT / 'selection.json')['selected'], attempts=attempts, runs=runs, raw_sources=raw,
        confirmed_targets=confirmed, stopped_on='no confirmed headroom' if not confirmed else None,
        analysis_outputs=[dict(path=str(p), sha256=sha(p)) for p in analyses],
        analysis_sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
        lock_sha256=sha(OUTPUT / 'lock.json'),
        scope='Frozen actor, fixed residual codes, offline current-filter replay. No learning updates.',
        media='No simulation media generated or uploaded in this phase. Local raw arrays/checkpoints remain under logs.')
    save(DOCS / 'EVIDENCE.json', value)
    print({k: value[k] for k in ('formal_episodes', 'formal_total', 'smoke_attempts_total', 'selected')})


if __name__ == '__main__':
    main()
