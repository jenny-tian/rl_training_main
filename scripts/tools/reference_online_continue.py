"""Approved continuation after the dt callback repair, with inherited budgets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import reference_online_collect as collector
from reference_online_targets import (ROOT, OUTPUT as OLD_OUTPUT, DOCS as OLD_DOCS,
    protocol, provenance, rows, read, MODEL_PATHS, HASHES)
from reference_stairs_eval import save, sha

OUTPUT = ROOT / 'logs/reference_six/online_targets_repaired_20261116'
DOCS = ROOT / 'docs/lite3-online-targets-repaired-20261116'


def registration():
    old = read(OLD_OUTPUT / 'lock.json')
    reviewed = read(OLD_DOCS / 'EVIDENCE.json')
    assert old == read(OLD_DOCS / 'LOCK.json') and old['protocol'] == protocol()
    approved_changes = {str(ROOT / r['path']): r for r in reviewed['repaired_sources']}
    sources = []
    for row in old['sources']:
        digest = sha(row['path'])
        if row['path'] in approved_changes:
            change = approved_changes[row['path']]
            assert row['sha256'] == change['before_sha256'] and digest == change['after_sha256']
        else:
            assert digest == row['sha256'], row['path']
        sources.append(dict(path=row['path'], sha256=digest))
    sources.append(dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    value = dict(protocol=old['protocol'], provenance=old['provenance'], sources=sources,
        continuation=dict(approved=True, approval='User: 同意, after repair proposal',
            prior_lock_sha256=sha(OLD_OUTPUT / 'lock.json'),
            repair_evidence_sha256=sha(OLD_DOCS / 'EVIDENCE.json'),
            proposal_sha256=sha(OLD_DOCS / 'NEXT_STAGE_PROPOSAL.md'),
            prior_failed_attempt=str(OLD_OUTPUT / 'smoke_B'), inherited_smoke_attempts=1,
            protocol_changed=False, callback_repair_only=True))
    path = OUTPUT / 'lock.json'
    if path.exists():
        assert read(path) == value and read(DOCS / 'LOCK.json') == value
    else:
        assert provenance() == old['provenance']
        previous = read(OLD_OUTPUT / 'attempts.json')
        assert len(previous) == 1 and previous[0]['kind'] == 'smoke'
        save(path, value)
        save(DOCS / 'LOCK.json', value)
        save(OUTPUT / 'attempts.json', [dict(previous[0], inherited=True)])
    assert {name: sha(path) for name, path in MODEL_PATHS.items()} == HASHES
    return value


def validate_budget(attempts, kind, count, output):
    if any(a['output'] == str(output) for a in attempts):
        raise RuntimeError('attempt already reserved; no automatic retry')
    if kind == 'smoke':
        if sum(a['kind'] == 'smoke' for a in attempts) >= 4:
            raise RuntimeError('all four smoke attempts have been used, including prior failure')
    else:
        used = sum(a['episodes'] for a in attempts if a['kind'] == kind)
        if used + count > protocol()['budgets'][kind]:
            raise RuntimeError(f'{kind} episode budget exceeded')


def completed(directory):
    execution, report = read(directory / 'execution.json'), read(directory / 'report.json')
    assert execution['status'] == 'completed' and execution['exit_code'] == 0
    assert report['status'] == 'passed' and report['actor_unchanged']
    assert report['optimizer_updates'] == 0 and report['checkpoint_sha256'] == HASHES[report['model']]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--kind', choices=('smoke', 'baseline', 'grid', 'confirmation'))
    parser.add_argument('--model', choices=tuple(MODEL_PATHS))
    parser.add_argument('--condition', choices=protocol()['conditions'])
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.register_only:
        registration()
        return 0
    if args.kind is None or args.model is None:
        parser.error('kind and model are required')
    if args.kind in ('smoke', 'baseline') and args.condition is not None:
        parser.error('smoke/baseline include all six locked conditions')
    if args.kind in ('grid', 'confirmation') and (args.model != 'B' or args.condition is None):
        parser.error('residual evaluation requires B and a selected target')
    name = f'{args.kind}_{args.model}' + (f'_{args.condition}' if args.condition else '')
    args.output = OUTPUT / name
    args.winner = OUTPUT / f'winner_{args.condition}.json' if args.kind == 'confirmation' else None
    if not args.child:
        registration()
        attempts = read(OUTPUT / 'attempts.json')
        # The acknowledged previous-stage failure is retained in the budget.
        # Any new failure stops this continuation before another worker starts.
        for attempt in attempts:
            if not attempt.get('inherited'):
                completed(Path(attempt['output']))
        if args.kind != 'smoke':
            for model in ('B', 'zero10100'):
                completed(OUTPUT / f'smoke_{model}')
        if args.kind in ('grid', 'confirmation'):
            assert args.condition in read(OUTPUT / 'selection.json')['selected']
        if args.kind == 'confirmation':
            completed(OUTPUT / f'grid_B_{args.condition}')
        specs = rows(args.kind, args.condition, read(args.winner) if args.winner else None)
        validate_budget(attempts, args.kind, len(specs), args.output)
        attempts.append(dict(kind=args.kind, model=args.model, condition=args.condition,
            output=str(args.output), episodes=len(specs), inherited=False,
            max_control_steps=600 if args.kind == 'smoke' else 1050))
        save(OUTPUT / 'attempts.json', attempts)
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
            gpu=1, output=args.output, timeout=2400, minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        collector.OUTPUT = OUTPUT  # Only the new registration/output directory changes.
        collector.collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc()
        save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
