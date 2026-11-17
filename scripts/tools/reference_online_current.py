"""Approved current-memory integration; inherit both failed smoke attempts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from reference_stairs_eval import ROOT, save, sha
from reference_online_targets import MODEL_PATHS, HASHES, protocol, provenance, rows, read
from reference_online_continue import OUTPUT as PRIOR_OUTPUT, DOCS as PRIOR_DOCS, validate_budget
from reference_online_failure_capture import capture_failure

OUTPUT = ROOT / 'logs/reference_six/online_targets_current_20261117'
DOCS = ROOT / 'docs/lite3-online-targets-current-20261117'


def registration():
    prior = read(PRIOR_OUTPUT / 'lock.json')
    assert prior == read(PRIOR_DOCS / 'LOCK.json') and prior['protocol'] == protocol()
    for source in prior['sources']:
        assert sha(source['path']) == source['sha256'], source['path']
    paths = [Path(__file__).resolve(), Path(__file__).with_name('reference_online_current_collect.py'),
        Path(__file__).with_name('reference_online_failure_capture.py'),
        ROOT / 'source/rl_training/rl_training/ea_lpl/current_safety.py',
        ROOT / 'source/rl_training/rl_training/ea_lpl/current_shadow.py']
    new = dict(protocol=prior['protocol'], provenance=prior['provenance'],
        sources=prior['sources'] + [dict(path=str(p), sha256=sha(p)) for p in paths],
        continuation=dict(approved=True, approval='User: 同意, after current-memory proposal',
            prior_lock_sha256=sha(PRIOR_OUTPUT / 'lock.json'),
            proposal_sha256=sha(PRIOR_DOCS / 'NEXT_STAGE_PROPOSAL.md'),
            evidence_sha256=sha(PRIOR_DOCS / 'EVIDENCE.json'), inherited_failed_smoke_attempts=2,
            safety_memory='projected_action_minus_projected_base_v1',
            changed=['current model safety memory', 'pre-decision snapshots and component diagnostics',
                     'exception capture', 'registration/output directory'],
            geometry_commands_seeds_and_outcomes_unchanged=True))
    target = OUTPUT / 'lock.json'
    if target.exists():
        assert read(target) == new and read(DOCS / 'LOCK.json') == new
    else:
        assert provenance() == prior['provenance']
        previous = read(PRIOR_OUTPUT / 'attempts.json')
        assert len(previous) == 2 and all(a['kind'] == 'smoke' for a in previous)
        save(target, new)
        save(DOCS / 'LOCK.json', new)
        save(OUTPUT / 'attempts.json', [dict(a, inherited=True) for a in previous])
    assert {k: sha(v) for k, v in MODEL_PATHS.items()} == HASHES
    return new


def completed(directory):
    run, result = read(directory / 'execution.json'), read(directory / 'report.json')
    assert run['status'] == 'completed' and run['exit_code'] == 0
    assert result['status'] == 'passed' and result['actor_unchanged'] and result['optimizer_updates'] == 0
    assert result['checkpoint_sha256'] == HASHES[result['model']]
    assert result['interface']['safety_memory'] == 'projected_action_minus_projected_base_v1'
    return result


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
        parser.error('kind and model required')
    if args.kind in ('smoke', 'baseline') and args.condition is not None:
        parser.error('all six conditions must be evaluated')
    if args.kind in ('grid', 'confirmation') and (args.model != 'B' or args.condition is None):
        parser.error('residual evaluations require B and a selected target')
    name = f'{args.kind}_{args.model}' + (f'_{args.condition}' if args.condition else '')
    args.output = OUTPUT / name
    args.winner = OUTPUT / f'winner_{args.condition}.json' if args.kind == 'confirmation' else None
    if not args.child:
        registration()
        attempts = read(OUTPUT / 'attempts.json')
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
        from reference_online_current_collect import collect
        collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception as error:
        original = traceback.format_exc()
        traceback.print_exc()
        try:
            capture_failure(error, args.output)
        except Exception:
            save(args.output / 'failure_capture_error.json', dict(error=traceback.format_exc()))
        save(args.output / 'error.json', dict(error=original))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
