"""Prepared reset-API continuation; one new smoke slot requires user approval."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import reference_online_c1b as original
import reference_online_c1b_continue as prior
from reference_stairs_eval import ROOT, save, sha
from reference_online_targets import read

OUTPUT = ROOT/'logs/reference_six/online_c1b_reset_api_20261117'
DOCS = ROOT/'docs/lite3-online-c1b-reset-api-20261117'


def registration():
    prior.registration()
    attempts = read(prior.OUTPUT/'attempts.json')
    assert [a['name'] for a in attempts] == ['smoke_1','smoke_2']
    assert 'reset_buf' in read(prior.OUTPUT/'smoke_2/error.json')['error']
    proposal = prior.DOCS/'NEXT_STAGE_PROPOSAL.md'
    evidence = prior.DOCS/'EVIDENCE.json'
    paths = [Path(__file__),Path(__file__).with_name('reference_online_c1b_env_reset_api.py'),
             Path(__file__).with_name('reference_online_c1b_collect_reset_api.py')]
    value = dict(protocol=original.protocol(),prior_lock_sha256=sha(prior.OUTPUT/'lock.json'),
        proposal_sha256=sha(proposal),evidence_sha256=sha(evidence),
        inherited_failed_smokes=['smoke_1','smoke_2'],new_smoke='smoke_3',
        smoke_profile=2,reset_keys='repeat smoke_2 keys; no choice from outcome',
        revised_budgets=dict(smoke=28800,train=320000,target=268800,retention=336000,total=953600),
        changes=['native ownership of step-created termination outputs','convert NumPy log scalars to Python'],
        sources=[dict(path=str(p),sha256=sha(p)) for p in paths])
    if (OUTPUT/'lock.json').exists():
        assert value == read(OUTPUT/'lock.json') == read(DOCS/'LOCK.json')
    else:
        save(OUTPUT/'lock.json',value); save(DOCS/'LOCK.json',value)
        save(OUTPUT/'attempts.json',[dict(a,inherited=True) for a in attempts])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only',action='store_true')
    parser.add_argument('--kind',choices=('smoke','train','target','retention'))
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    if args.register_only:
        registration(); return 0
    if args.kind is None: parser.error('kind required')
    args.smoke_index = 2
    name = 'smoke_3' if args.kind == 'smoke' else args.kind
    args.output = OUTPUT/name
    if not args.child:
        lock = registration()
        attempts = read(OUTPUT/'attempts.json')
        if any(a['name'] == name for a in attempts): raise RuntimeError('attempt already reserved; no retry')
        for a in attempts:
            if not a['inherited']: original.completed(OUTPUT/a['name'])
        if args.kind != 'smoke': original.completed(OUTPUT/'smoke_3')
        if args.kind in ('target','retention'): original.completed(OUTPUT/'train')
        if args.kind == 'retention': original.completed(OUTPUT/'target')
        n = len(original.rows(args.kind,2))
        steps = 600 if args.kind == 'smoke' else 20000 if args.kind == 'train' else 1050
        reserve = n*steps
        assert sum(a['reserved_steps'] for a in attempts if a['kind']==args.kind)+reserve <= lock['revised_budgets'][args.kind]
        assert sum(a['reserved_steps'] for a in attempts)+reserve <= lock['revised_budgets']['total']
        attempts.append(dict(name=name,kind=args.kind,reserved_steps=reserve,environments=n,inherited=False))
        save(OUTPUT/'attempts.json',attempts)
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=1,output=args.output,timeout=3600,minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        original.OUTPUT = OUTPUT
        import reference_online_c1b_env_reset_api as repaired
        sys.modules['reference_online_c1b_env'] = repaired
        from reference_online_c1b_collect_reset_api import collect
        collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception as error:
        original_error = traceback.format_exc(); traceback.print_exc()
        try:
            from reference_online_failure_capture import capture_failure
            capture_failure(error,args.output)
        except Exception:
            save(args.output/'failure_capture_error.json',dict(error=traceback.format_exc()))
        save(args.output/'error.json',dict(error=original_error))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__': sys.exit(main())
