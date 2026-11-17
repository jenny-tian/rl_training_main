"""Prepared continuation after indexed-delay failure; requires next phase approval."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import reference_online_c1b as original
from reference_stairs_eval import ROOT, save, sha
from reference_online_targets import read

PRIOR = original.OUTPUT
OUTPUT = ROOT / 'logs/reference_six/online_c1b_repaired_20261117'
DOCS = ROOT / 'docs/lite3-online-c1b-repaired-20261117'


def registration():
    original.registration()
    error = read(PRIOR / 'smoke_1/error.json')['error']
    assert 'got Int for the destination and Long for the source' in error
    attempts = read(PRIOR / 'attempts.json')
    assert len(attempts) == 1 and attempts[0]['name'] == 'smoke_1'
    paths = [Path(__file__),Path(__file__).with_name('reference_online_c1b_env_repaired.py'),
             ROOT / 'source/rl_training/rl_training/ea_lpl/c1b_delay.py']
    value = dict(protocol=original.protocol(), prior_lock_sha256=sha(PRIOR / 'lock.json'),
        proposal_sha256=sha(ROOT / 'docs/lite3-online-c1b-20261117/NEXT_STAGE_PROPOSAL.md'),
        evidence_sha256=sha(ROOT / 'docs/lite3-online-c1b-20261117/EVIDENCE.json'),
        inherited_failed_smoke='smoke_1', remaining_smoke='smoke_2',
        change='cast indexed lag writes to native buffer storage dtype',
        sources=[dict(path=str(p),sha256=sha(p)) for p in paths])
    if (OUTPUT / 'lock.json').exists():
        assert value == read(OUTPUT / 'lock.json') == read(DOCS / 'LOCK.json')
    else:
        save(OUTPUT / 'lock.json',value); save(DOCS / 'LOCK.json',value)
        save(OUTPUT / 'attempts.json',[dict(attempts[0],inherited=True)])


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
    name = 'smoke_2' if args.kind == 'smoke' else args.kind
    args.output = OUTPUT / name
    if not args.child:
        registration()
        attempts = read(OUTPUT / 'attempts.json')
        if any(a['name'] == name for a in attempts): raise RuntimeError('attempt already reserved; no retry')
        for a in attempts:
            if not a.get('inherited'): original.completed(OUTPUT / a['name'])
        if args.kind != 'smoke': original.completed(OUTPUT / 'smoke_2')
        if args.kind in ('target','retention'): original.completed(OUTPUT / 'train')
        n = len(original.rows(args.kind,2))
        steps = 600 if args.kind == 'smoke' else 20000 if args.kind == 'train' else 1050
        reserve = n*steps
        assert sum(a['reserved_steps'] for a in attempts if a['kind']==args.kind)+reserve <= original.protocol()['budgets'][args.kind]
        attempts.append(dict(name=name,kind=args.kind,reserved_steps=reserve,environments=n,inherited=False))
        save(OUTPUT / 'attempts.json',attempts)
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=1,output=args.output,timeout=3600,minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        # Retain the locked collector and protocol. Only its environment import
        # and its new output/checkpoint location are explicitly redirected.
        original.OUTPUT = OUTPUT
        import reference_online_c1b_env_repaired as repaired
        sys.modules['reference_online_c1b_env'] = repaired
        from reference_online_c1b_collect import collect
        collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception as error:
        original_error = traceback.format_exc(); traceback.print_exc()
        try:
            from reference_online_failure_capture import capture_failure
            capture_failure(error,args.output)
        except Exception:
            save(args.output / 'failure_capture_error.json',dict(error=traceback.format_exc()))
        save(args.output / 'error.json',dict(error=original_error))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
