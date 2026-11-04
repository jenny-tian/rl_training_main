"""One CPU-only independent review of the approved real 1884-control evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
from rl_training.source_scan.delivery_stage import reference,read_reference,file_digest
from rl_training.source_scan.delivery_revision import (validate_revision,compatible_inputs,
    source_artifacts,LIMITED_SCHEMA)
from rl_training.source_scan.smoke_artifacts import atomic_json


def inspect_inputs(revision_ref, checkpoint):
    import torch
    from source_training_entrypoint import _git_state,_input_hashes
    from source_scale_startup_supervisor import dependencies
    from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path
    from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
    approval,source=validate_revision(revision_ref)
    revision,remote,dirty=_git_state()
    if dirty or revision!=remote or torch.cuda.is_initialized():
        raise ValueError('CPU review needs clean remote-matching code and no initialized CUDA')
    inputs=_input_hashes();compatible_inputs(source,inputs);source_artifacts(approval)
    if (dependencies()!=source['dependency_sha256'] or file_digest(checkpoint)!=CHECKPOINT_SHA256
            or BudgetLedger(canonical_budget_path(ROOT)).reference()!=source['budget_ledger']):
        raise ValueError('limited review original model/dependency/budget changed')
    return dict(revision=revision,input_sha256=inputs,dependency_sha256=dependencies(),
        source_manifest=approval['source_manifest'],stage_revision=revision_ref)


def supervise(args):
    revision_ref=reference(args.approval,'stage_revision')
    frozen=inspect_inputs(revision_ref,args.checkpoint)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'review_manifest.json',frozen)
    command=['timeout','--signal=TERM','--kill-after=15s','1800s',sys.executable,
        str(Path(__file__).resolve()),'probe','--approval',str(args.approval.resolve()),
        '--checkpoint',str(args.checkpoint.resolve()),'--output-dir',str(out)]
    started=time.monotonic();process=None
    execution=dict(command=command,timed_out=False,returncode=None,process_group_empty=False)
    from source_live_ppo_supervisor import group_exists,terminate_group
    try:
        with (out/'runtime.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'',
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            execution['process_group']=process.pid;atomic_json(out/'execution.json',execution)
            execution['returncode']=process.wait(timeout=1820)
    except BaseException as error:
        execution['error']=type(error).__name__+': '+str(error)
    finally:
        if process is not None:
            if group_exists(process.pid):terminate_group(process,15)
            execution['process_group_empty']=not group_exists(process.pid)
            execution['timed_out']=process.returncode in (124,137,-15,-9) or execution['returncode'] is None
        execution['elapsed_s']=time.monotonic()-started
        atomic_json(out/'execution.json',execution)
    good=execution['returncode']==0 and not execution['timed_out'] and execution['process_group_empty']
    report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {}
    receipt=dict(schema=LIMITED_SCHEMA,stage_revision=revision_ref,
        scope_qualified=good and report.get('scope_qualified') is True,
        report=reference(out/'report.json','limited_motion_report') if report else None,
        execution=reference(out/'execution.json','limited_review_execution'))
    atomic_json(out/'receipt.json',receipt)
    print(json.dumps(dict(scope_qualified=receipt['scope_qualified'],execution=execution),indent=2))
    return 0 if receipt['scope_qualified'] else 1


def probe(args):
    from source_scale_motion_validation import validate
    revision_ref=reference(args.approval,'stage_revision');frozen=inspect_inputs(revision_ref,args.checkpoint)
    out=args.output_dir.resolve()
    if frozen!=json.loads((out/'review_manifest.json').read_text()) or (out/'report.json').exists():
        raise ValueError('CPU review ticket/source/restart differs')
    approval,source=validate_revision(revision_ref);folder=Path(source['output_dir'])
    result=validate(folder,source,json.loads((folder/'execution.json').read_text()),args.checkpoint,
                    limited_revision=revision_ref)
    result['review_revision']=frozen['revision']
    if inspect_inputs(revision_ref,args.checkpoint)!=frozen:raise ValueError('CPU review inputs changed')
    atomic_json(out/'report.json',result)
    print(json.dumps(result,indent=2))
    return 0 if result.get('scope_qualified') is True else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('supervise','probe'))
    for name in ('approval','checkpoint','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();raise SystemExit(supervise(args) if args.action=='supervise' else probe(args))
