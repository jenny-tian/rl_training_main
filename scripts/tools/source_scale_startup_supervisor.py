"""Freeze or supervise one 16-environment G1 attempt after separate launch approval."""
import argparse
import hashlib
import json
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
from rl_training.source_scan.scale_startup import CONTRACT,proposal,validate_plan
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path
from rl_training.source_scan.smoke_artifacts import atomic_json
from source_training_entrypoint import _git_state,_input_hashes
from source_live_ppo_supervisor import dependencies as reused_dependencies,group_exists,terminate_group
from source_scale_startup_validation import validate


def dependencies():
    import numpy as np
    from numpy.core import _methods,_multiarray_umath
    if np.__version__!='1.26.0':raise RuntimeError('scale adapters are audited for NumPy 1.26.0')
    hashes=reused_dependencies()
    for obj in (np,inspect.unwrap(np.unique),inspect.unwrap(np.allclose),inspect.unwrap(np.linalg.norm),
                inspect.unwrap(np.any),_methods,_multiarray_umath):
        path=Path(inspect.getfile(obj));hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def preflight(args,*,child=False):
    plan=json.loads(args.manifest.read_text())
    if plan.get('user_approved') is not True or plan.get('status')!='approved':
        raise PermissionError('separate approval of the single 16-environment G1 launch is required')
    revision,remote,dirty=_git_state()
    if dirty or revision!=remote:raise ValueError('G1 requires clean committed and remote-matching source')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0'):raise ValueError('G1 GPU0 visibility mapping differs')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:raise ValueError('registered source checkpoint changed')
    output=Path(plan['output_dir']).resolve()
    if child:
        if args.manifest.resolve()!=output/'run_manifest.json' or not (output/'supervision.json').is_file():
            raise ValueError('G1 child requires parent-owned output and supervision ticket')
        ticket=json.loads((output/'supervision.json').read_text())
        if ticket!={'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'revision':revision}:
            raise ValueError('G1 supervision ticket changed')
        # validate_plan checks a fresh output; the child must use the parent-created directory.
        if (output/'result.json').exists():raise FileExistsError('G1 child cannot restart a result')
    elif args.output_dir.resolve()!=output:raise ValueError('G1 output differs from manifest')
    reference=BudgetLedger(canonical_budget_path(ROOT)).reference()
    if child:
        expected=proposal(revision=revision,remote=remote,inputs=_input_hashes(),dependencies=dependencies(),budget_reference=reference,output_dir=output)
        for key,value in expected.items():
            if key not in ('status','user_approved') and (type(plan.get(key)) is not type(value) or plan[key]!=value):raise ValueError('G1 child input mismatch: '+key)
    else:
        validate_plan(plan,revision=revision,inputs=_input_hashes(),dependencies=dependencies(),budget_reference=reference,output_dir=output)
    return plan


def prepare(args):
    if args.manifest.exists() or args.output_dir.exists():raise FileExistsError('G1 proposal/attempt already exists')
    revision,remote,dirty=_git_state()
    if dirty or revision!=remote:raise ValueError('freeze G1 only after commit, push and remote verification')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:raise ValueError('original checkpoint changed')
    plan=proposal(revision=revision,remote=remote,inputs=_input_hashes(),dependencies=dependencies(),
        budget_reference=BudgetLedger(canonical_budget_path(ROOT)).reference(),output_dir=args.output_dir)
    validate_plan(plan,revision=revision,inputs=plan['input_sha256'],dependencies=plan['dependency_sha256'],
        budget_reference=plan['budget_ledger'],output_dir=args.output_dir,approved=False)
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    with args.manifest.open('x') as stream:json.dump(plan,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='frozen_not_approved',manifest=str(args.manifest),base_revision=revision,
        real_source_ppo_updates=0,isaac_sim_launches=0,gpu_jobs=0),indent=2))
    return 0


def deadline_expired(elapsed,text):
    return elapsed>240 or elapsed>120 and '[scale-g1] initial-ready' not in text


def run(args):
    plan=preflight(args)
    query=['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader']
    compute_query=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader']
    before=subprocess.check_output(query,text=True,timeout=10)
    gpu0=next(row for row in before.splitlines() if row.startswith('0,')).split(',')
    if int(gpu0[4].strip().split()[0])<12288:raise RuntimeError('GPU0 free memory below G1 gate')
    if shutil.disk_usage(args.output_dir.resolve().parent).free<2147483648:raise RuntimeError('G1 disk space below gate')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'run_manifest.json',plan)
    atomic_json(output/'supervision.json',dict(manifest_sha256=hashlib.sha256((output/'run_manifest.json').read_bytes()).hexdigest(),revision=plan['base_revision']))
    (output/'gpu_before.csv').write_text(before)
    (output/'compute_before.csv').write_text(subprocess.check_output(compute_query,text=True,timeout=10))
    command=['timeout','--signal=TERM','--kill-after=15s','240s',sys.executable,str(ROOT/'scripts/tools/source_scale_startup_probe.py'),
        '--manifest',str(output/'run_manifest.json'),'--checkpoint',str(args.checkpoint.resolve())]
    process=None;started=time.monotonic();execution=dict(command=command,monitor_errors=[],timed_out=False)
    try:
        with (output/'runtime.log').open('x') as log,(output/'gpu_samples.jsonl').open('x') as samples:
            process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'source/rl_training')},
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            execution['process_group']=process.pid;atomic_json(output/'execution.json',execution);next_sample=0
            while process.poll() is None:
                elapsed=time.monotonic()-started
                if deadline_expired(elapsed,(output/'runtime.log').read_text(errors='replace')):
                    execution['timed_out']=True;raise TimeoutError('G1 startup or wall timeout')
                if elapsed>=next_sample:
                    samples.write(json.dumps(dict(elapsed_s=elapsed,gpu=subprocess.check_output(query,text=True,timeout=10),
                        processes=subprocess.check_output(compute_query,text=True,timeout=10)))+'\n');samples.flush();next_sample=elapsed+5
                time.sleep(.2)
    except BaseException as error:execution['monitor_errors'].append(type(error).__name__+': '+str(error))
    finally:
        if process is not None:
            if group_exists(process.pid):terminate_group(process,15)
            execution.update(returncode=process.poll(),process_group_empty=not group_exists(process.pid))
        execution['elapsed_s']=time.monotonic()-started
        try:
            (output/'gpu_after.csv').write_text(subprocess.check_output(query,text=True,timeout=10))
            (output/'compute_after.csv').write_text(subprocess.check_output(compute_query,text=True,timeout=10))
            if (_input_hashes()!=plan['input_sha256'] or dependencies()!=plan['dependency_sha256']
                    or BudgetLedger(canonical_budget_path(ROOT)).reference()!=plan['budget_ledger']
                    or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256):
                execution['monitor_errors'].append('G1 source/dependencies/budget changed during execution')
        except Exception as error:execution['monitor_errors'].append(str(error))
        atomic_json(output/'execution.json',execution)
    result=validate(output,plan,execution);atomic_json(output/'acceptance.json',result)
    print(json.dumps(result,indent=2))
    return 0 if result['accepted'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    for name in ('manifest','output-dir','checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();raise SystemExit(prepare(args) if args.action=='prepare' else run(args))
