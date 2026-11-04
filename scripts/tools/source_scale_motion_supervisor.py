"""Freeze or supervise one separately approved G3 run; preparation uses CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
from rl_training.source_scan.scale_motion import contract_for,proposal,validate_plan,validate_predecessor,same_value
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256
from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path
from rl_training.source_scan.smoke_artifacts import atomic_json
from source_training_entrypoint import _git_state,_input_hashes
from source_scale_startup_supervisor import dependencies
from source_live_ppo_supervisor import group_exists,terminate_group


def preflight(args,*,child=False):
    plan=json.loads(args.manifest.read_text())
    if plan.get('status')!='approved' or plan.get('user_approved') is not True:raise PermissionError('separate G3 GPU approval is required')
    contract_for(args.num_envs);revision,remote,dirty=_git_state()
    if dirty or revision!=remote:raise ValueError('G3 requires clean committed and remote-matching source')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0'):raise ValueError('G3 GPU0 mapping differs')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:raise ValueError('registered actor checkpoint changed')
    output=Path(plan['output_dir']).resolve();inputs=_input_hashes();deps=dependencies();reference=BudgetLedger(canonical_budget_path(ROOT)).reference()
    if child:
        if (args.manifest.resolve()!=output/'run_manifest.json' or (output/'result.json').exists()
            or json.loads((output/'supervision.json').read_text())!=dict(revision=revision,
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())):raise ValueError('G3 child ticket/restart differs')
        expected=proposal(revision=revision,remote=remote,inputs=inputs,dependencies=deps,budget_reference=reference,output_dir=output)
        for k,v in expected.items():
            if k not in ('status','user_approved') and not same_value(plan.get(k),v):raise ValueError('G3 child input mismatch: '+k)
        validate_predecessor(plan['predecessor_qualification'])
    else:
        if args.output_dir.resolve()!=output:raise ValueError('G3 output differs from approved manifest')
        validate_plan(plan,revision=revision,inputs=inputs,dependencies=deps,budget_reference=reference,output_dir=output)
    return plan


def prepare(args):
    if args.manifest.exists() or args.output_dir.exists():raise FileExistsError('G3 proposal/attempt already exists')
    revision,remote,dirty=_git_state()
    if dirty or revision!=remote:raise ValueError('freeze G3 only after commit, push and remote verification')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:raise ValueError('original actor changed')
    plan=proposal(revision=revision,remote=remote,inputs=_input_hashes(),dependencies=dependencies(),
        budget_reference=BudgetLedger(canonical_budget_path(ROOT)).reference(),output_dir=args.output_dir,num_envs=args.num_envs)
    validate_plan(plan,revision=revision,inputs=plan['input_sha256'],dependencies=plan['dependency_sha256'],
        budget_reference=plan['budget_ledger'],output_dir=args.output_dir,approved=False)
    with args.manifest.open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='frozen_not_approved',base_revision=revision,manifest=str(args.manifest),gpu_jobs=0,isaac_sim_launches=0)))
    return 0


def deadline_expired(elapsed,text,plan):
    return elapsed>plan['wall_timeout_s'] or elapsed>plan['startup_timeout_s'] and '[scale-g3] initial-ready' not in text


def run(args):
    plan=preflight(args);out=args.output_dir.resolve()
    gpu_query=['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader']
    compute_query=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader']
    query=lambda cmd:subprocess.check_output(cmd,text=True,timeout=10)
    before=query(gpu_query);gpu0=next(v for v in before.splitlines() if v.startswith('0,')).split(',')
    if int(gpu0[4].strip().split()[0])<plan['minimum_free_vram_mib']:raise RuntimeError('G3 GPU0 memory gate failed')
    if shutil.disk_usage(out.parent).free<plan['minimum_free_disk_bytes']:raise RuntimeError('G3 disk gate failed')
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'run_manifest.json',plan)
    atomic_json(out/'supervision.json',dict(revision=plan['base_revision'],manifest_sha256=hashlib.sha256((out/'run_manifest.json').read_bytes()).hexdigest()))
    (out/'gpu_before.csv').write_text(before);(out/'compute_before.csv').write_text(query(compute_query))
    command=['timeout','--signal=TERM','--kill-after=15s',str(plan['wall_timeout_s'])+'s',sys.executable,
        str(ROOT/'scripts/tools/source_scale_motion_probe.py'),'--manifest',str(out/'run_manifest.json'),
        '--checkpoint',str(args.checkpoint.resolve()),'--num-envs','120']
    process=None;started=time.monotonic();execution=dict(command=command,timed_out=False,monitor_errors=[])
    try:
        with (out/'runtime.log').open('x') as log,(out/'gpu_samples.jsonl').open('x') as samples:
            process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'source/rl_training')},
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            execution['process_group']=process.pid;atomic_json(out/'execution.json',execution);next_sample=0
            while process.poll() is None:
                elapsed=time.monotonic()-started
                if deadline_expired(elapsed,(out/'runtime.log').read_text(errors='replace'),plan):
                    execution['timed_out']=True;raise TimeoutError('G3 startup/wall deadline reached')
                if (out/'runtime.log').stat().st_size>plan['runtime_log_byte_limit']:raise RuntimeError('G3 runtime log ceiling reached')
                if elapsed>=next_sample:
                    samples.write(json.dumps(dict(elapsed_s=elapsed,gpu=query(gpu_query),processes=query(compute_query)))+'\n');samples.flush();next_sample=elapsed+5
                time.sleep(.2)
    except BaseException as error:execution['monitor_errors'].append(type(error).__name__+': '+str(error))
    finally:
        if process is not None:
            if group_exists(process.pid):terminate_group(process,15)
            execution.update(returncode=process.poll(),process_group_empty=not group_exists(process.pid))
        execution['elapsed_s']=time.monotonic()-started
        try:
            (out/'gpu_after.csv').write_text(query(gpu_query));(out/'compute_after.csv').write_text(query(compute_query))
            validate_predecessor(plan['predecessor_qualification'])
            if (_input_hashes()!=plan['input_sha256'] or dependencies()!=plan['dependency_sha256']
                    or BudgetLedger(canonical_budget_path(ROOT)).reference()!=plan['budget_ledger']
                    or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256):raise ValueError('G3 immutable inputs changed')
        except Exception as error:execution['monitor_errors'].append(str(error))
        atomic_json(out/'execution.json',execution)
    validation=None
    try:
        cmd=['timeout','--signal=TERM','--kill-after=15s',str(plan['validation_timeout_s'])+'s',sys.executable,
            str(ROOT/'scripts/tools/source_scale_motion_validation.py'),'--folder',str(out),'--checkpoint',str(args.checkpoint.resolve())]
        with (out/'validation.log').open('x') as log:
            validation=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            execution['validation_returncode']=validation.wait(timeout=plan['validation_timeout_s']+30)
    except Exception as error:execution['monitor_errors'].append('CPU validation: '+str(error))
    finally:
        if validation is not None and group_exists(validation.pid):terminate_group(validation,15)
        execution['validation_process_group_empty']=validation is None or not group_exists(validation.pid)
        atomic_json(out/'execution.json',execution)
    result=json.loads((out/'acceptance.json').read_text()) if (out/'acceptance.json').exists() else dict(accepted=False,reasons=['G3 CPU validation did not complete'])
    if execution['monitor_errors'] or execution.get('validation_returncode')!=0 or not execution['validation_process_group_empty']:
        result['accepted']=False;result.setdefault('reasons',[]).append('G3 supervision or CPU validation failed')
    atomic_json(out/'acceptance.json',result);print(json.dumps(result,indent=2));return 0 if result['accepted'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--num-envs',type=int,choices=(120,),required=True)
    for name in ('manifest','output-dir','checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();raise SystemExit(prepare(args) if args.action=='prepare' else run(args))
