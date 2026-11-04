import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT=Path('/home/tjx/rl_training_main/.worktrees/lite3-source-parallel-qualification-20261027')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
from source_training_entrypoint import _git_state,_input_hashes
from source_scale_startup_supervisor import dependencies
from rl_training.source_scan.scale_motion import validate_plan
from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256

def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

proposal=Path('/tmp/lite3_source_g3_120env_20261104_v1.proposal.json')
approved=Path('/tmp/lite3_source_g3_120env_20261104_v1.approved.json')
expected='424fee88b75b2830d991b1cc7724b0763ccf8f9c77bb6c90d7826855f3eafc4a'
assert hashlib.sha256(proposal.read_bytes()).hexdigest()==expected
plan=json.loads(proposal.read_text());output=Path(plan['output_dir'])
assert plan['base_revision']=='2135009e30b854a2522761ef0f397fbf3c6a8101'
assert not approved.exists() and not output.exists()
matches=[]
names={'source_scale_motion_supervisor.py','source_scale_motion_probe.py','source_scale_motion_validation.py'}
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        if path.stat().st_uid!=os.getuid():continue
        args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if any(Path(arg).name in names for arg in args):matches.append(dict(pid=int(path.name),argv=args))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
audit=dict(output_existed_before=output.exists(),approved_file_existed_before=approved.exists(),
    prior_matching_processes=matches,prior_matching_process_found=bool(matches))
write('attempt_audit.json',audit);assert not matches
revision,remote,dirty=_git_state();assert revision==remote==plan['base_revision'] and not dirty
identity=json.loads(subprocess.check_output(['gh','api','repos/jenny-tian/rl_training_main'],cwd=ROOT,text=True,timeout=30))
assert identity['private'] is True and identity['owner']['login']=='jenny-tian'
inputs=_input_hashes();deps=dependencies();ledger_path=canonical_budget_path(ROOT);reference=BudgetLedger(ledger_path).reference()
assert inputs==plan['input_sha256'] and len(inputs)==249 and deps==plan['dependency_sha256'] and len(deps)==51
assert reference==plan['budget_ledger']
checkpoint=Path(json.loads((ROOT/'docs/lite3-source-shared-bank-g3-120env-preparation-20261104/raw/checkpoint_reference.json').read_text())['path'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==CHECKPOINT_SHA256
assert subprocess.check_output(['git','submodule','status','--','deep_robotics_model'],cwd=ROOT,text=True).startswith(' b452a1f404d37155400242f18e454fed8bcc51b8 ')
prep=ROOT/'docs/lite3-source-shared-bank-g3-120env-preparation-20261104'
review=runpy.run_path(str(prep/'review.py'))['main'](ROOT)
assert review==json.loads((prep/'review.json').read_text());write('preparation_review.json',review)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True,timeout=10)
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,timeout=10)
(HERE/'gpu_preflight.csv').write_text(gpu);(HERE/'compute_preflight.csv').write_text(compute)
gpu0=next(row for row in gpu.splitlines() if row.startswith('0,')).split(',')
free=int(gpu0[4].strip().split()[0]);disk_free=shutil.disk_usage(output.parent).free
assert free>=plan['minimum_free_vram_mib'] and disk_free>=plan['minimum_free_disk_bytes']
validate_plan(plan,revision=revision,inputs=inputs,dependencies=deps,budget_reference=reference,output_dir=output,approved=False)
for name,value in [('input_hashes_before.json',inputs),('dependencies_before.json',deps),('production_ledger_reference.json',reference)]:write(name,value)
shutil.copyfile(ledger_path,HERE/'production_ledger_before.json');shutil.copyfile(proposal,HERE/'frozen_proposal.json')
authorization=plan|dict(status='approved',user_approved=True)
validate_plan(authorization,revision=revision,inputs=inputs,dependencies=deps,budget_reference=reference,output_dir=output)
with approved.open('x') as stream:json.dump(authorization,stream,indent=2,allow_nan=False);stream.write('\n')
shutil.copyfile(approved,HERE/'approved_manifest.json')
write('approval.json',dict(user_message='同意',scope='one complete GPU0 120-env G3 v1 run and independent CPU acceptance only',
    proposal_sha256=expected,approved_manifest=str(approved),changed_fields=['status','user_approved'],
    approved_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False))
import torch
assert not torch.cuda.is_initialized()
result=dict(status='passed',revision=revision,remote_revision=remote,working_tree_clean=True,
    repository=identity['full_name'],repository_private=True,checkpoint_path=str(checkpoint),checkpoint_sha256=CHECKPOINT_SHA256,
    gpu0_free_mib=free,gpu0_used_mib=int(gpu0[3].strip().split()[0]),disk_free_bytes=disk_free,
    production_inputs=249,installed_dependencies=51,preparation_review_passed=True,proposal_sha256=expected,
    output_dir=str(output),approved_manifest=str(approved),cuda_initialized=False)
write('preflight.json',result);print(json.dumps(result,indent=2))
