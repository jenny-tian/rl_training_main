import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path.cwd();scratch=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'source/rl_training'),str(root/'scripts/tools'),str(root/'tests/source_scan')]
import torch
from scale_reset_cpu_fixture import build_fixture
from source_scale_reset_validation import validate
from source_training_entrypoint import _input_hashes
from source_scale_startup_supervisor import dependencies
from rl_training.source_scan.scale_reset import proposal
from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path
from rl_training.source_scan.frozen_actor import CHECKPOINT_SHA256

def write(name,value):
    (scratch/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

started=time.monotonic()
for name,failure in [('cpu_example',False),('cpu_partial_failure',True)]:
    def fault(env,batch):
        if batch==2:env.scene['robot'].data.root_state_w[118,0]+=1
    folder=scratch/name
    plan,execution=build_fixture(folder,reset_fault=fault if failure else None,allow_failure=failure)
    acceptance=validate(folder,plan,execution)
    assert acceptance['accepted'] is not failure,acceptance
    for filename,value in [('fixture_manifest.json',plan),('fixture_execution_inputs.json',execution),
        ('fixture_acceptance.json',acceptance),('provenance.json',dict(synthetic_cpu_fixture=True,
        real_runtime_qualification=False,real_source_transitions=0,real_source_ppo_updates=0,gpu_jobs=0,
        actual_Lite3IsaacLabAdapter=True,native_sensor_bookkeeping=True,physical_backend='supplied_CPU_tensors'))]:
        (folder/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
inputs=_input_hashes();deps=dependencies();ledger=canonical_budget_path(root);reference=BudgetLedger(ledger).reference()
assert deps==json.loads((scratch/'dependencies_before.json').read_text())
assert ledger.read_bytes()==(scratch/'production_ledger_before.json').read_bytes()
write('input_hashes_after.json',inputs);write('dependencies_after.json',deps)
shutil.copyfile(ledger,scratch/'production_ledger_after.json')
write('test_source_sha256.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'tests').rglob('*.py'))})
write('prior_archive_sha256.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['lite3-source-shared-bank-g2-120env-preparation-20261103','lite3-source-shared-bank-g2-120env-runtime-20261104']
    for p in sorted((root/'docs'/name).rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
checkpoint=Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt')
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==CHECKPOINT_SHA256
write('checkpoint_reference.json',dict(path=str(checkpoint),sha256=CHECKPOINT_SHA256))
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
draft=proposal(revision=revision,remote=None,inputs=inputs,dependencies=deps,budget_reference=reference,
    output_dir='/tmp/lite3_source_shared_g2_120env_20261104_v2')
assert not Path(draft['output_dir']).exists()
write('g2.v2.proposal.draft.json',draft)
write('launch_command.proposal.json',dict(status='proposed_not_executed',separate_user_approval_required=True,
    argv=[sys.executable,'scripts/tools/source_scale_reset_supervisor.py','run','--num-envs','120',
        '--manifest','/tmp/lite3_source_g2_120env_20261104_v2.approved.json','--output-dir',draft['output_dir'],
        '--checkpoint',str(checkpoint)],environment_overrides=dict(CUDA_VISIBLE_DEVICES='0'),
    manifest_to_freeze_after_commit='/tmp/lite3_source_g2_120env_20261104_v2.proposal.json'))
assert not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled()
write('artifact_generation.json',dict(elapsed_seconds=time.monotonic()-started,synthetic_examples=2,
    actual_adapter_callback=True,installed_sensor_methods_executed=True,backend='CPU_tensors',
    cuda_initialized=False,gpu_queries=0,gpu_jobs=0,isaac_sim_launches=0,real_source_transitions=0,real_source_ppo_updates=0))
print(json.dumps(dict(CPU_examples=2,inputs=len(inputs),dependencies=len(deps),unapproved_v2_draft=True),indent=2))
