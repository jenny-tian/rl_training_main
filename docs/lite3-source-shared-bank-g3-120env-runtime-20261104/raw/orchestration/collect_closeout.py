import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path('/home/tjx/rl_training_main/.worktrees/lite3-source-parallel-qualification-20261027')
HERE=Path(__file__).resolve().parent
OUT=Path('/tmp/lite3_source_shared_g3_120env_20261104_v1')
ARCHIVE=ROOT/'docs/lite3-source-shared-bank-g3-120env-runtime-20261104'
STORE=Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/source-g3-evidence/lite3_source_shared_g3_120env_20261104_v1')
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
from source_training_entrypoint import _input_hashes
from source_scale_startup_supervisor import dependencies
from rl_training.source_scan.training_budget_ledger import canonical_budget_path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

plan=json.loads((OUT/'run_manifest.json').read_text())
report=json.loads((OUT/'result.json').read_text())
inputs=_input_hashes();deps=dependencies();ledger=canonical_budget_path(ROOT)
before_ledger=HERE/'production_ledger_before.json'
checkpoint=Path(json.loads((HERE/'preflight.json').read_text())['checkpoint_path'])
assert inputs==plan['input_sha256'] and deps==plan['dependency_sha256']
assert ledger.read_bytes()==before_ledger.read_bytes()
assert digest(checkpoint)==plan['checkpoint_sha256']
assert report['actor_sha256_before']==report['actor_sha256_after']
assert report['status']=='failed' and 'nonfinite scanner evidence' in report['error']
assert json.loads((OUT/'acceptance.json').read_text())['accepted'] is False
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==plan['base_revision']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
assert subprocess.check_output(['git','submodule','status','--','deep_robotics_model'],cwd=ROOT,text=True).startswith(' b452a1f404d37155400242f18e454fed8bcc51b8 ')
write(HERE/'input_hashes_after.json',inputs);write(HERE/'dependencies_after.json',deps)
shutil.copyfile(ledger,HERE/'production_ledger_after.json')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True,timeout=10)
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,timeout=10)
(HERE/'gpu_postflight.csv').write_text(gpu);(HERE/'compute_postflight.csv').write_text(compute)
names={'source_scale_motion_supervisor.py','source_scale_motion_probe.py','source_scale_motion_validation.py'}
matches=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if any(Path(a).name in names for a in args):matches.append(dict(pid=int(p.name),argv=args))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not matches
owned=[1712935,1713160,1713161]
assert all(not Path('/proc',str(pid)).exists() for pid in owned)
assert all(not re.search(r',\s*'+str(pid)+r',',compute) for pid in owned)
write(HERE/'process_cleanup_audit.json',dict(matching_processes=matches,owned_pids=owned,owned_pids_absent=True,gpu_worker_absent=True))
last=None;count=0;done_count=0
with (OUT/'result.steps.jsonl').open() as f:
    for line in f:
        row=json.loads(line);count+=1
        assert row['step']==count
        last=row;done_count+=sum(row['done'])
assert count==report['steps_written']==506 and report['simulation_control_steps']==507
write(HERE/'last_complete_control.json',last)
cc=report['contact_capture'];ground=cc['ground_contact_points']
log=(OUT/'runtime.log').read_text(errors='replace')
summary=dict(status='failed',runtime_qualification=False,source_revision=plan['base_revision'],
    simulation_control_steps=507,complete_control_records=count,reported_diagnostic_transitions=60840,
    diagnostic_transitions_with_complete_control_records=count*120,physics_samples=report['motion_samples'],
    ordinary_episode_endings=int(done_count),reset_batches=report['reset_batches'],
    pending_recovery_ids=report['pending_recovery_ids'],flat_completion_witnesses=report['flat_completion_witnesses'],
    all_120_envs_ground_points_reported=len(ground)==120 and min(ground.values())>0,
    ground_points_per_env_min=min(ground.values()),ground_points_per_env_max=max(ground.values()),
    cross_environment_contacts_reported=cc['cross_environment_contacts'],
    contact_capacity_peak_used=cc['peak_used'],contact_capacity=cc['topology']['capacity'],
    max_body_speed_mps=report['max_body_speed_mps'],max_body_displacement_m=report['max_body_displacement_m'],
    max_finite_difference_acceleration_mps2=report['max_finite_difference_acceleration_mps2'],
    measurements_are_partial_not_independently_qualified=True,training_transitions_this_attempt=0,ppo_updates_this_attempt=0,
    original_checkpoint_sha256=digest(checkpoint),production_ledger_sha256=digest(ledger),
    actor_sha256=report['actor_sha256_after'],source_inputs_unchanged=len(inputs),dependencies_unchanged=len(deps),
    error='FloatingPointError: nonfinite scanner evidence',invalid_sensor_env_ray_not_identified=True,
    no_user_stop_signal_sent=True,stop_reason='automatic hard failure before user stop reply',
    stop_user_reply='现在停止，先看合并交付方案',automatic_retry=False,new_gpu_launches_after_failure=0,
    warnings_in_runtime_log=sum('[Warning]' in s or ' WARNING:' in s for s in log.splitlines()),
    traceback_headers_in_runtime_log=log.count('Traceback (most recent call last):'),
    runtime_error_tags=log.count('[Error]')+log.count('[ERROR]'))
write(HERE/'closeout_summary.json',summary)
write(HERE/'stop_authorization.json',dict(user_reply='现在停止，先看合并交付方案',
    observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_already_exited=True,
    no_signal_necessary=True,next_stage_user_approved=False))
import torch
assert not torch.cuda.is_initialized()
write(HERE/'postflight.json',dict(status='passed',captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_inputs_unchanged=True,dependency_hashes_unchanged=True,checkpoint_unchanged=True,
    actor_unchanged=True,production_ledger_byte_identical=True,processes_cleaned=True,cuda_initialized=False,
    source_input_count=len(inputs),dependency_count=len(deps),runtime_accepted=False))
assert not STORE.exists() and not ARCHIVE.exists()
STORE.mkdir(parents=True)
shutil.copytree(OUT,STORE/'run');shutil.copytree(HERE,STORE/'orchestration')
logs=[Path('/home/tjx/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261104_154917.log'),Path('/tmp/isaaclab/logs/isaaclab_2026-11-04_15-49-21.log')]
(STORE/'logs').mkdir()
for path in logs:
    assert path.is_file();shutil.copy2(path,STORE/'logs'/path.name)
(ARCHIVE/'raw').mkdir(parents=True)
bulk=[];small=[]
for path in sorted(STORE.rglob('*')):
    if not path.is_file():continue
    relative=path.relative_to(STORE);item=dict(path=str(relative),size_bytes=path.stat().st_size,sha256=digest(path))
    if item['size_bytes']>=20*1024**2:bulk.append(item)
    else:
        target=ARCHIVE/'raw'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);small.append(item)
write(ARCHIVE/'bulk_manifest.json',dict(schema='lite3_local_bulk_evidence_manifest_v1',
    artifact_root=str(STORE),original_output=str(OUT),original_orchestration=str(HERE),
    storage_scope='local durable copy under git common metadata; bulk not uploaded to GitHub',
    files=bulk,all_evidence_files=len(bulk)+len(small)))
write(ARCHIVE/'provenance.json',dict(schema='lite3_g3_failed_run_provenance_v1',
    executed_revision=plan['base_revision'],raw_originals_copied_byte_identically=True,
    runtime_accepted=False,automatic_retry=False,gpu_attempts=1,source_modified=False,
    bulk_evidence_index='bulk_manifest.json',small_evidence_files=len(small),bulk_evidence_files=len(bulk),
    total_evidence_bytes=sum(x['size_bytes'] for x in small+bulk)))
print(json.dumps(dict(summary=summary,archive=str(ARCHIVE),artifact_root=str(STORE),
    small_files=len(small),bulk_files=bulk),indent=2))
