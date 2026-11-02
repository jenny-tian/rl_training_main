from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

root = Path('/home/tjx/rl_training_main/.worktrees/lite3-source-parallel-qualification-20261027')
out = Path('/tmp/lite3_source_training_integration_smoke_20261102')
if out.exists():
    raise SystemExit('Output directory already exists; no retry permitted')
audit = json.loads((root/'docs/lite3-source-training-integration-20261102/cpu_integration_audit.json').read_text())
for rel, digest in audit['input_sha256'].items():
    if hashlib.sha256((root/rel).read_bytes()).hexdigest() != digest:
        raise SystemExit('Input hash changed: '+rel)
if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
    raise SystemExit('Worktree not clean')
revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if revision != 'ae19a40015c81e1c772168530761124188abfe24':
    raise SystemExit('Unexpected revision')
query = ['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader']
gpu = subprocess.check_output(query,text=True)
row = [r for r in gpu.splitlines() if r.split(',')[0].strip() == '0'][0]
free = int(row.split(',')[3].strip().split()[0])
if free < 12288:
    raise SystemExit('GPU0 resource gate failed')
out.mkdir()
(out/'gpu_before.csv').write_text(gpu)
compute_query = ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader']
(out/'compute_before.csv').write_text(subprocess.check_output(compute_query,text=True))
plan = json.loads((root/'docs/lite3-source-training-integration-20261102/next_stage.json').read_text())
plan.update(status='approved_for_single_attempt',user_approved=True,approval_message='同意',
            prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_revision=revision,
            remote_revision_verified=revision,input_sha256=audit['input_sha256'],gpu0_free_mib=free)
(out/'run_manifest.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
command = ['timeout','--signal=TERM','--kill-after=15s','240s',
           '/home/tjx/anaconda3/envs/env_isaaclab/bin/python',
           'scripts/tools/source_training_integration_smoke.py','--headless','--device','cuda:0',
           '--checkpoint','/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt',
           '--output',str(out/'result.json')]
env = dict(os.environ, PYTHONPATH=str(root/'source/rl_training'))
started = time.monotonic()
with (out/'runtime.log').open('x') as log, (out/'gpu_samples.jsonl').open('x') as samples:
    process = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    execution = {'command':command,'supervisor_pid':os.getpid(),'process_group':process.pid,
                 'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    print('Started single attempt; process group',process.pid,'GPU0 free MiB',free,flush=True)
    while process.poll() is None:
        sample = subprocess.run(query,capture_output=True,text=True,timeout=10)
        samples.write(json.dumps({'elapsed_s':time.monotonic()-started,'returncode':sample.returncode,
                                  'gpu':sample.stdout,'error':sample.stderr})+'\n')
        samples.flush()
        time.sleep(5)
    execution.update(returncode=process.returncode,elapsed_s=time.monotonic()-started,
                     end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
(out/'gpu_after.csv').write_text(subprocess.check_output(query,text=True))
(out/'compute_after.csv').write_text(subprocess.check_output(compute_query,text=True))
print('Attempt ended:',json.dumps(execution),flush=True)
if (out/'result.json').exists():
    result = json.loads((out/'result.json').read_text())
    print('Result:',result.get('status'),'control steps:',result.get('simulation_control_steps'),
          'PPO updates:',result.get('ppo_updates'),flush=True)
    if result.get('error'):
        print(result['error'],flush=True)
raise SystemExit(0 if process.returncode == 0 else 1)
