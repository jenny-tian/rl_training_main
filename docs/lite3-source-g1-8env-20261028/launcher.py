import datetime, hashlib, json, os, pathlib, signal, subprocess, time
root=pathlib.Path('/home/tjx/rl_training_main/.worktrees/lite3-source-parallel-qualification-20261027')
run=pathlib.Path('/tmp/lite3_g1_8env_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
run.mkdir()
print(str(run),flush=True)
def cmd(args): return subprocess.check_output(args,cwd=root,text=True).strip()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not cmd(['git','status','--porcelain']), 'worktree must be clean'
gpu=cmd(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'])
(run/'gpu_before.txt').write_text(gpu+'\n')
processes=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'])
(run/'gpu_processes_before.txt').write_text(processes+'\n')
uuid=cmd(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'])
assert uuid not in processes, 'GPU0 has compute processes'
fields=gpu.splitlines()[0].split(',')
assert int(fields[3].split()[0])-int(fields[2].split()[0])>=8192
files=cmd(['git','ls-files']).splitlines()
hashes={p:sha(root/p) for p in files if (root/p).is_file()}
audit=json.loads((root/'docs/lite3-usd-collision-geometry-audit-20261027.json').read_text())
for asset in audit['assets'].values(): assert sha(root/asset['path'])==asset['sha256']
subprocess.run(['git','archive','--format=tar','-o',str(run/'code.tar'),'HEAD'],cwd=root,check=True)
launch='source /home/tjx/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && PYTHONPATH=source/rl_training python scripts/tools/source_parallel_terrain_startup_smoke.py --headless --device cuda:0 --num-envs 8 --seed 130001 --output '+str(run/'result.json')+' --trace-output '+str(run/'trace.log')
manifest={'revision':cmd(['git','rev-parse','HEAD']),'model_revision':cmd(['git','-C','deep_robotics_model','rev-parse','HEAD']),'file_sha256':hashes,'code_tar_sha256':sha(run/'code.tar'),'launcher_sha256':sha(pathlib.Path(__file__)),'command':launch,'gpu_snapshot':gpu,'env_seed':130001,'sampler_seed':130001,'terrain_seed':10001,'num_envs':8,'startup_timeout_s':120,'total_timeout_s':180,'policy_loaded':False,'ppo_updates':0}
(run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with (run/'dependencies.txt').open('w') as f: subprocess.run(['/home/tjx/anaconda3/envs/env_isaaclab/bin/python','-m','pip','freeze'],stdout=f,stderr=subprocess.STDOUT)
started=time.monotonic(); peak=0; stop=None
with (run/'console.log').open('w') as log, (run/'gpu_samples.jsonl').open('w') as samples:
 p=subprocess.Popen(['/bin/bash','-lc',launch],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  elapsed=time.monotonic()-started
  trace=(run/'trace.log').read_text() if (run/'trace.log').exists() else ''
  if elapsed>180 or (elapsed>120 and 'initial-reset-complete' not in trace):
   stop='startup_timeout' if 'initial-reset-complete' not in trace else 'total_timeout'
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  sample=cmd(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
  peak=max(peak,int(sample.split(',')[0]))
  samples.write(json.dumps({'elapsed_s':elapsed,'gpu':sample})+'\n');samples.flush()
  time.sleep(1)
 result={'exit_code':p.returncode,'stop_reason':stop,'elapsed_s':time.monotonic()-started,'sampled_gpu_peak_mib':peak,'sampling_interval_s':1,'code_hashes_unchanged':all(sha(root/p)==h for p,h in hashes.items())}
(run/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
(run/'gpu_after.txt').write_text(cmd(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'])+'\n')
print(json.dumps(result),flush=True)
