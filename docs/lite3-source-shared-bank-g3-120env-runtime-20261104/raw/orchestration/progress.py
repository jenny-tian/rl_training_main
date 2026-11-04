import datetime
import json
import os
from pathlib import Path
import re
import csv
import time

HERE=Path(__file__).resolve().parent
OUT=Path('/tmp/lite3_source_shared_g3_120env_20261104_v1')
cursor_path=HERE/'progress_cursor.json'
cursor=json.loads(cursor_path.read_text()) if cursor_path.exists() else dict(offset=0,step=0,episodes=0,flat_witnesses=[],last_rows=[])
step_file=OUT/'result.steps.jsonl'
if step_file.exists():
    with step_file.open('rb') as stream:
        stream.seek(cursor['offset'])
        while True:
            offset=stream.tell();line=stream.readline(512*1024**2+1)
            if not line or not line.endswith(b'\n'):break
            row=json.loads(line);cursor.update(offset=stream.tell(),step=row['step'])
            cursor['episodes']+=sum(row['done'])
            terminal=row['terminal']
            if terminal is not None and row['step']<=2000:
                for j,i in enumerate(terminal['ids']):
                    if row['columns_before'][i]==0 and terminal['completed'][j]:
                        cursor['flat_witnesses'].append(dict(step=row['step'],env_id=i,hold=terminal['hold'][j]))
            cursor['last_rows']=(cursor['last_rows']+[dict(step=row['step'],done=sum(row['done']))])[-5:]
summary=dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_controls=cursor['step'],
    planned_controls=2001,completed_diagnostic_transitions=120*cursor['step'],recorded_episode_endings=cursor['episodes'],
    candidate_flat_completion_witnesses=cursor['flat_witnesses'],recent_controls=cursor['last_rows'])
for filename in ['execution.json','result.json','acceptance.json']:
    path=OUT/filename
    if path.exists():
        data=json.loads(path.read_text())
        keys=['status','accepted','reasons','simulation_control_steps','completed_diagnostic_transitions','motion_samples',
            'reset_batches','reset_attempts','returncode','process_group','process_group_empty','timed_out',
            'validation_returncode','validation_process_group_empty']
        summary[filename]={key:data[key] for key in keys if key in data}
for name in ['events.jsonl.gz','result.steps.jsonl','runtime.log','validation.log']:
    path=OUT/name
    if path.exists():summary[name+'_bytes']=path.stat().st_size
log=OUT/'runtime.log'
if log.exists():
    lines=log.read_text(errors='replace').splitlines()
    summary['runtime_markers']=[line for line in lines if '[scale-g3]' in line or 'Traceback' in line
        or line.startswith(('AssertionError:','ValueError:','RuntimeError:','FloatingPointError:'))][-8:]
processes=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        if path.stat().st_uid!=os.getuid():continue
        args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
        names=[Path(arg).name for arg in args]
        if any(name in names for name in ['source_scale_motion_probe.py','source_scale_motion_supervisor.py','source_scale_motion_validation.py']):
            processes.append(dict(pid=int(path.name),process_group=os.getpgid(int(path.name)),argv=args))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
summary['owned_task_processes']=processes
for item in processes:
    if any(Path(arg).name=='source_scale_motion_probe.py' for arg in item['argv']) and Path(item['argv'][0]).name.startswith('python'):
        cursor['worker_pid']=item['pid']
gpu_path=OUT/'gpu_samples.jsonl'
if gpu_path.exists():
    with gpu_path.open('rb') as stream:
        stream.seek(cursor.get('gpu_offset',0))
        while True:
            line=stream.readline()
            if not line or not line.endswith(b'\n'):break
            sample=json.loads(line);cursor['gpu_offset']=stream.tell()
            rows=lambda text:[[v.strip() for v in row] for row in csv.reader(text.splitlines()) if row]
            gpu0=next(row for row in rows(sample['gpu']) if row[0]=='0')
            cursor['gpu_samples']=cursor.get('gpu_samples',0)+1
            cursor['gpu0_peak_used_mib']=max(cursor.get('gpu0_peak_used_mib',0),int(gpu0[3].split()[0]))
            cursor['gpu0_min_free_mib']=min(cursor.get('gpu0_min_free_mib',10**9),int(gpu0[4].split()[0]))
            cursor['gpu0_latest']=dict(used_mib=int(gpu0[3].split()[0]),free_mib=int(gpu0[4].split()[0]),utilization=gpu0[6],temperature_c=gpu0[7])
            for row in rows(sample['processes']):
                if row[1]==str(cursor.get('worker_pid')):cursor['worker_peak_mib']=max(cursor.get('worker_peak_mib',0),int(row[3].split()[0]))
summary['resources']={k:cursor[k] for k in ('gpu_samples','gpu0_peak_used_mib','gpu0_min_free_mib','gpu0_latest','worker_pid','worker_peak_mib') if k in cursor}
cursor_path.write_text(json.dumps(cursor,indent=2)+'\n')
with (HERE/'process_observations.jsonl').open('a') as stream:stream.write(json.dumps(dict(observed_at_utc=summary['observed_at_utc'],processes=processes))+'\n')
with (HERE/'progress_observations.jsonl').open('a') as stream:stream.write(json.dumps(summary)+'\n')
summary['owned_task_processes']=[dict(pid=p['pid'],process_group=p['process_group'],
    role=next((Path(a).name for a in p['argv'] if Path(a).name in ['source_scale_motion_probe.py','source_scale_motion_supervisor.py','source_scale_motion_validation.py']),'unknown')) for p in processes]
print(json.dumps(summary,indent=2))
