import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/tjx/rl_training_main/.worktrees/lite3-source-parallel-qualification-20261027')
HERE=Path(__file__).resolve().parent
OUT=Path('/tmp/lite3_source_shared_g3_120env_20261104_v1')
assert json.loads((HERE/'preflight.json').read_text())['status']=='passed'
launch=json.loads((ROOT/'docs/lite3-source-shared-bank-g3-120env-preparation-20261104/raw/launch_command.proposal.json').read_text())
command=launch['argv']
env={**os.environ,**launch['environment_overrides'],'GIT_TERMINAL_PROMPT':'0','GIT_CONFIG_COUNT':'2',
    'GIT_CONFIG_KEY_0':'credential.helper','GIT_CONFIG_VALUE_0':'','GIT_CONFIG_KEY_1':'credential.helper',
    'GIT_CONFIG_VALUE_1':'!gh auth git-credential'}
record=dict(command=command,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    single_supervisor_invocation=True,automatic_retry=False,external_watchdog_s=15000,watchdog_exceeded=False)
record_path=HERE/'supervisor_execution.json'
with record_path.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
process=None;started=time.monotonic()
try:
    with (HERE/'supervisor.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['supervisor_pid']=process.pid
        record_path.write_text(json.dumps(record,indent=2)+'\n')
        try:record['returncode']=process.wait(timeout=15000)
        except subprocess.TimeoutExpired:
            record['watchdog_exceeded']=True;groups={process.pid}
            if (OUT/'execution.json').is_file():
                group=json.loads((OUT/'execution.json').read_text()).get('process_group')
                if isinstance(group,int) and group>1:groups.add(group)
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():continue
                try:
                    if path.stat().st_uid!=os.getuid():continue
                    args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
                    if any(Path(arg).name=='source_scale_motion_validation.py' for arg in args) and str(OUT) in args:
                        groups.add(os.getpgid(int(path.name)))
                except (FileNotFoundError,PermissionError,ProcessLookupError):pass
            for sig in (signal.SIGTERM,signal.SIGKILL):
                for group in groups:
                    try:os.killpg(group,sig)
                    except ProcessLookupError:pass
                if sig==signal.SIGTERM:
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:pass
            record['returncode']=process.wait(timeout=15)
except BaseException as error:
    record['runner_error']=repr(error);raise
finally:
    record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    record_path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
