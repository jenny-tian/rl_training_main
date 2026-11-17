"""Run only the approved C1-a schedule, stopping on any run/audit failure.

Each simulator retains its existing bounded supervisor. Run this coordinator
in its own OS session so a terminal disconnect does not lose that supervisor.
No failed or unaudited attempt is automatically retried or recovered here.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from reference_online_c1a import OUTPUT,DOCS,jobs,completed,source_manifest
from reference_online_targets import read
from reference_stairs_eval import ROOT,save,sha


def pending_jobs():
    assert source_manifest()==read(OUTPUT/'lock.json'),'registered sources changed'
    schedule = jobs()
    attempts = read(OUTPUT/'attempts.json')
    if attempts != schedule[:len(attempts)]:
        raise RuntimeError('attempts are not the registered schedule prefix')
    for attempt in attempts:
        name = attempt['name']
        completed(name)
        audit = read(DOCS/f'AUDIT_{name}.json')
        if audit['status']!='passed' or audit['report_sha256']!=sha(OUTPUT/name/'report.json'):
            raise RuntimeError(f'missing or stale audit: {name}')
        if audit['trace_index_sha256']!=sha(OUTPUT/name/'trace_index.json'):
            raise RuntimeError(f'trace audit mismatch: {name}')
    if len(attempts)<2: raise RuntimeError('both smoke runs must already be completed and audited')
    return schedule[len(attempts):]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    pending = pending_jobs()
    if args.check_only:
        print('Remaining C1-a jobs:',len(pending),[j['name'] for j in pending],flush=True)
        return 0
    lock = (OUTPUT/'sequence.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending = pending_jobs()
    state = dict(status='running',pid=os.getpid(),started_unix=time.time(),
        coordinator_sha256=sha(Path(__file__)),source_lock_sha256=sha(OUTPUT/'lock.json'),
        approval_sha256=sha(DOCS/'APPROVAL.md'),remaining_at_start=[j['name'] for j in pending],completed=[])
    path = OUTPUT/'sequence.json'
    environment = dict(os.environ,PYTHONPATH=str(ROOT/'source/rl_training'))
    try:
        for job in pending:
            name = job['name']
            state.update(current_job=name,phase='simulation'); save(path,state)
            print('SEQUENCE simulation:',name,flush=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/tools/reference_online_c1a.py'),'--job',name],
                           cwd=ROOT,env=environment,check=True)
            state['phase']='cpu_audit'; save(path,state)
            print('SEQUENCE audit:',name,flush=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/tools/reference_online_c1a_audit.py'),'--job',name],
                           cwd=ROOT,env=environment,check=True)
            state['completed'].append(name); save(path,state)
        state.update(current_job=None,phase='statistics'); save(path,state)
        subprocess.run([sys.executable,str(ROOT/'scripts/tools/reference_online_c1a_results.py')],
                       cwd=ROOT,env=environment,check=True)
        state.update(status='completed',phase='awaiting_human_stage_report',ended_unix=time.time())
        save(path,state)
        print('SEQUENCE COMPLETE: C1-a only; no EA or later stage started',flush=True)
        return 0
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),ended_unix=time.time())
        save(path,state)
        raise


if __name__=='__main__': sys.exit(main())
