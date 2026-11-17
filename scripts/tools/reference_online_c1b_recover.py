"""Observe an orphaned bounded training worker without relaunching it.

An interrupted tool session can lose the parent supervisor's exit status while
its detached child continues. Keep the original execution.json unchanged and
record that limitation. Continuation requires complete artifacts and a passed
offline audit, not an invented exit code.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import psutil

import reference_online_c1b as original
import reference_online_c1b_reset_api as pilot
from reference_online_targets import read
from reference_stairs_eval import save,sha
from source_block_runtime import fatal_line

NATIVE_COMPLETED=original.completed


def recovered_completed(directory):
    execution=read(directory/'execution.json')
    if execution['status']=='completed': return NATIVE_COMPLETED(directory)
    assert directory==pilot.OUTPUT/'train'
    recovery=read(directory/'execution_recovery.json')
    assert recovery['status']=='artifacts_complete_exit_status_unavailable'
    assert recovery['original_execution_sha256']==sha(directory/'execution.json')
    for record in recovery['artifacts']:
        assert sha(Path(record['path']))==record['sha256']
    assert not psutil.pid_exists(execution['pid']) or psutil.Process(execution['pid']).status()==psutil.STATUS_ZOMBIE
    report=read(directory/'report.json')
    assert report['status']=='passed' and report['control_steps']==20000
    assert report['base_unchanged'] and report['frozen_unchanged'] and report['base_optimizer_updates']==0
    assert report['trace_sha256']==sha(directory/'trace.npz') and report['final_sha256']==sha(directory/'final.pt')
    return report


def observe():
    pilot.registration()
    directory=pilot.OUTPUT/'train'
    execution=read(directory/'execution.json')
    assert execution['status']=='running'
    pid=execution['pid']
    started=time.time()
    try: process=psutil.Process(pid)
    except psutil.NoSuchProcess: process=None
    created=None
    if process is not None:
        assert process.cmdline()==execution['command']
        created=process.create_time()
        assert process.ppid()==1
    save(directory/'recovery_observer.json',dict(pid=pid,created=created,observer_started=started,
        original_execution_sha256=sha(directory/'execution.json'),observer_pid=os.getpid(),
        already_exited=process is None,
        reason='tool turn interruption; detached child continued after original supervisor exited'))
    failure=None
    while process is not None and psutil.pid_exists(pid) and process.status()!=psutil.STATUS_ZOMBIE:
        text=(directory/'runtime.log').read_text(errors='replace')
        bad=[line for line in text.splitlines() if fatal_line(line)]
        if bad or time.time()-created>3600:
            failure=bad[0] if bad else 'original worker wall deadline exceeded'
            # Only this validated task-owned worker is in scope.
            process.terminate()
            try: process.wait(timeout=10)
            except psutil.TimeoutExpired: process.kill(); process.wait(timeout=10)
            break
        time.sleep(2)
    text=(directory/'runtime.log').read_text(errors='replace')
    bad=[line for line in text.splitlines() if fatal_line(line)]
    files=('report.json','episodes.json','trace.npz','initial.pt','at_10000.pt','at_20000.pt','final.pt','prepared_states.json','runtime.log')
    complete=not failure and not bad and not (directory/'error.json').exists() and all((directory/f).exists() for f in files)
    if complete:
        report=read(directory/'report.json')
        complete=(report['status']=='passed' and report['control_steps']==20000 and
            f"COMPLETE train: episodes={report['complete_episodes']}, adapter_updates={report['adapter_updates']}" in text)
    result=dict(status='artifacts_complete_exit_status_unavailable' if complete else 'failed_or_incomplete',
        pid=pid,created=created,observer_started=started,observed_exit_at=time.time(),
        exit_code=None,exit_code_reason='not a child of replacement observer; original supervisor was interrupted',
        original_execution_sha256=sha(directory/'execution.json'),failure=failure,fatal_messages=bad,
        new_simulations_started=0,original_execution_preserved=True,
        gpu_peak_unknown=True,source_sha256=sha(Path(__file__)),
        artifacts=[dict(path=str(directory/f),sha256=sha(directory/f)) for f in files if (directory/f).exists()])
    save(directory/'execution_recovery.json',result)
    save(pilot.DOCS/'EXECUTION_RECOVERY.json',result)
    assert complete,result
    recovered_completed(directory)
    print(result['status'],flush=True)


def audit(name):
    import reference_online_c1b_audit as auditor
    auditor.completed=recovered_completed
    auditor.torch.set_num_threads(1)
    return auditor.audit(name)


def main():
    if sys.argv[1:]==['--observe-train']: observe(); return 0
    if sys.argv[1:]==['--audit-train']: audit('train'); return 0
    if sys.argv[1:]==['--results']:
        import reference_online_c1b_results as results
        results.completed=recovered_completed
        results.main(); return 0
    assert sys.argv[1:] in (['--kind','target'],['--kind','retention'])
    pilot.registration()
    recovered_completed(pilot.OUTPUT/'train')
    train_audit=read(pilot.DOCS/'AUDIT_train.json')
    assert train_audit['report']==read(pilot.OUTPUT/'train/report.json')
    paths=[Path(__file__),pilot.DOCS/'EXECUTION_RECOVERY.json',pilot.DOCS/'AUDIT_train.json']
    lock=dict(reason='recover supervision after user interrupted tool turn; no experiment settings changed',
        source_lock_sha256=sha(pilot.OUTPUT/'lock.json'),
        files=[dict(path=str(p),sha256=sha(p)) for p in paths])
    if (pilot.DOCS/'RECOVERY_LOCK.json').exists(): assert lock==read(pilot.DOCS/'RECOVERY_LOCK.json')
    else: save(pilot.DOCS/'RECOVERY_LOCK.json',lock)
    original.completed=recovered_completed
    return pilot.main()


if __name__=='__main__': sys.exit(main())
