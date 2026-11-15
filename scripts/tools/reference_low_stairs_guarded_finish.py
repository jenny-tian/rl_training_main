"""等待已启动的前 100 次；依照预定检查结束已批准阶段，不重启失败训练。"""
import subprocess
import sys
import time

from reference_stairs_eval import ROOT, save
from reference_low_stairs_guarded_bank import OUTPUT
from reference_low_stairs_guarded_train import directory
from reference_low_stairs_guarded_evaluate import read


def main():
    deadline=time.monotonic()+7200
    execution=directory(1)/'execution.json'
    if not execution.exists():
        raise RuntimeError('只接管已启动的训练')
    while True:
        state=read(execution)
        if state['status']=='completed': break
        if state['status']!='running' or time.monotonic()>deadline:
            raise RuntimeError('前 100 次训练失败或超时；不重启')
        time.sleep(30)
    def run(script,*args):
        subprocess.run([sys.executable,str(ROOT/'scripts/tools'/script),*args],cwd=ROOT,check=True)
    run('reference_low_stairs_guarded_evaluate.py','--updates','100','--mode','gate')
    gate=read(OUTPUT/'gate100.json')
    updates=100
    if gate['status']=='passed':
        run('reference_low_stairs_guarded_train.py','--phase','2')
        updates=200
    elif gate['status']!='stopped_for_regression':
        raise RuntimeError('中途检查状态未知')
    run('reference_low_stairs_guarded_evaluate.py','--updates',str(updates),'--mode','final')
    run('reference_low_stairs_guarded_analyze.py','--updates',str(updates))
    save(OUTPUT/'stage_complete.json',dict(status='passed',training_updates=updates,
        stop_reason=gate['status'] if updates==100 else 'authorized_budget_reached',
        automatic_extension=False,summary=str(ROOT/'docs/lite3-low-stairs-guarded-trial-20261115/summary.json')))


if __name__=='__main__':
    main()
