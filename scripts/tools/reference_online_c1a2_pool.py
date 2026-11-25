"""跑完一个目标的训练和评估；已通过的任务跳过，任何任务失败就停止派发新任务。

与 reference_online_c1a2_sequence.py 的区别是评估任务可以同时进行几个。每个任务开始前在全局文件锁里挑 GPU：
取空闲显存最多的一块，并扣掉最近 150 秒内刚派出去、显存还没有占上的任务，剩余不少于 --minimum-mib 才开始。

用法：C1A2_VARIANT=… C1A2_TARGET=… python reference_online_c1a2_pool.py --workers 4
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from reference_online_c1a2 import CANDIDATE, LEAD, OUTPUT, RATE, REFLEX, SOURCE, TARGET, jobs
from reference_stairs_eval import ROOT

LOCK, PENDING = '/tmp/c1a2_gpu.lock', '/tmp/c1a2_gpu_pending.json'
RESERVE_MIB, SETTLE_S, MINIMUM_MIB = 3500, 150, 9000  # 每个任务实际占用约 2.7 GB


def passed(name):
    path = OUTPUT / name / 'report.json'
    return path.exists() and json.loads(path.read_text())['status'] == 'passed'


def choose(free, pending, now, minimum=MINIMUM_MIB):
    """free：各 GPU 的空闲显存；pending：[gpu, 派出时刻]。返回 (选中的 GPU 或 None, 仍然有效的 pending)。"""
    pending = [p for p in pending if now - p[1] < SETTLE_S]
    effective = {g: f - RESERVE_MIB * sum(p[0] == g for p in pending) for g, f in free.items()}
    gpu = max(effective, key=effective.get)
    return (gpu if effective[gpu] >= minimum else None), pending


def allowed(free, spec):
    """spec 是环境变量 C1A2_GPUS 的值（如 '1' 或 '1,3'）；给出时只在这些 GPU 里挑，用于别的任务占着其余 GPU 的时候。"""
    return free if not spec else {g: f for g, f in free.items() if g in {int(x) for x in spec.split(',')}}


def pick_gpu(minimum=MINIMUM_MIB, wait_minutes=720):
    for _ in range(wait_minutes * 2):
        with open(LOCK, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try: pending = json.load(open(PENDING))
            except (FileNotFoundError, ValueError): pending = []
            free = {int(a): int(b) for a, b in (line.split(',') for line in subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).strip().splitlines())}
            gpu, pending = choose(allowed(free, os.environ.get('C1A2_GPUS')), pending, time.time(), minimum)
            if gpu is not None: pending.append([gpu, time.time()])
            json.dump(pending, open(PENDING, 'w'))
        if gpu is not None: return gpu
        time.sleep(30)
    raise RuntimeError('no GPU became free')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--only', choices=['train', 'evaluate'], help='只跑训练或只跑评估')
    parser.add_argument('--minimum-mib', type=int, default=MINIMUM_MIB, help='开始一个任务所需的空闲显存（扣掉刚派出的任务之后）')
    parser.add_argument('--skip-retention', action='store_true', help='只做目标评估')
    args = parser.parse_args()
    assert SOURCE or CANDIDATE or LEAD or RATE or REFLEX or passed('smoke'), 'run the smoke job first'  # 参数迁移和候选评估只有评估任务，用的是已经跑通的入口
    schedule = [j for j in jobs() if j['kind'] != 'smoke' and not j.get('script') and not (args.skip_retention and j['kind'] == 'retention')]
    stages = [[j for j in schedule if j['kind'] == 'train'], [j for j in schedule if j['kind'] in ('target', 'retention')]]
    if args.only == 'train': stages = stages[:1]
    if args.only == 'evaluate': stages = stages[1:]
    failed = []

    def run(job):
        name = job['name']
        if passed(name) or failed: return
        gpu = pick_gpu(args.minimum_mib)
        print('POOL', TARGET, name, 'gpu', gpu, time.strftime('%H:%M'), flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/tools/reference_online_c1a2.py'), '--job', name, '--gpu', str(gpu)],
                       cwd=ROOT, stdout=subprocess.DEVNULL, env=dict(os.environ, C1A2_MIN_FREE_MIB=str(args.minimum_mib - 500)))
        if not passed(name):
            failed.append(name); print('POOL FAILED', TARGET, name, flush=True)
        else:
            print('POOL PASSED', TARGET, name, time.strftime('%H:%M'), flush=True)

    for stage in stages:
        with ThreadPoolExecutor(args.workers) as pool: list(pool.map(run, stage))
        if failed: return 1
    print('POOL COMPLETE', TARGET, flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
