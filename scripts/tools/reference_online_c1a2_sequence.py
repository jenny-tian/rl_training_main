"""按顺序跑完一个目标的训练、目标评估和保持评估；任何一步失败就停。已通过的任务跳过。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from reference_online_c1a2 import OUTPUT, TARGET, jobs
from reference_stairs_eval import ROOT


def passed(name):
    path = OUTPUT / name / 'report.json'
    return path.exists() and json.loads(path.read_text())['status'] == 'passed'


def free_gpu(minimum_mib=10240, wait_minutes=720):
    """空闲显存最多且不少于 minimum_mib 的 GPU；连续两次检查都满足才返回，否则每分钟再看一次。"""
    def best():
        rows = [[int(x) for x in line.split(',')] for line in subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).strip().splitlines()]
        index, free = max(rows, key=lambda row: row[1])
        return index if free >= minimum_mib else None
    for _ in range(wait_minutes):
        first = best()
        if first is not None:
            time.sleep(30)
            if best() == first: return first
        time.sleep(60)
    raise RuntimeError('no GPU became free')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='0 到 3，或 auto：每个任务开始前挑当时空闲显存最多的一块')
    args = parser.parse_args()
    assert passed('smoke'), 'run the smoke job first'
    for job in jobs()[1:]:
        name = job['name']
        if passed(name) or job.get('script'): continue
        gpu = free_gpu() if args.gpu == 'auto' else int(args.gpu)
        print('SEQUENCE', TARGET, name, 'gpu', gpu, time.strftime('%H:%M'), flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/tools/reference_online_c1a2.py'), '--job', name,
                        '--gpu', str(gpu)], cwd=ROOT, stdout=subprocess.DEVNULL)
        if not passed(name):
            print('SEQUENCE FAILED', TARGET, name, flush=True)
            return 1
    print('SEQUENCE COMPLETE', TARGET, flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
