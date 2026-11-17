"""按顺序跑完一个目标的训练、目标评估和保持评估；任何一步失败就停。已通过的任务跳过。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from reference_online_c1a2 import OUTPUT, TARGET, jobs
from reference_stairs_eval import ROOT


def passed(name):
    path = OUTPUT / name / 'report.json'
    return path.exists() and json.loads(path.read_text())['status'] == 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    assert passed('smoke'), 'run the smoke job first'
    for job in jobs()[1:]:
        name = job['name']
        if passed(name): continue
        print('SEQUENCE', TARGET, name, flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/tools/reference_online_c1a2.py'), '--job', name,
                        '--gpu', str(args.gpu)], cwd=ROOT, stdout=subprocess.DEVNULL)
        if not passed(name):
            print('SEQUENCE FAILED', TARGET, name, flush=True)
            return 1
    print('SEQUENCE COMPLETE', TARGET, flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
