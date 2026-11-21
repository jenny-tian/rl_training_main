#!/bin/bash
# 加快学习 B（C1-b-W）一个目标的全流程：冒烟、训练、评估。已通过的任务跳过。
# 用法：reference_online_window_chain.sh <目标> <冒烟和训练用的 GPU>
t=$1; g=$2
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W || exit 2
export C1A2_VARIANT=w2 C1A2_TARGET=$t PYTHONPATH=$W/source/rl_training C1A2_MIN_FREE_MIB=6500
d=$W/logs/reference_six/online_c1b_window_20261121/$t
mkdir -p $d
for job in smoke train; do
  if ! grep -q '"status": "passed"' $d/$job/report.json 2>/dev/null; then
    echo "CHAIN $t $job gpu $g $(date +%H:%M)"
    $PY scripts/tools/reference_online_c1a2.py --job $job --gpu $g > $d/$job.out 2>&1
    grep -q '"status": "passed"' $d/$job/report.json 2>/dev/null || { echo "CHAIN FAILED $t $job"; exit 1; }
  fi
done
echo "CHAIN $t evaluate $(date +%H:%M)"
cd $W/scripts/tools && $PY reference_online_c1a2_pool.py --only evaluate --workers 2 --minimum-mib 7000
echo "CHAIN DONE $t status=$? $(date +%H:%M)"
