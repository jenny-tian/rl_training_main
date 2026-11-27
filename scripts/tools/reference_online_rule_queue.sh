#!/bin/bash
# 规则对照探路：从共享的队列文件里逐个取目标，在给定的 GPU 上做评估。可以同时开几路。
# 用法：reference_online_rule_queue.sh <队列文件> <GPU>
q=$1; g=$2
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
while :; do
  t=$( (flock 9; head -n 1 "$q"; sed -i 1d "$q") 9> "$q.lock")
  [ -z "$t" ] && break
  echo "RULE $t gpu $g $(date +%H:%M)"
  C1A2_GPUS=$g C1A2_VARIANT=r2 C1A2_TARGET=$t C1A2_RULE=directed PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 2 --minimum-mib 7000 > /dev/null 2>&1
  echo "RULE DONE $t status=$? $(date +%H:%M)"
done
echo "RULE QUEUE EMPTY gpu $g $(date +%H:%M)"
