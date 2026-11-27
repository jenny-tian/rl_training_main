#!/bin/bash
# 上阶最后一级的脚本化检查：从共享的队列文件里逐条取 <组> [confirm]，在给定的 GPU 上做评估。可以同时开几路。
# 用法：reference_online_top_queue.sh <队列文件> <GPU>
q=$1; g=$2
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
while :; do
  line=$( (flock 9; head -n 1 "$q"; sed -i 1d "$q") 9> "$q.lock")
  [ -z "$line" ] && break
  read -r group confirm <<< "$line"
  echo "TOP $group $confirm gpu $g $(date +%H:%M)"
  env ${confirm:+C1A2_TOP_CONFIRM=1} C1A2_GPUS=$g C1A2_VARIANT=r2 C1A2_TARGET=stairs_up_07 C1A2_TOP=$group PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 2 --minimum-mib 7000 > /dev/null 2>&1
  echo "TOP DONE $group $confirm status=$? $(date +%H:%M)"
done
echo "TOP QUEUE EMPTY gpu $g $(date +%H:%M)"
