#!/bin/bash
# 每步变化上限的仿真扫描：一个目标上依次评估几个上限，不学习。已通过的任务跳过。
# 方案见 docs/lite3-online-rate-20261123/PLAN.md。用法：reference_online_rate_chain.sh <目标> <库编号> <上限...>
t=$1; e=$2; shift 2
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
for r in "$@"; do
  echo "RATE $t $r:$e $(date +%H:%M)"
  C1A2_VARIANT=s2n C1A2_TARGET=$t C1A2_RATE=$r:$e PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 4 --minimum-mib 7000
  echo "RATE DONE $t $r:$e status=$? $(date +%H:%M)"
done
echo "RATE ALL DONE $t $(date +%H:%M)"
