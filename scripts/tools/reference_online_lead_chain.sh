#!/bin/bash
# 提前起抬的脚本化检查：一个目标上依次评估几条规则，不学习。已通过的任务跳过。
# 方案见 docs/lite3-online-lead-20261123/PLAN.md。用法：reference_online_lead_chain.sh <目标> <库编号> <规则...>
t=$1; e=$2; shift 2
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
for r in "$@"; do
  echo "LEAD $t $r:$e $(date +%H:%M)"
  C1A2_VARIANT=s2n C1A2_TARGET=$t C1A2_LEAD=$r:$e PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 4 --minimum-mib 7000
  echo "LEAD DONE $t $r:$e status=$? $(date +%H:%M)"
done
echo "LEAD ALL DONE $t $(date +%H:%M)"
