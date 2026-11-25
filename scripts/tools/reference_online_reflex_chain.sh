#!/bin/bash
# 抬腿反射的对照评估：依次在几个目标上评估一条规则，不学习。已通过的任务跳过。
# 方案见 docs/lite3-online-baselines-20261125/PLAN.md。用法：reference_online_reflex_chain.sh <规则> <目标...>
r=$1; shift
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
for t in "$@"; do
  echo "REFLEX $r $t $(date +%H:%M)"
  C1A2_VARIANT=s2n C1A2_TARGET=$t C1A2_REFLEX=$r PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 3 --minimum-mib 7000
  echo "REFLEX DONE $r $t status=$? $(date +%H:%M)"
done
echo "REFLEX ALL DONE $r $(date +%H:%M)"
