#!/bin/bash
# 扩充参数库的 F0：给定的几组库参数各自直接用到一个更难的新地形上评估，不学习。已通过的任务跳过。
# 方案见 docs/lite3-online-library2-20261123/PLAN.md。用法：reference_online_library2_transfer_chain.sh <目标> <来源...>
t=$1; shift
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
for s in "$@"; do
  echo "TRANSFER $t from $s $(date +%H:%M)"
  C1A2_VARIANT=s2n C1A2_TARGET=$t C1A2_SOURCE=$s PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 4 --minimum-mib 7000
  echo "TRANSFER DONE $t from $s status=$? $(date +%H:%M)"
done
echo "TRANSFER ALL DONE $t $(date +%H:%M)"
