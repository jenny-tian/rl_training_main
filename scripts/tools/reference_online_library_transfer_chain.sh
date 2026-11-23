#!/bin/bash
# 加快学习 C 的 C0：参数库里三组学到的参数各自直接用到两个新地形上评估，不学习。已通过的任务跳过。
# 方案见 docs/lite3-online-library-20261122/PLAN.md。用法：reference_online_library_transfer_chain.sh
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
for t in stairs_up_07 boxes_07; do
  for s in s2n:stairs_up_06:60000:pooled s2n:stairs_up_065:60000:pooled bs:boxes_06:40000:pooled; do
    echo "TRANSFER $t from $s $(date +%H:%M)"
    C1A2_VARIANT=s2n C1A2_TARGET=$t C1A2_SOURCE=$s PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 2 --minimum-mib 7000
    echo "TRANSFER DONE $t from $s status=$? $(date +%H:%M)"
  done
done
echo "TRANSFER ALL DONE $(date +%H:%M)"
