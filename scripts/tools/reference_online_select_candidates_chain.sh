#!/bin/bash
# 加快学习 D 的 D0：9 组候选参数各自直接用到一个新地形上评估，不学习。已通过的任务跳过。
# 方案见 docs/lite3-online-select-20261123/PLAN.md。用法：reference_online_select_candidates_chain.sh <目标> [传给评估调度的其他参数]
t=$1; shift
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
for c in 01m 02m 12m 01u 01d 10u 10d 20u 20d; do
  echo "CANDIDATE $t $c $(date +%H:%M)"
  C1A2_VARIANT=s2n C1A2_TARGET=$t C1A2_CANDIDATE=$c PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --minimum-mib 7000 "$@"
  echo "CANDIDATE DONE $t $c status=$? $(date +%H:%M)"
done
echo "CANDIDATE ALL DONE $t $(date +%H:%M)"
