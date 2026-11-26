#!/bin/bash
# 不学习的规则对照（docs/lite3-online-adapter2-20261126/PLAN.md 第 12 节）：依次在几个目标上做评估，已通过的任务跳过。
# 用法：reference_online_rule_chain.sh <目标...>      可用的 GPU 由环境变量 C1A2_GPUS 限定（如 0,3）
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
cd $W/scripts/tools || exit 2
for t in "$@"; do
  echo "RULE $t $(date +%H:%M)"
  C1A2_VARIANT=r2 C1A2_TARGET=$t C1A2_RULE=directed PYTHONPATH=$W/source/rl_training $PY reference_online_c1a2_pool.py --only evaluate --workers 4 --minimum-mib 7000
  echo "RULE DONE $t status=$? $(date +%H:%M)"
done
echo "RULE ALL DONE $(date +%H:%M)"
