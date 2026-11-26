#!/bin/bash
# 从一个共享的队列文件里逐条取出 <变体> <目标>，在给定的一块 GPU 上依次跑完整条运行链（冒烟、训练、评估）。
# 可以同时开几个，各取各的；别的任务占着其余 GPU 时只用空着的那块，之后有 GPU 空出来可以再开新的来取剩下的。
# 用法：reference_online_finetune_queue.sh <队列文件> <GPU> [每条链开始前要求的空闲显存 MiB，默认 9000]
q=$1; g=$2; need=${3:-9000}
W=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
export C1A2_GPUS=$g
while :; do
  line=$( (flock 9; head -n 1 "$q"; sed -i 1d "$q") 9> "$q.lock")
  [ -z "$line" ] && break
  read -r v t <<< "$line"
  while [ "$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$need" ]; do sleep 30; done
  d=$(cd $W/scripts/tools && C1A2_VARIANT=$v C1A2_TARGET=$t PYTHONPATH=$W/source/rl_training $PY -c "from reference_online_c1a2 import OUTPUT, UNFILTERED, SKIP_RETENTION; print(OUTPUT, '--skip-retention' if UNFILTERED or SKIP_RETENTION else '')")
  read -r d extra <<< "$d"
  mkdir -p $d
  echo "QUEUE $v $t gpu $g $(date +%H:%M)"
  bash $W/scripts/tools/reference_online_window_chain.sh $t $g $v --workers 3 $extra >> $d/sequence.log 2>&1
  echo "QUEUE DONE $v $t $(tail -n 1 $d/sequence.log) $(date +%H:%M)"
done
echo "QUEUE EMPTY gpu $g $(date +%H:%M)"
