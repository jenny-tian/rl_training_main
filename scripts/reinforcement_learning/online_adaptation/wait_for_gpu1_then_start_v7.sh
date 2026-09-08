#!/usr/bin/env bash
set -euo pipefail

BLOCKING_PID=3108809
WORKTREE=/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
LAUNCHER=$WORKTREE/scripts/reinforcement_learning/online_adaptation/start_step2cm_fixed_v7.sh

while ps -p "$BLOCKING_PID" >/dev/null 2>&1; do
    sleep 30
done

exec /bin/bash "$LAUNCHER"
