#!/usr/bin/env bash
set -euo pipefail

source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

WORKTREE=/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
BASE_CHECKPOINT=$WORKTREE/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt
BASE_MANIFEST=$WORKTREE/artifacts/online_validation_20260805/base_manifest.json
FLAT_REPLAY=$WORKTREE/artifacts/online_validation_20260805/flat_replay_model1998_formal/flat_replay.pt
RESIDUAL_CHECKPOINT=$WORKTREE/artifacts/online_validation_20260806/residual_step_pretrain_and_measure_v4_terminal_observation/residual_pretrain_59d_flat_anchor_20000/latest.pt
OUTPUT=$WORKTREE/artifacts/online_validation_20260809/step2cm_fixed03_same_seed_v5
LOG=$OUTPUT/run.log

mkdir -p "$OUTPUT"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$WORKTREE:$WORKTREE/source/rl_training:/home/tjx/IsaacLab/source"
unset CUDA_VISIBLE_DEVICES
unset DISPLAY
export TERM=xterm-256color

python3 -u "$WORKTREE/scripts/reinforcement_learning/online_adaptation/measure_step_completion.py" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --base-manifest "$BASE_MANIFEST" \
    --initial-residual-checkpoint "$RESIDUAL_CHECKPOINT" \
    --flat-replay "$FLAT_REPLAY" \
    --output-dir "$OUTPUT" \
    --terrain-stage step_2cm \
    --max-episodes 500 \
    --max-steps-per-episode 2000 \
    --forward-speed 0.3 \
    --residual-gain 0.10 \
    --adapting-residual-gain 0.10 \
    --post-step-residual-gain 0.10 \
    --comparison-seed 101 \
    --device cuda:1 \
    --headless \
    --video
