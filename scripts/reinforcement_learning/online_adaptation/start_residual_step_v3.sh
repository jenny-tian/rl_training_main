#!/usr/bin/env bash
set -euo pipefail

source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

WORKTREE=/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
BASE_CHECKPOINT=$WORKTREE/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt
BASE_MANIFEST=$WORKTREE/artifacts/online_validation_20260805/base_manifest.json
FLAT_REPLAY=$WORKTREE/artifacts/online_validation_20260805/flat_replay_model1998_formal/flat_replay.pt
RUN_ROOT=$WORKTREE/artifacts/online_validation_20260806/residual_step_pretrain_and_measure_v3_timeout_terminal
PRETRAIN_DIR=$RUN_ROOT/residual_pretrain_59d_flat_anchor_20000
MEASURE_DIR=$RUN_ROOT/step2cm_first_completion
LOG=$RUN_ROOT/run.log

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$WORKTREE:$WORKTREE/source/rl_training:/home/tjx/IsaacLab/source"
unset CUDA_VISIBLE_DEVICES
unset DISPLAY
export TERM=xterm-256color

echo "[lite3-v3] GPU1 timeout-safe 59D residual pretrain + step completion measurement"
echo "[lite3-v3] output: $RUN_ROOT"

if [[ ! -f "$PRETRAIN_DIR/latest.pt" ]]; then
    mkdir -p "$PRETRAIN_DIR"
    echo "[lite3-v3] starting timeout-safe residual pretraining from frozen base"
    python3 -u "$WORKTREE/scripts/reinforcement_learning/online_adaptation/train_residual_sac.py" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --task FlatOnline-Deeprobotics-Lite3-v0 \
        --num-envs 1024 \
        --seed 7 \
        --max-steps 20000 \
        --save-interval 1000 \
        --log-dir "$PRETRAIN_DIR" \
        --flat-replay "$FLAT_REPLAY" \
        --forward-speed 0.3 \
        --residual-gain 0.1 \
        --n-step 3 \
        --flat-anchor-weight 0.5 \
        --flat-fraction-initial 0.75 \
        --flat-fraction-ready 0.25 \
        --flat-critic-warmup-updates 1000 \
        --terrain-step-height 0.0 \
        --terrain-feature-source geometry \
        --device cuda:1 \
        --headless
else
    echo "[lite3-v3] compatible residual checkpoint already exists; skipping pretraining"
fi

FRAME_DIM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["agent"]["frame_dim"])' "$PRETRAIN_DIR/config.json")
if [[ "$FRAME_DIM" != "59" ]]; then
    echo "[lite3-v3] ERROR: pretraining frame_dim=$FRAME_DIM, expected 59"
    exit 2
fi

if [[ ! -f "$MEASURE_DIR/summary.json" ]]; then
    echo "[lite3-v3] measuring first step completion without a 100-episode acceptance deadline"
    python3 -u "$WORKTREE/scripts/reinforcement_learning/online_adaptation/measure_step_completion.py" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --base-manifest "$BASE_MANIFEST" \
        --initial-residual-checkpoint "$PRETRAIN_DIR/latest.pt" \
        --flat-replay "$FLAT_REPLAY" \
        --output-dir "$MEASURE_DIR" \
        --terrain-stage step_2cm \
        --max-episodes 500 \
        --max-steps-per-episode 2000 \
        --residual-gain 0.10 \
        --device cuda:1 \
        --headless \
        --video
else
    echo "[lite3-v3] measurement already complete"
fi

echo "[lite3-v3] finished; summary: $MEASURE_DIR/summary.json"
