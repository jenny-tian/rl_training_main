#!/usr/bin/env bash
set -euo pipefail

source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation

export PYTHONPATH=/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation:/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/source/rl_training:/home/tjx/IsaacLab/source
unset CUDA_VISIBLE_DEVICES
unset DISPLAY
export TERM=xterm-256color

exec python scripts/reinforcement_learning/online_adaptation/measure_step_completion.py \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt \
  --base-manifest artifacts/online_validation_20260805/base_manifest.json \
  --initial-residual-checkpoint artifacts/online_validation_20260811/initializers/v10_step2cm_safe_updater.pt \
  --flat-replay artifacts/online_validation_20260805/flat_replay_model1998_formal/flat_replay.pt \
  --output-dir artifacts/online_validation_20260812/step4cm_fixed06_v24_fall_softreset_gpu2 \
  --terrain-stage step_4cm \
  --max-episodes 500 \
  --max-steps-per-episode 2000 \
  --forward-speed 0.3 \
  --residual-gain 0.05 \
  --adapting-residual-gain 0.10 \
  --post-step-residual-gain 0.10 \
  --post-step-stability-weight 1.0 \
  --post-step-progress-weight 2.0 \
  --stall-penalty-weight 2.0 \
  --completion-bonus 5.0 \
  --pre-step-clearance-weight 2.0 \
  --pre-step-clearance-target 0.060 \
  --pre-step-approach-distance 0.5 \
  --actor-updates-per-second 0.5 \
  --adaptation-min-replay-size 128 \
  --stalled-actor-replay-stride 16 \
  --control-timeout-ms 50 \
  --critic-huber-delta 1.0 \
  --max-critic-grad-norm 3000 \
  --max-policy-residual-mse 0.01 \
  --target-q-min -500 \
  --target-q-max 300 \
  --deterministic-actor-weight 0.25 \
  --comparison-seed 101 \
  --device cuda:2 \
  --headless \
  --video
