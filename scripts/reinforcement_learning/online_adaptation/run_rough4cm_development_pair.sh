#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 GPU_INDEX TERRAIN_SEED [TERRAIN_SEED ...]" >&2
    exit 2
fi

GPU_INDEX=$1
shift
TERRAIN_SEEDS=("$@")

source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

WORKTREE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKTREE/artifacts/semester_2026/09/paired_rough4cm_development_v6}
BASE_CHECKPOINT=$WORKTREE/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt
RESIDUAL_CHECKPOINT=$WORKTREE/artifacts/semester_2026/09/rocky_road_online/v12_fix_validation_20260910/initializer/residual_sac_initial.pt
FLAT_REPLAY=$WORKTREE/artifacts/online_validation_20260805/flat_replay_model1998_formal/flat_replay.pt
NOVELTY_BASELINE=$WORKTREE/artifacts/online_validation_20260811/step4cm_fixed03_same_seed_v19_qcal_detmean_gpu2/novelty_baseline.json
NOVELTY_BASELINE_SHA256=397ebdc2da4be4c2e284b5b837644a2574ec4f5e80e2f3ac1bcf4a3be399a3ee

USD_LIBS=$(find "$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache" \
    -maxdepth 1 -type d -name 'omni.usd.libs-*' | head -1)
export PYTHONPATH="$WORKTREE:$WORKTREE/source/rl_training:/home/tjx/IsaacLab/source:$USD_LIBS"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$USD_LIBS/bin:${LD_LIBRARY_PATH:-}"
export TERM=xterm-256color
unset CUDA_VISIBLE_DEVICES
unset DISPLAY
PROTOCOL_SHA256=$(python -c 'from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol; print(default_flat_online_protocol().sha256())')

for required in "$BASE_CHECKPOINT" "$RESIDUAL_CHECKPOINT" "$FLAT_REPLAY" "$NOVELTY_BASELINE"; do
    if [[ ! -f "$required" ]]; then
        echo "missing input: $required" >&2
        exit 2
    fi
done

run_base() {
    local seed=$1
    local run_dir=$2
    if [[ -s "$run_dir/summary.json" ]]; then
        echo "[seed $seed] frozen PPO already complete"
        return
    fi
    mkdir -p "$run_dir"
    echo "[seed $seed] starting frozen PPO on cuda:$GPU_INDEX"
    python -u "$WORKTREE/scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --mode base \
        --task RockyRoad-Deeprobotics-Lite3-v0 \
        --seed 7 \
        --forward-speed 0.3 \
        --max-steps 2000 \
        --run-dir "$run_dir" \
        --terrain-stage rough_4cm \
        --terrain-seed "$seed" \
        --terrain-split development \
        --protocol-sha256 "$PROTOCOL_SHA256" \
        --require-track-completion \
        --terrain-feature-source proprio \
        --headless \
        --device "cuda:$GPU_INDEX" \
        >"$run_dir/runtime.log" 2>&1
    echo "[seed $seed] frozen PPO complete"
}

run_lora() {
    local seed=$1
    local run_dir=$2
    if [[ -s "$run_dir/latest.pt" && -s "$run_dir/summary.json" ]]; then
        if ! jq -e '.learning_contract == "fixed_command_qualification_boundary_v5"' \
            "$run_dir/summary.json" >/dev/null; then
            echo "incompatible LoRA result; use a new OUTPUT_ROOT: $run_dir" >&2
            exit 2
        fi
        echo "[seed $seed] LoRA-PPO already complete"
        return
    fi
    mkdir -p "$run_dir"
    echo "[seed $seed] starting LoRA-PPO on cuda:$GPU_INDEX"
    python -u "$WORKTREE/scripts/reinforcement_learning/online_adaptation/train_lora_ppo.py" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --task RockyRoad-Deeprobotics-Lite3-v0 \
        --terrain-stage rough_4cm \
        --terrain-seed "$seed" \
        --protocol-sha256 "$PROTOCOL_SHA256" \
        --require-track-completion \
        --seed 7 \
        --forward-speed 0.3 \
        --max-steps 2000 \
        --rollout-steps 256 \
        --output "$run_dir" \
        --rank 2 \
        --action-std 0.01 \
        --stop-after-termination \
        --retain-terminal-learning \
        --residual-rate-limit 0.03 \
        --actor-lr 0.00005 \
        --critic-lr 0.00005 \
        --nonfoot-contact-rollback-force 20 \
        --ppo-epochs 2 \
        --minibatch-size 64 \
        --clip-eps 0.1 \
        --target-kl 0.01 \
        --gamma 0.99 \
        --gae-lambda 0.95 \
        --entropy-coef 0 \
        --value-coef 0.5 \
        --max-grad-norm 1 \
        --headless \
        --device "cuda:$GPU_INDEX" \
        >"$run_dir/runtime.log" 2>&1
    echo "[seed $seed] LoRA-PPO complete"
}

run_sac() {
    local seed=$1
    local run_dir=$2
    if [[ -s "$run_dir/summary.json" ]]; then
        echo "[seed $seed] Residual SAC already complete"
        return
    fi
    mkdir -p "$run_dir"
    echo "[seed $seed] starting Residual SAC on cuda:$GPU_INDEX"
    python -u "$WORKTREE/scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --residual-checkpoint "$RESIDUAL_CHECKPOINT" \
        --mode online \
        --task RockyRoad-Deeprobotics-Lite3-v0 \
        --seed 7 \
        --forward-speed 0.3 \
        --max-steps 2000 \
        --run-dir "$run_dir" \
        --min-replay-size 256 \
        --replay-capacity 10000 \
        --updates-per-second 5 \
        --actor-updates-per-second 0.5 \
        --update-schedule control-step \
        --critic-updates-per-transition 0.5 \
        --actor-updates-per-transition 0.005 \
        --novelty-calibration-steps 256 \
        --novelty-enter-threshold 2.5 \
        --novelty-exit-threshold 1 \
        --novelty-enter-count 10 \
        --novelty-enter-window 20 \
        --novelty-recovery-steps 100 \
        --adaptation-min-replay-size 128 \
        --stalled-actor-replay-stride 16 \
        --rollback-window 50 \
        --rollback-patience 4 \
        --rollback-degradation-ratio 0.1 \
        --stable-improvement-ratio 0.05 \
        --adaptation-cooldown-steps 100 \
        --control-timeout-ms 50 \
        --adapting-stochastic-control \
        --max-critic-loss 800 \
        --max-mean-td-error 10 \
        --max-critic-grad-norm 3000 \
        --target-q-min -500 \
        --target-q-max 300 \
        --initial-alpha 1 \
        --deterministic-actor-weight 0.25 \
        --terrain-stage rough_4cm \
        --terrain-seed "$seed" \
        --terrain-split development \
        --protocol-sha256 "$PROTOCOL_SHA256" \
        --novelty-baseline-in "$NOVELTY_BASELINE" \
        --novelty-baseline-sha256 "$NOVELTY_BASELINE_SHA256" \
        --require-track-completion \
        --experiment-checkpoint-out "$run_dir/experiment.pt" \
        --flat-replay "$FLAT_REPLAY" \
        --flat-anchor-weight 0.5 \
        --max-policy-residual-mse 0.00001 \
        --terrain-feature-source proprio \
        --residual-frame-dim 49 \
        --terrain-min-valid-fraction 0.9 \
        --n-step 1 \
        --target-semantics timeout-aware \
        --critic-replay-mode dual \
        --flat-fraction-initial 0.75 \
        --flat-fraction-ready 0.25 \
        --flat-critic-warmup-updates 1000 \
        --residual-gain 0.05 \
        --adapting-residual-gain 0.1 \
        --post-step-residual-gain 0.1 \
        --post-step-recovery-steps 25 \
        --post-step-stability-weight 3 \
        --post-step-progress-weight 2 \
        --stall-penalty-weight 2 \
        --completion-bonus 5 \
        --rough-swing-lift-weight 0 \
        --rough-swing-lift-target 0.045 \
        --pre-step-clearance-weight 4 \
        --pre-step-clearance-target 0.045 \
        --pre-step-foot-forward-weight 2 \
        --pre-step-foot-forward-target 0.04 \
        --pre-step-approach-distance 0.5 \
        --residual-rate-limit 0.03 \
        --headless \
        --device "cuda:$GPU_INDEX" \
        >"$run_dir/runtime.log" 2>&1
    echo "[seed $seed] Residual SAC complete"
}

mkdir -p "$OUTPUT_ROOT"
for seed in "${TERRAIN_SEEDS[@]}"; do
    seed_root=$OUTPUT_ROOT/seed-$seed
    run_base "$seed" "$seed_root/base_frozen"
    run_lora "$seed" "$seed_root/lora_ppo"
    run_sac "$seed" "$seed_root/residual_sac"
done
