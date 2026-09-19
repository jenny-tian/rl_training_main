#!/usr/bin/env bash
set -uo pipefail

source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

WORKTREE=/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
# Strict completion qualification is a new protocol boundary.  Keep its
# outputs separate from the legacy sealed20 run, whose summaries predate the
# in-lane obstacle-interaction evidence and must not be mixed into validation.
OUTPUT_ROOT="$WORKTREE/artifacts/semester_2026/09/rocky_road_online_sealed20_strict_v1"
BASE_CHECKPOINT="$WORKTREE/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt"
RESIDUAL_CHECKPOINT="$WORKTREE/artifacts/semester_2026/09/rocky_road_online/v12_fix_validation_20260910/initializer/residual_sac_initial.pt"
FLAT_REPLAY="$WORKTREE/artifacts/online_validation_20260805/flat_replay_model1998_formal/flat_replay.pt"
NOVELTY_BASELINE="$WORKTREE/artifacts/online_validation_20260811/step4cm_fixed03_same_seed_v19_qcal_detmean_gpu2/novelty_baseline.json"
NOVELTY_SHA256=397ebdc2da4be4c2e284b5b837644a2574ec4f5e80e2f3ac1bcf4a3be399a3ee

export PYTHONPATH="$WORKTREE:$WORKTREE/source/rl_training:/home/tjx/IsaacLab/source"
unset CUDA_VISIBLE_DEVICES
unset DISPLAY
export TERM=xterm-256color

mkdir -p "$OUTPUT_ROOT"

for required_file in \
    "$BASE_CHECKPOINT" \
    "$RESIDUAL_CHECKPOINT" \
    "$FLAT_REPLAY" \
    "$NOVELTY_BASELINE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "missing required input: $required_file" >&2
        exit 2
    fi
done

run_seed() {
    local gpu=$1
    local terrain_seed=$2
    local run_dir="$OUTPUT_ROOT/online_amp4cm_sealed_seed${terrain_seed}_v1"
    local runtime_log="$run_dir/runtime.log"

    if [[ -s "$run_dir/episode_summary.json" && -s "$run_dir/videos/rl-video-step-0.mp4" ]]; then
        echo "[gpu${gpu}] seed ${terrain_seed}: complete outputs already exist; skipping"
        return 0
    fi

    mkdir -p "$run_dir"
    echo "[gpu${gpu}] seed ${terrain_seed}: starting"
    if python -u "$WORKTREE/scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --residual-checkpoint "$RESIDUAL_CHECKPOINT" \
        --mode online \
        --task RockyRoad-Deeprobotics-Lite3-v0 \
        --seed 7 \
        --forward-speed 0.3 \
        --max-steps 2000 \
        --run-dir "$run_dir" \
        --terrain-stage rough_4cm \
        --terrain-split sealed \
        --terrain-seed "$terrain_seed" \
        --terrain-rough-amplitude 0.04 \
        --terrain-feature-source proprio \
        --residual-frame-dim 49 \
        --flat-replay "$FLAT_REPLAY" \
        --novelty-baseline-in "$NOVELTY_BASELINE" \
        --novelty-baseline-sha256 "$NOVELTY_SHA256" \
        --experiment-checkpoint-out "$run_dir/experiment.pt" \
        --actor-updates-per-second 0.5 \
        --control-timeout-ms 50 \
        --target-q-min -500 \
        --target-q-max 300 \
        --deterministic-actor-weight 0.25 \
        --max-policy-residual-mse 0.00001 \
        --residual-gain 0.05 \
        --adapting-residual-gain 0.10 \
        --post-step-residual-gain 0.10 \
        --require-track-completion \
        --video \
        --video-length 2000 \
        --enable_cameras \
        --device "cuda:${gpu}" \
        --headless \
        --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1" \
        >"$runtime_log" 2>&1; then
        echo "[gpu${gpu}] seed ${terrain_seed}: finished"
        return 0
    else
        local status=$?
        echo "[gpu${gpu}] seed ${terrain_seed}: worker exited ${status}; see ${runtime_log}" >&2
        return "$status"
    fi
}

run_lane() {
    local gpu=$1
    shift
    local lane_status=0
    local terrain_seed
    for terrain_seed in "$@"; do
        if ! run_seed "$gpu" "$terrain_seed"; then
            lane_status=1
        fi
    done
    return "$lane_status"
}

run_lane 0 1009 1019 1031 1039 1051 1063 1087 1093 1103 1117 & lane0=$!
run_lane 1 1013 1021 1033 1049 1061 1069 1091 1097 1109 1123 & lane1=$!

trap 'kill "$lane0" "$lane1" 2>/dev/null || true' INT TERM

batch_status=0
for lane_pid in "$lane0" "$lane1"; do
    if ! wait "$lane_pid"; then
        batch_status=1
    fi
done

complete_summaries=$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name episode_summary.json -size +0c | wc -l)
complete_videos=$(find "$OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -name 'rl-video-step-0.mp4' -size +0c | wc -l)
echo "sealed20 complete: summaries=${complete_summaries}/20 videos=${complete_videos}/20"
exit "$batch_status"
