# Lite3 Motion Prior Workflow

This is a Motion-Priors-style flat PPO enhancement for `Flat-Deeprobotics-Lite3-v0`.
It is intentionally not a claim of reproducing *Motion Priors Reimagined* end to end.

## What the paper does

The paper uses four stages:

1. Retarget animal motion capture to the target robot with inverse kinematics from pelvis/neck/feet.
2. Train an FLD temporal encoder offline, then train a flat-ground low-level PPO policy from proprioception plus the latent motion embedding. The implementation uses ground-truth reference sequences during training instead of rolling out the FLD decoder.
3. Freeze that low-level prior and train a privileged high-level teacher that emits a latent command and 12 joint residuals. A squared residual penalty controls how far rough-terrain behavior can move from the prior.
4. Distill the teacher to a noisy-observation student for sim-to-real. Foot elevation scans and downsampled LiDAR are used in this later task-training stage, not as a requirement for flat low-level pre-training.

## What this repository adds first

The initial flat PPO slice keeps the current proprioceptive actor and privileged critic. An external reference NPZ is opt-in and contributes reference rewards for joint position, foot position, contact phase, swing clearance, bilateral symmetry, and joint smoothness. Compare three explicit runs:

- `prior_off`: formal flat-only baseline, no reference path.
- `prior_on`: fixed reference weight.
- `prior_decay`: linear reference-weight decay over `motion_prior_decay_steps`.

The reference is not an action label. The policy still learns through PPO and may trade off the prior against velocity tracking, stability, contact, and torque terms.

## Data and retargeting

The canonical output is a provenance-bearing compressed NPZ containing `time`, 12 Lite3 joint positions in `LITE3_POLICY_JOINT_NAMES` order, body-frame positions for four feet, binary contact, optional velocity command, and JSON metadata. `retarget_motion.py` accepts source Go2-style joint arrays, body-frame feet, AI4Animation dog skeleton positions, or raw BVH clips and applies explicit mapping, sign/scale/offset normalization, and analytic Lite3 IK. The downloaded `MotionCapture.zip` is the official AI4Animation/SIGGRAPH 2018 archive (CC BY-NC 4.0); preserve that attribution in generated metadata and do not treat raw BVH as Lite3 telemetry.

AI4Animation dog clips are usable source motion, but their `*_joint_pos.txt` files are dog skeleton positions, not Lite3 or Go2 joint angles. They require a skeleton-aware preprocessing step to extract pelvis/neck/feet and then IK retargeting. The public source is CC BY-NC 4.0; keep it outside this repository and preserve attribution/license metadata. The `nomagiclab/kine2go-pipeline` code is BSD-3-Clause and provides a useful Go2 retargeting reference, but its bundled dog motion is still source mocap rather than Go2 real-robot telemetry.

No external trajectory is bundled in the formal flat baseline. A real Go2 deployment log can be used after enabling telemetry recording, checking joint order/units/time synchronization, deriving foot/contact signals, and recording its provenance.

## Example

```bash
PYTHONPATH=source/rl_training \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/reinforcement_learning/motion_prior/retarget_motion.py \
  --input /data/AI4Animation/dog_walk00_joint_pos.txt \
  --input-format ai4animation_joint_pos \
  --output /tmp/lite3_dog_walk00_reference.npz \
  --source-id ai4animation-dog-walk00 \
  --source-format ai4animation-dog-mocap \
  --flat-only-split train
```

For the raw BVH archive, select one clip explicitly and use the BVH retargeting mode:

```bash
PYTHONPATH=source/rl_training \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/reinforcement_learning/motion_prior/retarget_motion.py \
  --input /home/tjx/rl_training_main/MotionCapture.zip \
  --bvh-member D1_001_KAN01_001.bvh \
  --input-format ai4animation_bvh \
  --joint-limit-mode report \
  --output /tmp/lite3_ai4animation_D1_001_reference.npz \
  --source-id ai4animation-siggraph2018-D1_001_KAN01_001 \
  --source-format ai4animation-bvh-dog-mocap \
  --flat-only-split train
```

The BVH loader removes root translation and orientation, extracts four named
limb end sites in `[FL, FR, HL, HR]` order, maps dog coordinates to Lite3 body
coordinates, and records the selected member/scale in the source metadata supplied
by the caller. Use `--bvh-position-scale` and `--bvh-forward-sign` when a different
BVH archive uses another unit or forward-axis convention.

To inspect a retargeted clip before training, render the IK skeleton as an MP4 or
GIF. This is a kinematic reference playback, not a physics rollout of a learned
policy:

```bash
MPLCONFIGDIR=/tmp/mpl_cache PYTHONPATH=source/rl_training \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/reinforcement_learning/motion_prior/visualize_motion.py \
  --input /tmp/lite3_reference.npz \
  --output /tmp/lite3_reference.mp4 \
  --fps 30 --stride 2
```

For an archive containing many BVH clips, screen every member and scale before
selecting a training reference. The scanner uses world-frame foot stationarity
for contact (so a planted foot is not rejected just because the body frame is
moving), checks the analytic Lite3 workspace and joint limits, and records the
root path used to reject stationary poses:

```bash
PYTHONPATH=source/rl_training \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/reinforcement_learning/motion_prior/screen_bvh_clips.py \
  --input /home/tjx/rl_training_main/MotionCapture.zip \
  --output-csv /home/tjx/rl_training_main/outputs/motion_prior/bvh_screening.csv \
  --output-json /home/tjx/rl_training_main/outputs/motion_prior/bvh_screening.json
```

The current archive screening selected `D1_009_KAN01_001.bvh` at scale `0.006`
as the primary flat reference (`9.12 s`, diagonal contact `0.172`, two-contact
ratio `0.425`, unreachable frames `0`, raw joint-element violation `0.012`,
root path `5.61 m`). Its training NPZ uses the explicit
`--joint-limit-mode clip` projection and records that choice in metadata;
`report` remains the default for inspection. `D1_053_KAN01_001.bvh` at the same
scale is retained as a long validation reference and is also projected before
saving. These are kinematic retargeting results; they still need to be checked
in Isaac Sim and must not be treated as real Lite3 telemetry.

Training comparisons should use separate experiment names and keep the source NPZ path and metadata in the run manifest. Isaac Sim is installed through the `env_isaaclab` pip environment on this machine, so activate that environment before invoking `isaaclab.sh`; otherwise the launcher falls back to the absent binary-install path `_isaac_sim/python.sh`.

```bash
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm PYTHONPATH=source/rl_training:. CUDA_VISIBLE_DEVICES=0 \
/home/tjx/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Flat-Deeprobotics-Lite3-v0 --headless --device cuda:0 \
  --num_envs 4096 --max_iterations 10000 \
  --motion_prior_path /home/tjx/rl_training_main/outputs/motion_prior/lite3_ai4animation_D1_001_reference.npz \
  --motion_prior_mode decay --motion_prior_weight 1.0 \
  --motion_prior_decay_steps 100000
```

The one-iteration CUDA smoke test using this exact reference completed successfully
on 2026-08-04 and wrote `model_0.pt`. Unit tests verify schema, mapping, IK,
interpolation, rewards, and configuration; the smoke test verifies runtime wiring,
but neither establishes final locomotion quality or real-robot transfer.
