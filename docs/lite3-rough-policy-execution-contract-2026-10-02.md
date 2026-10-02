# Lite3 rough-policy execution contract

This describes the anchored route policies from training seeds283/307. Independent101/107 qualification passed; the four registered new-development terrains are still being evaluated. It is an implementation contract, not a declaration that the overall goal is complete.

## Deterministic actor input

The actor consumes48 float32 values in the following order, measured before the control action. Empirical observation normalization is disabled. The tested command is forward0.3m/s, lateral0, yaw rate0, on a world+X course.

| Python slice | Meaning | Scaling |
|---|---|---|
|0:3|Body angular velocity|0.25|
|3:6|Projected unit gravity in body frame|1|
|6:9|Forward/lateral velocity and yaw-rate command|1|
|9|World lateral offset from course origin|divide by1.65m, clip[-2,2]|
|10|Sine of world yaw relative to+X|1|
|11|Cosine of world yaw relative to+X|1|
|12:24|Joint position relative to default pose|1|
|24:36|Joint velocity|0.05|
|36:48|Previous emitted, clipped policy action|1; zeros after reset|

At zero yaw, the route sine/cosine pair is(0,1). Current experiments obtain route position and yaw from simulator state. Equivalent localization and real hardware execution have not been validated. The actor does not consume the critic's privileged linear velocity or a terrain height scan. The51D critic is used for training only.

Canonical joint order is FL,FR,HL,HR, each with HipX,HipY,Knee. The authoritative mapping is `rl_training.lite3_joint_conventions.LITE3_POLICY_JOINT_NAMES`.

## Output and physical action

Use the actor mean with exploration disabled. Apply `clip_lite3_policy_action` before environment stepping and before storing the next action-history input. Per-leg policy bounds are HipX[-3.60,3.60], HipY[-7.40,3.17], Knee[-2.57,5.43]. The physical target is default joint position plus policy action times scale: defaults(0,-0.65,1.3)rad and scales(0.125,0.25,0.25)rad per leg. The shared Lite3 environment retains actuator, contact and orientation limits.

The frozen original-flat action target exists only in training. Inference is a single learned actor, with no teacher blending or flat fallback. The route observation term is enabled using `--route-observations sim-localization-v1` in the audited frozen worker.

## Authoritative artifacts

- Seed283: `artifacts/full_policy_flat_anchor_20261002/train/final.pt`, SHA256 `5f75f26340f6c98002bb105cd62932c0e6e0542fe12ff936a43cac365867d894`.
- Seed307: `artifacts/full_policy_flat_anchor_replication_20261002/train/final.pt`, SHA256 `05f5fd60e291aae2d919c5f5836c040af6a60292e700c390f9bf867781ce020d`.
- Training, history and flat-anchor audits are stored alongside each run. Paired evaluation and the joint replication audit are immutable evidence for101/107.
- `docs/lite3_flat_anchor_transfer_v2_protocol_20261002.json` locks the new-terrain test; its final audit is pending. Do not rerun completed trials or reuse their outputs as new samples.

The tested control interval is0.02s. Gait timing plots and trajectory statistics are descriptive evidence of the learned walking pattern; no new gait-family classification or hardware-transfer claim is implied.
