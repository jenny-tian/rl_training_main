# Lite3 Deterministic Deployment Material Design

## Goal

Make the frozen flat Lite3 policy safe and reproducible in `FlatOnline-Deeprobotics-Lite3-v0` and `HeldoutBoxes-Deeprobotics-Lite3-v0` after a direct standing reset, without changing the policy checkpoint, the 30 rad/s safety limit, the Lite3 USD asset, the strict-flat training task, or `/home/tjx/Lite3_rl_deploy`.

## Root Cause Evidence

The same `model_9999.pt` checkpoint passes the original strict-flat task for 200 steps with zero falls, zero safety pauses, zero projected actions, and a mean velocity error of 0.0598 m/s. Its joint speed remains below the 30 rad/s protection threshold.

The failure can be reproduced in the original Flat task by applying the deployment configuration differences one at a time:

- A fixed +0.3 m/s command remains safe.
- A deterministic upright root reset remains safe.
- Disabling policy observation corruption remains safe.
- Disabling all startup physical-property events reproduces the FlatOnline failure, including a 59.34 rad/s peak joint speed and the same divergent trajectory.
- Restoring only `randomize_rigid_body_material` reduces the peak to 17.25 rad/s and restores forward motion.

The Lite3 USD therefore does not provide a deployment-safe nominal collision material on its own. Removing the startup material assignment exposed an unintended physics-domain mismatch. Command slew, generated-mesh collision, the direct standing reset, and the checkpoint are not the cause.

## Selected Design

The two deployment-matched tasks will explicitly assign one deterministic collision material to every Lite3 rigid-body collision shape during environment startup:

- static friction: 0.65
- dynamic friction: 0.55
- restitution: 0.0
- material buckets: 1

The implementation may use IsaacLab's existing `randomize_rigid_body_material` event mechanism, but it is not randomization: every range has identical lower and upper bounds and the event has only one bucket. Every run receives exactly the same material.

The values lie inside the physical-property distribution used to train the frozen Flat policy. Zero restitution is selected to avoid injecting rebound energy during the initial foot contacts.

All other deployment physical-property randomization remains disabled, including mass, inertia, center of mass, actuator gain, external force, and push randomization. Observation corruption remains disabled. The standing reset and acceleration-limited command lifecycle remain unchanged.

## Configuration Ownership

A shared helper in `rl_training.online_adaptation.terrain_guards` will return the canonical deterministic material parameters. `FlatOnline` and `HeldoutBoxes` will use this helper so the two environments cannot silently drift apart.

`assert_flat_online_deployment_config` will require the material event, require the Isaac material-assignment function, and validate the complete fixed parameter mapping. It will continue rejecting every other startup or reset randomization that is not part of the standing reset contract.

The original `Flat-Deeprobotics-Lite3-v0` task will not change. Its broad material randomization remains part of PPO robustness training.

## Safety and Failure Handling

The 30 rad/s joint-speed protection threshold remains unchanged. No action limit, actuator limit, termination threshold, or safety fallback will be relaxed to make the probe pass.

If the fixed nominal material still exceeds the safety threshold in the actual FlatOnline path, the change is not accepted. The implementation returns to root-cause investigation rather than tuning the safety threshold.

## Verification

Implementation follows test-driven development:

1. Add failing configuration and guard tests that require the canonical fixed material mapping in both deployment tasks and reject altered values.
2. Implement the shared mapping and enable the deterministic startup material event.
3. Run the focused tests, the complete CPU suite, syntax compilation, and `git diff --check`.
4. Run the frozen checkpoint on FlatOnline for a short diagnostic and then 500 steps. It must not fall or report a joint-velocity safety violation.
5. Recheck the original Flat task for 500 steps to ensure the baseline was not changed.
6. Run the HeldoutBoxes evaluation before starting online learning.

Before any real-robot deployment, run a deterministic material sweep using at least low, nominal, and high friction cases within the Flat training distribution. Each case uses fixed values rather than stochastic sampling. All cases must preserve the 30 rad/s protection threshold and existing fall/tilt safety gates.

## Out of Scope

- Retraining the frozen Flat PPO solely for this bug.
- Adding belly-down stand-up behavior.
- Editing Lite3 USD assets.
- Editing `/home/tjx/Lite3_rl_deploy`.
- Randomizing material during the online-learning run.
- Relaxing safety limits.
