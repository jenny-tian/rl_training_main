# Lite3 Safe Flat Baseline Design

## Goal

Produce a newly trained strict-flat Lite3 PPO whose deterministic actions can be collected for 100,000 transitions with zero falls and zero safety-invalid rows. Preserve the existing formal protocol: invalid rows are neither hidden nor skipped, and safety thresholds are not relaxed.

## Confirmed Failure

The current seed-42 checkpoint passes the existing five-seed fall and velocity-error qualification, but formal replay collection deterministically fails at attempted row 1005. The safety evidence shows no fall, pause, non-finite value, excessive tilt, low height, excessive joint velocity, or non-foot contact. The frozen PPO requests `FR_HipX=-4.227554`; converting that request to a physical target crosses the safety filter's joint-position boundary and produces a `0.245395` normalized-action correction. The row is therefore correctly safety-invalid under the current protocol.

The investigation also found an independent adapter defect: Isaac stores Lite3 joints in joint-type order, while the PPO uses leg-major order. All online-adaptation joint observations, limits, targets, and power values must be explicitly selected into PPO order.

## Selected Approach

Retrain the flat PPO with per-joint action clipping that is strictly inside the same joint-target boundary enforced by `SafetyFilter`.

For each joint, derive the normalized action interval from:

```text
lower_action = (soft_joint_lower + 0.02 - default_joint_position) / action_scale
upper_action = (soft_joint_upper - 0.02 - default_joint_position) / action_scale
```

The configured clip uses a small inward numerical tolerance, so the training action manager and runtime safety filter cannot disagree because of floating-point round trips. The bounds remain joint-specific rather than applying a restrictive common bound to every joint.

This is preferred over the alternatives:

- A reward penalty cannot guarantee zero violations.
- Clipping only during replay would change the executed controller after qualification and make the manifest misleading.
- Ignoring projected rows or disabling joint projection would weaken the formal safety criterion.
- Changing the Actor output architecture is more invasive than constraining the existing action interface.

## Components

### Joint-order adapter

Define one canonical Lite3 PPO joint-name sequence. Resolve its Isaac joint IDs with order preservation when the adapter starts, fail fast if all names do not match, and use those IDs for joint positions, velocities, limits, default positions, applied torques, and proprioceptive frames.

### Safe training action configuration

Replace the current effectively unbounded action clip with explicit HipX, HipY, and Knee normalized-action ranges derived from the asset configuration and the safety margin. Keep the existing action scales and default-position offsets unchanged.

The implementation must expose the derivation as a small deterministic helper or validated constants so tests can prove that every clipped physical target remains inside the safety boundary. Configuration and safety tests must fail if the asset defaults, scales, limits, or margin change incompatibly.

### Qualification

Extend qualification records with safety evidence while retaining the existing fixed seeds `(7, 11, 19, 23, 31)`, 2,000 steps per seed, zero falls, and mean velocity-error threshold `0.35`.

Every deterministic action must be passed through the same ordered joint state and `SafetyFilter` used by replay collection. Qualification passes only when all seeds have:

- zero falls;
- zero paused rows;
- zero projected rows;
- zero non-finite rows;
- aggregate mean velocity error at or below `0.35`.

The qualification report records these counts. The baseline manifest hashes the new checkpoint, its saved configurations, training command, and the enhanced qualification report.

### Formal replay collection

Run the existing strict-flat collector with seed 42 for exactly 100,000 accepted transitions. It continues to abort immediately on the first fall or safety-invalid row. It does not discard invalid rows and continue, reinterpret a projected action as the PPO base action, or reduce the safety margin.

Diagnostic metrics retain the reason fields added during root-cause analysis. The final manifest still requires exactly 100,000 transitions, zero falls, zero safety-invalid rows, and a frozen replay.

## Data Flow

```text
PPO output
  -> training/runtime per-joint action clip
  -> ordered Lite3 joint-target conversion
  -> unchanged SafetyFilter verification
  -> Isaac environment
  -> ordered transition fields
  -> immutable flat replay
```

Training and inference use identical action scales, offsets, ordering, and clipping. Qualification independently rechecks the result with the runtime safety filter before the checkpoint can be manifested.

## Failure Handling

- Reject a joint-name/order mismatch before stepping Isaac.
- Reject derived bounds that are non-finite, inverted, or do not leave an inward tolerance.
- Stop qualification on malformed evidence and mark any nonzero intervention count as failed.
- Stop formal replay collection on the first fall or safety-invalid row.
- Preserve failed diagnostic directories; use a new output directory for every retry.
- Do not select a later checkpoint or alter thresholds based on formal replay outcomes.

## Verification

Before training:

- Unit-test raw-Isaac-to-PPO joint reordering.
- Unit-test every configured action bound against physical joint boundaries.
- Unit-test qualification aggregation for projected, paused, and non-finite failures.
- Run the complete CPU test suite, compilation, and whitespace checks.

After training:

1. Resolve the unique new run and highest checkpoint.
2. Run enhanced five-seed qualification on GPU 2.
3. Build and validate a new baseline manifest.
4. Run a short diagnostic collection through at least two episode resets without an intervention.
5. Run the formal 100,000-transition collection on GPU 2.
6. Validate replay and manifest hashes and rerun the complete test suite.

## Expected Duration

Implementation and verification should take about 1–2 hours. The 10,000-iteration GPU training, qualification, reset probe, and formal replay collection are expected to take approximately 6–8 additional hours, for a total of about 7–10 hours depending on GPU throughput.
