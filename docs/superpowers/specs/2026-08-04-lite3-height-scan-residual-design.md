# Lite3 Height-Scan Residual Observation Design

## Goal

Expose the existing Isaac Lab `height_scanner` to the high-level online
residual policy without changing the frozen Flat PPO observation contract.
The residual policy receives a bounded, fixed-size terrain summary and is
forced back to zero residual when the scan is stale, malformed, or non-finite.

This is the first perceptive-adaptation slice motivated by *Motion Priors
Reimagined*. It does not add a depth camera, radar, a learned visual encoder,
or a new terrain-training protocol.

## Motivation and Non-Goals

The current Lite3 adapter can append ten terrain-related values, but those
values are derived from known task geometry. They are useful for held-out
step bookkeeping but are not deployable perception. The new scanner source
replaces that privileged geometry source only when explicitly enabled.

The frozen PPO policy continues to consume the original proprioceptive
observation. The scanner is used only by the residual SAC actor/critic. The
following are out of scope:

- changing the Flat PPO network input or retraining its checkpoint;
- adding raw camera images or a convolutional/transformer encoder;
- treating perception as a hard safety guarantee;
- enabling scanner features in the existing formal flat-only baseline;
- learning or recalibrating scanner normalization from sealed evaluation data.

## Architecture

### Feature source

`Lite3IsaacLabAdapter` gains an explicit terrain feature source:

- `geometry`: the existing development/legacy geometry features;
- `height_scan`: features computed from `height_scanner.data.ray_hits_w`.

The default remains `geometry` for backward compatibility. A height-scan run
must opt in through an immutable runtime/configuration value. The selected
source is recorded in the experiment manifest and checkpoint metadata.

The scanner pattern is fixed for this slice. Ray-hit heights are expressed
relative to the scanner/base reference, clipped to the configured physical
range, and normalized with fixed constants. The ten output values are:

1. finite-hit fraction;
2. global minimum height;
3. global maximum height;
4. global mean height;
5. global height standard deviation;
6. forward-half minimum height;
7. forward-half mean height;
8. forward-half maximum height;
9. left/right mean-height difference;
10. forward/backward mean-height difference.

The helper must validate the expected batch and grid dimensions. Invalid
values are replaced with finite zeros for logging and tensor-shape stability,
but the separate `terrain_valid` mask is false whenever the finite-hit ratio
or required scanner data is invalid.

### Runtime data flow

At each control step:

```text
Isaac Lab height scanner
        |
        v
fixed ten-dimensional summary + terrain_valid
        |
Lite3 adapter frame (existing residual frame dimension)
        |
history buffer -> residual SAC actor/critic
        |
zero residual when terrain_valid is false
        |
SafetyFilter(base PPO action + residual)
```

`terrain_valid` is not appended as an eleventh feature. It is a control and
learning gate so an invalid sensor cannot be interpreted as a real flat patch.
When false, `begin_step` must produce the same action as
`force_zero_residual=True`, mark the pending transition as non-adaptation, and
prevent that transition from entering the adaptation replay. The global replay
may still retain it for diagnostics if its existing safety mask permits.

The scanner is sampled at the existing control update period. This slice does
not add a separate asynchronous sensor thread. A future hardware adapter may
populate the same interface with timestamped data.

## Checkpoint and Provenance Rules

The Flat PPO checkpoint remains compatible because its observation is
unchanged. Residual actor checkpoints are source-specific even though the
tensor dimensions stay fixed: a checkpoint trained on `geometry` features
must not silently load into a `height_scan` run. Loading must reject a source
mismatch with an auditable error.

The metadata records:

- terrain feature source and schema version;
- scanner grid shape and update period;
- normalization/clipping constants;
- minimum valid-hit fraction;
- whether invalid scans were observed during the run.

The existing geometry/terrain protocol hash remains a geometry protocol hash;
the feature source is an algorithm/runtime configuration value and must be
frozen in the selected experiment configuration.

## Failure Handling

The adapter must fail closed for missing scanner objects, shape mismatch,
non-finite ray hits, or a valid-hit fraction below the configured minimum.
The runtime records a structured perception event and uses the frozen base
policy with zero residual for that control step. Repeated invalid scans close
the adaptation gate; recovery requires the existing novelty/safety state
machine to reopen it.

No scanner value may bypass joint limits, residual limits, rate limits,
orientation guards, timeout handling, or rollback logic.

## Verification

Unit coverage must include:

- deterministic ten-feature summaries for a synthetic grid;
- clipping, normalization, and valid-hit fraction behavior;
- missing/non-finite/shape-mismatched scanner data failing closed;
- adapter source selection and metadata;
- invalid-scan zero-residual behavior and adaptation-replay exclusion;
- source mismatch rejection when loading a residual checkpoint;
- legacy `geometry` mode preserving the existing frame shape and behavior.

The full online-adaptation suite must remain green. GPU simulation, real sensor
latency, and sim-to-real domain randomization are explicitly follow-up
experiments, not claims of this implementation slice.
