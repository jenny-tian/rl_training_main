# Lite3 Frozen Novelty, Progress, and Timeout Design

## Goal

Make residual adaptation reliably start and persist when Lite3 stalls at a new
terrain challenge, without retraining the frozen flat PPO or recollecting the
100,000-transition flat replay.

## Evidence and root cause

The first full-track run showed two independent failures:

- Episode 14 completed 85 actor updates, then a single 35.94 ms control-wall-time
  spike triggered `control_timeout` and restored actor version 0. Normal control
  compute time was approximately 2-5 ms.
- `NoveltyGate` updated its exponential baseline while monitoring the challenge.
  Repeated stalled observations were therefore absorbed into the baseline. By
  episode 17, the robot remained near x=-7.154 with velocity error near 0.5, but
  novelty score had fallen to 0.000195 and actor updates were disabled.
- The gate's initial 256-step calibration also ran on the challenge. At 20 ms per
  step this is 5.12 seconds, long enough for the base policy to reach and stall
  at the obstacle, so freezing only the post-calibration EMA is insufficient.

The SAC optimizer itself is operational: actor updates were published, and the
critic continued to update in episodes where the actor gate remained closed.

## Considered approaches

### Selected: dedicated flat baseline probe

Before the first challenge episode, run the frozen base policy on deterministic
flat terrain and save the resulting novelty statistics as a provenance-checked
artifact. Every formal challenge loads this artifact and prohibits baseline
updates. This adds one short simulation probe but establishes the baseline on
the behavior it is intended to represent.

### Rejected: shorter challenge calibration

Reducing calibration from 256 to roughly 100 steps might finish before the first
obstacle, but it depends on command ramp duration, actual speed, reset position,
and terrain geometry. A later geometry or startup change could silently
reintroduce contamination.

### Rejected: freeze the current challenge baseline

Stopping the EMA after calibration prevents future drift but preserves the
already contaminated initial challenge calibration. It does not address the
observed root cause.

## Architecture

### Frozen flat novelty baseline

The curriculum coordinator runs one base-policy flat probe before episode 1.
The probe writes a versioned baseline artifact containing the novelty
configuration, observation count, means, variances, source seed, and source
metrics hash. The coordinator records the artifact hash in the run manifest.

The first online worker loads the artifact into `NoveltyGate` in `MONITORING`
state. Later workers restore the same statistics through the experiment
checkpoint. Formal challenge workers call the gate with baseline updates
disabled. Flat probes may update their local baseline, but can never overwrite
the sealed baseline artifact after challenge execution starts.

Missing, malformed, non-finite, configuration-mismatched, or hash-mismatched
baseline artifacts fail closed before any online interaction.

### Forward-progress novelty

A separate reset-local monitor observes base x while a positive forward command
is active. It keeps monotonic best progress and a counter since the last
meaningful improvement. At least 0.05 m of new progress resets the counter. A
100-step interval without meaningful progress emits persistent progress
novelty.

`NoveltyGate` combines progress novelty with velocity, tilt, reward, and contact
novelty. Persistent progress novelty opens the same actor gate and keeps it open
until progress resumes and the existing recovery rule is satisfied. Paused and
zero-command steps do not advance the stall counter.

The monitor resets at each formal episode. Its current status, best x, stalled
steps, and progress contribution are written to `metrics.jsonl` and the episode
summary so a run can prove why adaptation was or was not enabled.

### Control timeout semantics

A control timeout remains a strict safety violation for the episode. The worker
immediately applies the existing zero-residual/base-action fallback, pauses actor
updates for that step, excludes the transition from both replay buffers, and
records `control_timeout`.

An isolated timeout does not issue an actor rollback command and does not clear
the adaptation episode. Policy weights do not change network structure or
inference workload, so a wall-time scheduling spike is not evidence that the
new actor parameters are unsafe. Actor rollback remains mandatory for physical
policy failures such as excessive tilt, invalid values, non-foot contact,
environment termination, and measured post-update performance degradation.

This changes only weight-retention semantics. It does not weaken the episode
pass contract: any timeout still makes the episode fail strict acceptance.

## Protocol and checkpoint compatibility

The formal protocol includes fixed identifiers and parameters for:

- frozen-flat novelty baseline semantics;
- progress novelty version, 0.05 m improvement threshold, and 100-step window;
- timeout fallback-without-weight-rollback semantics.

These values participate in the protocol SHA-256. Existing curriculum and
experiment checkpoints therefore cannot resume under the new behavior. The new
run must use a fresh output directory and a newly initialized residual learner,
while continuing to reference the existing frozen PPO and flat replay.

## Verification

Unit and integration tests must demonstrate:

- challenge observations cannot modify a loaded frozen baseline;
- a flat artifact round-trip preserves finite statistics and rejects schema,
  configuration, provenance, and hash mismatches;
- stationary commanded motion enters adaptation after the configured window;
- genuine progress, pauses, and zero command do not produce false stall events;
- a control timeout applies fallback and marks strict failure but emits no
  rollback command and preserves the active actor version;
- physical safety violations and performance degradation still roll back;
- the new protocol fields change the hash and reject old checkpoints;
- the complete online-adaptation test suite passes.

After automated verification, start a clean curriculum run. The first 2 cm
episode must show a sealed flat baseline, nonzero progress novelty when stalled,
actor updates after novelty, and no actor-version reset caused solely by an
isolated control timeout.
