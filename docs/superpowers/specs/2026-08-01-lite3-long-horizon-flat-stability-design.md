# Lite3 Long-Horizon Flat Stability Design

## Goal

Train and qualify a frozen Lite3 flat PPO that completes the existing formal
100,000-transition replay with zero falls and zero safety-invalid rows. Keep the
runtime tilt limit at `0.65`, abort on the first invalid row, and never discard
or reinterpret invalid evidence.

## Confirmed Root Cause

The formal seed-42 replay failed deterministically at attempted row 65,558. The
robot gradually tilted from `0.57965` to `0.63535` and then `0.67776`, crossing
the runtime projected-gravity XY norm limit of `0.65`. There was no non-finite
state, non-foot contact, joint-position projection, or fall before the abort.

Two protocol mismatches allowed this checkpoint to qualify:

1. Lite3 training terminates when either projected-gravity XY component exceeds
   `0.7`, whereas runtime safety terminates on the XY vector norm exceeding
   `0.65`. Some runtime-unsafe orientations are therefore training-valid.
2. Qualification uses short random-command strict-flat runs, while replay uses
   deterministic resets and a fixed `0.5 m/s` command. The existing five seeds
   cover only two episodes each and cannot expose the seed-42 episode-66 event.

## Selected Approach

Align the training failure boundary with runtime safety, add a continuous
pre-boundary recovery incentive, and qualify on the exact deployment
distribution for 100 episodes before formal replay.

This combines prevention and detection. Merely extending qualification would
reject the current checkpoint without improving it. Merely changing reward
weights would leave training and runtime with contradictory definitions of an
unsafe orientation.

## Training Contract

Define a shared Lite3 projected-gravity XY norm limit of `0.65`. Both the Lite3
training termination and the online-adaptation safety filter must consume that
same value. The generic locomotion environment may retain its existing behavior;
the change is scoped to Lite3 configuration and a parameterized termination
function. A back-up flip (`projected_gravity_z > 0`) remains immediately terminal.

Add a continuous Lite3 tilt-margin penalty based on XY norm. It is zero below a
guard value of `0.55` and rises smoothly between `0.55` and `0.65`. Crossing
`0.65` remains terminal. The penalty supplies recovery gradients close to the
boundary without changing the formal boundary or suppressing ordinary gait
motion. Existing flat-orientation and base-height penalties remain enabled.

Before a full 10,000-iteration run, execute a short training smoke test and
inspect resolved configuration and episode metrics to prove that the new reward
and termination are active and finite. A smoke checkpoint is diagnostic only and
cannot be published as the baseline.

## Qualification Contract

Qualification has two gates:

1. Preserve the existing five-seed strict-flat screening for zero falls, zero
   paused/projected/non-finite rows, and mean velocity error at most `0.35`.
2. Add a long-horizon deployment-matched gate using the `FlatOnline` task, seed
   42, deterministic resets, fixed `0.5 m/s` command, and exactly 100 completed
   1,000-step episodes. It must execute 100,000 attempted rows with 100 timeout
   truncations and no early termination.

The long-horizon record includes attempted steps, completed episodes, timeouts,
falls, safety intervention counts, maximum projected-gravity XY norm, minimum
base height, non-finite count, and mean velocity error. It fails on any fall,
premature termination, pause, projection, non-finite row, incomplete episode, or
tilt above `0.65`. The worker stops immediately on invalid safety evidence; it
does not skip the row.

Qualification and formal replay remain separate runs. Passing qualification is
permission to attempt replay, not permission to weaken replay validation.

## Formal Replay and Provenance

After the new checkpoint passes both qualification gates:

1. create a new baseline manifest containing the checkpoint, resolved configs,
   training command, and both qualification reports;
2. run a short cross-reset probe in a fresh directory;
3. run the unchanged 100,000-transition formal collector in a fresh directory;
4. abort on the first fall or safety-invalid row and preserve all failure logs.

No later checkpoint selection, threshold tuning, or seed changes may be based on
the formal replay outcome. Any failed attempt requires a newly trained candidate
and a fresh qualification cycle.

## Testing

Tests are written before production changes and must prove:

- Lite3 termination uses XY norm and trips just above `0.65`, including diagonal
  vectors whose individual components are below the old `0.7` boundary;
- the continuous margin penalty is zero at/below `0.55`, finite and monotonic in
  the guard band, and maximal at `0.65`;
- training and safety import the same threshold constant;
- long-horizon qualification rejects incomplete steps/episodes, premature done,
  excessive tilt, and every nonzero safety count;
- the qualification CLI defaults identify the deployment-matched task, seed 42,
  100 episodes, and 100,000 steps;
- existing action clipping, joint order, collector abort, and manifest tests stay
  green.

Before claiming completion, run the complete CPU suite, compilation, whitespace
checks, GPU smoke training, full training, both qualification gates, reset probe,
and formal replay validation.

## Expected Duration

Implementation, tests, and smoke validation should take roughly 1–2 hours. Full
training is expected to take about 5 hours, followed by approximately 2–3 hours
for qualification, reset probing, and formal replay. Total expected wall time is
about 8–10 hours if the first candidate passes.
