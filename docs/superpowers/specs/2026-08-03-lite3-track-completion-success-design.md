# Lite3 Track Completion Success Design

## Goal

Make a formal curriculum episode pass only when Lite3 safely traverses the complete stage-specific challenge. A robot that survives in front of the obstacle must fail with a finite progress cost. Preserve the frozen flat PPO and frozen 100,000-transition flat replay while restarting online adaptation from a clean learner state.

## Root Cause and Evidence

`play_online_adaptation.py` currently defines an episode pass as `not fell and not safety_violation`. It writes a `completion_boundary`, but never compares robot position with that boundary. The supervisor trusts the worker's Boolean without requiring progress evidence.

The two reported passes demonstrate the false-positive path:

- episode 1 ended at `x=-7.114`;
- episode 8 ended at `x=-7.159`;
- the challenge starts at `x=-7.0`;
- both summaries contain `heldout_samples=0`.

Both robots survived for 2,000 steps but never entered the step region. Their missing velocity cost was also converted to `0.0`, making the first no-progress episode the impossible-to-beat best cost and corrupting the no-improvement stop rule.

## Scope

This change applies strict track completion to formal development curriculum workers and sealed challenge evaluation workers. Flat-retention probes continue to measure flat survival and performance and do not inherit challenge completion targets. Generic demonstration runs retain their existing behavior unless explicitly launched as formal challenge workers.

The following artifacts remain valid and must be reused without retraining or recollection:

- frozen flat PPO `model_9999.pt`;
- frozen 100,000-transition flat replay;
- their provenance manifests and checksums.

All residual learner checkpoints, curriculum state, and pass/fail statistics produced by the invalid run are non-formal and must not seed the replacement run.

## Completion Geometry

Completion targets are calculated from the generated `FlatOnlineTrackCfg` dimensions and the robot's post-reset initial world `x`, not from hard-coded world coordinates. The current track resets the robot at the center of the initial flat segment.

Let:

- `x0` be post-reset base position;
- `F` be `flat_length`;
- `S` be `step_length`;
- `L` be total track length;
- `M=0.5 m` be the body-clearance margin;
- `B=0.5 m` be the end-of-track safety buffer.

For `step_2cm`, `step_4cm`, and `step_6cm`, the completion target is:

```text
x0 + F / 2 + S + M
```

This places the target 0.5 m beyond the far edge of the complete 2 m raised platform. Reaching the target proves the robot climbed onto, traversed, and descended from the platform rather than merely touching its leading edge.

For `rough_2cm` and `rough_4cm`, the completion target is:

```text
x0 + L - F / 2 - B
```

This places the target 0.5 m before the physical end of the track. Reaching it proves traversal of the complete rough segment while leaving room before the out-of-bounds edge.

Target construction rejects non-finite inputs, non-positive usable distances, unsupported stage names, and geometry that places the target outside the track interior.

## Worker Runtime Semantics

A formal challenge worker records `initial_x`, `max_x`, `completion_target_x`, `completion_remaining_m`, and `completion_reached`.

After every environment step, it updates maximum progress and evaluates completion before starting the next control step. When the target is reached, it ends the episode immediately and saves the normal atomic experiment checkpoint. This prevents a successful robot from walking off the end of the terrain and being reclassified as a fall.

An episode passes only when all three conditions hold:

```text
completion_reached and not fell and not safety_violation
```

If environment termination and target completion occur on the same step, termination wins and the episode fails. The 0.5 m end buffer should make that case exceptional and preserves fail-closed behavior.

Flat-retention probes explicitly disable challenge completion. Their summaries label the completion mode so their survival result cannot be confused with a formal challenge pass.

## Progress Cost

Every formal challenge summary contains a finite cost:

```text
remaining = max(0, completion_target_x - max_x)
failure_penalty = target_distance if fell or safety_violation else 0
cost = remaining + failure_penalty
```

Here `target_distance = completion_target_x - initial_x`. A safe successful traversal has cost `0`. A safe partial traversal improves monotonically as forward progress increases. A fall or safety violation remains worse than a safe run with the same progress. No missing metric is converted to a perfect score.

## Supervisor Validation

The supervisor does not trust `passed` alone. For every formal episode it verifies:

- completion mode is the expected strict mode;
- all completion positions and cost are finite numbers;
- `max_x >= initial_x` and target distance is positive;
- `completion_remaining_m` matches the positions;
- `completion_reached` agrees with `max_x >= completion_target_x`;
- `passed` agrees with completion, fall, and safety fields;
- a passed episode has cost zero;
- stage and seed still match the request;
- checkpoint and checksum remain valid.

Malformed or contradictory evidence is a worker protocol error and cannot advance curriculum state.

## Protocol and Compatibility

The formal protocol hash includes completion semantics, margin, end buffer, and progress-cost version. Changing any of these values changes the hash. Existing experiment checkpoints and curriculum state therefore fail closed instead of silently resuming under different success semantics.

The replacement run starts in a new output directory with:

- the existing frozen PPO;
- the existing frozen flat replay;
- a freshly initialized residual actor, critics, optimizers, novelty gate, and curriculum coordinator.

The invalid run directories remain untouched as diagnostic evidence.

## Tests

Unit and integration tests must cover:

- step targets use the far platform edge plus 0.5 m;
- rough targets use the complete track minus the 0.5 m safety buffer;
- surviving for all requested steps before the target fails;
- crossing only the leading edge of a step fails;
- crossing the complete step target safely passes and stops early;
- crossing the complete rough target safely passes and stops early;
- simultaneous termination and completion fails;
- fall and safety penalties produce finite nonzero costs;
- additional safe progress strictly lowers cost;
- flat-retention probes remain separate from strict challenge completion;
- supervisor rejects a false `passed=true` without completion evidence;
- protocol hashes differ from the invalid semantics;
- stale curriculum and learner checkpoints cannot resume;
- the complete CPU suite, compile checks, and `git diff --check` pass.

After CPU verification, run short Isaac smoke probes for one incomplete trajectory and one synthetic/reachable completion trajectory before launching a new formal curriculum. The smoke evidence must show computed target, maximum position, completion decision, and safety state.

## Non-Goals

- No PPO retraining.
- No flat replay recollection.
- No changes to `/home/tjx/Lite3_rl_deploy`.
- No Lite3 USD edits.
- No safety-threshold relaxation.
- No real-robot online exploration.
- No reuse of residual learner state produced by the invalid curriculum.
