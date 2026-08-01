# Lite3 Startup Joint-Velocity Safety Design

## Objective

Make the existing Lite3 flat baseline pass the deployment safety contract by
replacing the instantaneous `0.5 m/s` FlatOnline command step with a bounded
startup ramp. The contract remains unchanged: observed joint velocity must not
exceed `30 rad/s`, invalid rows may not be skipped, and tilt and contact checks
remain active.

## Evidence

The current candidate passes five-seed strict-plane screening but fails the
FlatOnline deployment startup before transition four. The generated terrain
reaches `35.0339 rad/s`; an otherwise equivalent strict-plane probe reaches
`44.0181 rad/s` at transition three with a contact impulse near `363 N`.
Continuous zero actions remain below `30 rad/s`, while PPO actions trigger the
violation. An explicit `velocity_limit_sim` probe does not change the failing
trajectory. The current FlatOnline command is already `0.5 m/s` on the first
policy transition. These results isolate the immediate defect to the
policy/contact response to a command step under the deterministic deployment
startup distribution rather than terrain generation or a missing simulator
velocity field.

## Chosen Approach

Change the FlatOnline command trajectory without changing its final target:

1. Add an opt-in slew-limited velocity command term that stores the sampled
   target separately from the command exposed to observations and rewards.
2. On each environment step, move each exposed command component toward its
   target by at most `max_acceleration * step_dt`, without overshoot.
3. Enable the term only for `FlatOnline` and held-out deployment tracks. Reset
   the exposed command to zero at every episode boundary and keep the final
   target exactly `(0.5, 0.0, 0.0)`.
4. Use a forward acceleration limit of `0.5 m/s^2`, reaching `0.5 m/s` in one
   second. Lateral and yaw limits remain configured even though their current
   targets are zero.
5. Keep the existing policy checkpoint, action clipping, safety thresholds,
   replay validity rules, and terrain protocol unchanged.

The ramp configuration is part of the strict FlatOnline deployment guard so a
formal run cannot silently revert to an instantaneous command step.

## Alternatives Considered

Retraining with `joint_vel_limits` and `joint_vel_l2` could teach the PPO to
handle the command step, but it is slower and changes the policy before testing
the simpler deployment-matched command trajectory. It remains the fallback if
the ramp does not satisfy both safety and tracking criteria.

Relaxing the `30 rad/s` threshold or skipping violating rows would hide the
failure and is explicitly excluded. Setting `velocity_limit_sim` was tested and
did not reduce the contact-driven spike, so it is not part of the fix.

## Implementation Boundaries

The reusable slew limiter belongs in the locomotion velocity command module.
Only Lite3 FlatOnline and held-out deployment configurations enable it. It must
not alter the ordinary flat/rough training command distribution, online replay
validity, safety-monitor thresholds, or terrain geometry protocol.

The separate articulation/contact-sensor foot-index fix remains in scope as an
independent correctness fix for foot-slip metrics, but it is not evidence that
the startup velocity failure is resolved.

## Verification

Implementation follows test-driven development:

1. Add failing unit tests for positive and negative ramps, no overshoot,
   per-environment reset, and the disabled/default behavior.
2. Add failing configuration/guard tests for the exact FlatOnline acceleration
   limits, then make the minimal command and configuration changes.
3. Run the complete CPU suite, `compileall`, and `git diff --check`.
4. Run the deterministic generated-flat startup probe with the existing
   checkpoint and log the commanded velocity alongside joint speed. Reject the
   ramp if any observed joint velocity exceeds `30 rad/s`.
5. For a passing startup probe, rerun five-seed base qualification, then the
   formal 100-episode / 100,000-transition deployment qualification and replay.

Success requires zero falls, pauses, projected/invalid rows, nonfinite rows, and
joint-velocity violations while preserving the existing velocity-error
acceptance criterion. A short probe is only a gate for longer qualification,
not final acceptance evidence.
