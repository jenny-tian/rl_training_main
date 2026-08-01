# Lite3 Startup Joint-Velocity Safety Design

## Objective

Produce a new Lite3 flat baseline checkpoint that passes the existing deployment
safety contract during deterministic `0.5 m/s` startup. The contract remains
unchanged: observed joint velocity must not exceed `30 rad/s`, invalid rows may
not be skipped, and tilt and contact checks remain active.

## Evidence

The current candidate passes five-seed strict-plane screening but fails the
FlatOnline deployment startup before transition four. The generated terrain
reaches `35.0339 rad/s`; an otherwise equivalent strict-plane probe reaches
`44.0181 rad/s` at transition three with a contact impulse near `363 N`.
Continuous zero actions remain below `30 rad/s`, while PPO actions trigger the
violation. An explicit `velocity_limit_sim` probe does not change the failing
trajectory. These results isolate the defect to policy behavior under the
deterministic deployment startup distribution rather than terrain generation or
a missing simulator velocity field.

## Chosen Approach

Change the training objective and sampling distribution, then train a new
checkpoint:

1. Enable the existing `joint_vel_limits` reward for all Lite3 leg joints. Its
   soft ratio and weight will penalize approaching actuator velocity limits.
2. Enable a small `joint_vel_l2` penalty to discourage high-speed motion below
   the limit without dominating velocity tracking.
3. Ensure flat training samples include the deployment startup case: nominal
   joint pose, zero joint/root velocity, level orientation, and a fixed
   `0.5 m/s` forward command. Existing randomized cases remain present so the
   policy does not specialize to one reset.
4. Keep the existing action clipping, torque/power/contact penalties, tilt
   guard, and `30 rad/s` deployment threshold unchanged.

The exact reward weights and deterministic-start sampling proportion are
configuration constants covered by tests. Begin conservatively and change only
one training variable per diagnostic run when tuning is required.

## Alternatives Considered

Runtime action smoothing could make the existing checkpoint less aggressive,
but it changes the policy-to-actuator contract after training and can reduce
tracking performance. It remains a fallback only if a newly trained policy
cannot satisfy both safety and tracking criteria.

Relaxing the `30 rad/s` threshold or skipping violating rows would hide the
failure and is explicitly excluded. Setting `velocity_limit_sim` was tested and
did not reduce the contact-driven spike, so it is not part of the fix.

## Implementation Boundaries

The change belongs in the Lite3 training environment configuration and its
configuration tests. It must not alter online replay validity, safety-monitor
thresholds, deployment qualification logic, or held-out terrain protocol.

The separate articulation/contact-sensor foot-index fix remains in scope as an
independent correctness fix for foot-slip metrics, but it is not evidence that
the startup velocity failure is resolved.

## Verification

Implementation follows test-driven development:

1. Add failing configuration tests asserting nonzero `joint_vel_limits` and
   `joint_vel_l2` terms and deterministic-start sampling support.
2. Make the minimal configuration changes and run the focused tests.
3. Run the complete CPU suite, `compileall`, and `git diff --check`.
4. Train a short diagnostic candidate and run the deterministic strict-plane
   and generated-flat startup probes. Reject it immediately if any observed
   joint velocity exceeds `30 rad/s`.
5. For a passing candidate, rerun five-seed base qualification, then the formal
   100-episode / 100,000-transition deployment qualification and replay.

Success requires zero falls, pauses, projected/invalid rows, nonfinite rows, and
joint-velocity violations while preserving the existing velocity-error
acceptance criterion. A short probe is only a gate for longer qualification,
not final acceptance evidence.
