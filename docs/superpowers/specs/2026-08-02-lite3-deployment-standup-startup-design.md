# Lite3 Deployment Stand-Up Startup Design

## Goal

Make every `FlatOnline-Deeprobotics-Lite3-v0` and
`HeldoutBoxes-Deeprobotics-Lite3-v0` episode start the same way as the
reference Lite3 deployment controller: begin from a deterministic belly-down
configuration, execute a controlled two-stage stand-up, then hand control to
the unchanged PPO checkpoint and begin ramping the walking command.

The ordinary `Flat-Deeprobotics-Lite3-v0` and rough-terrain training tasks
remain unchanged. The reference `/home/tjx/Lite3_rl_deploy` repository is
read-only and is not modified.

## Motivation and Evidence

The existing FlatOnline reset places the base at `z=0.375 m` with the default
standing joint pose and immediately runs PPO. All four feet initially sit about
`4.85 cm` above the generated flat terrain. The first contact transient reached
`35.0339 rad/s` before command ramping and `40.429 rad/s` after the exposed
command was correctly reduced to `0.03 m/s`. Both exceed the strict
`30 rad/s` deployment threshold. Eight zero-action settling steps did not solve
the handoff: later PPO contact reached `43.971 rad/s`.

The reference Lite3 controller uses an explicit state machine instead:

1. idle with zero joint control while checking sensors;
2. cubic interpolation from the measured pose to a `0.12 m` folded pose;
3. cubic interpolation from `0.12 m` to a `0.33 m` standing pose;
4. PPO control only after stand-up completes.

The reference inverse-kinematics targets are consistent with its MuJoCo model:

- `0.12 m`: `[0, -1.354531, 2.549477]` per leg;
- `0.33 m`: `[0, -0.653535, 1.271092]` per leg.

A headless MuJoCo reproduction with zero control settled to a root height of
approximately `0.053417 m`, with torso and shank collision geometries touching
the floor. Therefore the intended initial condition is genuinely belly-down,
not merely a low standing pose. Isaac uses a different USD collision model, so
its root height must be calibrated from contacts rather than copied numerically
from MuJoCo.

## Scope

This change applies on every reset of FlatOnline and HeldoutBoxes, including
resets caused by timeouts, falls, or safety termination. It applies consistently
to deployment qualification and online-adaptation execution. Stand-up is a
deployment procedure, not a learned policy behavior. No PPO retraining is part
of this change.

Stand-up transitions are not policy experience: they are excluded from replay,
online updates, policy episode length, and the 100,000-step deployment
qualification count. Formal replay qualification remains separately blocked
until authentic checkpoint provenance inputs are recovered.

## Architecture

### Pure stand-up trajectory module

A dependency-light module owns the joint targets and cubic interpolation. It
accepts the measured initial joint positions and velocities, phase elapsed time,
and phase duration. It returns desired joint positions in canonical Lite3
leg-major order.

Each phase lasts exactly `1.5 s`:

1. measured reset state to `[0, -1.354531, 2.549477] x 4`, ending at zero target
   velocity;
2. folded pose to `[0, -0.653535, 1.271092] x 4`, again ending at zero target
   velocity.

The module has no Isaac dependency, never mutates inputs, validates tensor
shape and finite values, and can be fully tested on CPU. The generated physical
joint targets are converted to the existing normalized policy-action convention
before being sent to `JointPositionAction`. This preserves the environment's
12-dimensional action contract and existing actuator torque limits and PD
control. No joint-velocity action term is introduced.

### Shared deployment startup wrapper

`Lite3DeploymentStartupWrapper` wraps only FlatOnline and HeldoutBoxes. Its
`reset()` performs the environment reset and then advances all stand-up physics
steps internally. The outer caller receives only the post-stand-up observation.
The wrapper uses the same startup routine after every automatic episode reset.

All deployment-facing scripts install the wrapper through one shared environment
construction function. This prevents qualification, online adaptation, residual
training, and future replay collection from silently using different startup
semantics. Passing an unsupported task or environment shape is an error rather
than a fallback to an unprotected startup.

The wrapper exposes explicit startup state to the environment termination
functions. It does not merely discard `done` after stepping, because Isaac would
already have automatically reset the environment at that point.

### Startup-aware termination gates

FlatOnline and HeldoutBoxes use startup-aware non-foot-contact and orientation
termination functions. While `startup_active` is true, expected torso/shank
contact and the belly-down orientation are exempt from those two episode
terminations. Immediately before PPO handoff, `startup_active` becomes false and
the original deployment termination criteria apply without relaxation.

The exemption is narrowly scoped. It does not disable the wrapper's mandatory
finite-value, joint-position-limit, or `30 rad/s` joint-speed checks. It also
does not alter the normal-task configurations or their termination behavior.

### Command and history lifecycle

During both stand-up phases, the exposed velocity command and its sampled target
remain exactly zero. The existing `(0.5, 0.5, 0.5) m/s²` command acceleration
limits are retained, but their timer begins only after stand-up succeeds. The
forward command then rises from zero by approximately `0.01 m/s` per `0.02 s`
policy step until it reaches the unchanged `0.5 m/s` target.

Before returning the first PPO observation, the wrapper resets command-ramp
state, action-manager previous-action state, and policy-visible episode counters.
It increments a handoff generation included in reset information and step
extras. The shared deployment session observes that generation and calls a
public reset method on the online safety filter before the next policy decision.
This keeps the environment wrapper independent of the online agent while making
the safety-history reset mandatory and testable. The first PPO observation
therefore contains zero command and a canonical standing handoff action rather
than stale stand-up history.

## State and Data Flow

```text
environment reset
    -> deterministic folded, belly-down Isaac pose
    -> startup_active = true; command = target command = zero
    -> phase 1 (1.5 s): measured pose -> 0.12 m folded target
    -> phase 2 (1.5 s): folded target -> 0.33 m standing target
    -> validate final height, attitude, and four-foot support
    -> clear command/action/episode/safety temporal state
    -> startup_active = false
    -> return zero-command standing observation to PPO
    -> PPO acts; command ramps 0 -> 0.5 m/s
```

No phase-1 or phase-2 transition crosses the policy, replay, reward-learning, or
online-update boundary.

## Isaac Belly-Down Calibration

The deterministic folded joint target is fixed by the reference controller. The
Isaac base height is selected by a one-variable GPU calibration probe using the
actual FlatOnline USD and generated-flat terrain. The accepted height must
produce all of the following after settling without active PPO control:

- torso and shank contact consistent with a belly-down folded posture;
- no material collision penetration or explosive depenetration;
- finite joint/root state;
- no joint-limit violation;
- maximum joint speed at or below `30 rad/s`.

The calibrated value is then made an exact FlatOnline/HeldoutBoxes configuration
constant and locked by the strict deployment guard. Calibration is diagnostic,
not runtime searching: production reset remains deterministic.

## Failure Handling

Every internal stand-up step checks finite root and joint state, actual joint
limits, and maximum absolute joint velocity. A violation raises a startup error
that includes phase, elapsed time, affected joint or state, measured value, and
limit. It must not return a policy observation or append replay.

After the second phase, handoff additionally requires base height at least
`0.20 m`, projected-gravity XY norm at most `0.65`, and contact force greater
than `1 N` on every foot for 10 consecutive `0.02 s` internal steps. These are
standing validation steps with the final standing target held; they remain
hidden from policy/replay counts. A failed handoff is a startup failure, not an
ordinary policy episode termination. The current `30 rad/s` joint-speed and
`0.65` projected-gravity thresholds are never weakened.

## Verification

### CPU tests

- cubic trajectory position and velocity at both endpoints;
- continuity at the phase boundary and no target overshoot;
- canonical physical-target/action-space round trip;
- rejection of invalid shapes, times, durations, and non-finite inputs;
- wrapper hiding internal steps and returning only the handoff observation;
- startup termination exemption being active only inside startup;
- command, previous action, episode counter, handoff generation, and safety
  history reset at handoff;
- no startup behavior in ordinary Flat/Rough tasks.

### Isaac GPU gates

1. Calibrate and freeze the folded belly-down root height.
2. Run the complete three-second stand-up while logging joint speed, target
   tracking, contacts, base height, and projected gravity at every policy step.
3. Require every stand-up sample to remain at or below `30 rad/s`.
4. Run eight PPO transitions after handoff and require all samples to remain at
   or below `30 rad/s`, with no termination or safety pause.
5. Repeat on generated flat terrain and the strict-plane comparison.
6. Only after all startup gates pass, run the existing five-seed ordinary-flat
   source qualification and the 100-episode, 100,000-policy-step FlatOnline
   deployment qualification.

If a gate fails, investigation returns to the measured failing phase. Parameters
are not tuned randomly and safety thresholds are not relaxed to manufacture a
pass. The unchanged PPO checkpoint is retained unless the fully verified
stand-up handoff still proves incompatible with the policy.

## Acceptance Criteria

- Every FlatOnline/HeldoutBoxes reset visibly follows belly-down, folded,
  standing, and PPO phases.
- Each interpolation phase lasts exactly `1.5 s`.
- Stand-up steps never enter replay, online learning, policy step counts, or
  deployment qualification episode counts.
- The command remains zero throughout stand-up and ramps only after handoff.
- Non-foot-contact/orientation exemptions exist only while startup is active.
- All startup and eight-step handoff joint speeds are at most `30 rad/s` on both
  generated flat and strict-plane probes.
- The ordinary Flat/Rough configurations and existing checkpoint are unchanged.
- Full CPU regression, compilation, whitespace checks, and launched Isaac
  configuration checks pass before qualification evidence is recorded.
