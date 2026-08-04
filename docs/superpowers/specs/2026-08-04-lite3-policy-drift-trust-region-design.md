# Lite3 Policy-Drift Trust Region Design

## Goal

Add an optional behavior-space trust region for residual Actor publication. A
candidate update may train normally, but it is published only when its
deterministic residual change on the episode's fixed diagnostic batch stays
below a calibrated MSE threshold.

## Scope

This slice changes candidate acceptance only. It does not change the SAC loss,
the hard action safety filter, the frozen PPO policy, replay sampling, or the
default runtime behavior when no threshold is configured.

## Design

`ActorDriftMonitor.measure()` already compares a candidate Actor with an
episode-local frozen reference and returns `PolicyDrift.residual_mse`. The
publication helper will accept an optional `max_residual_mse` value:

```text
measure candidate drift
if max_residual_mse is configured and drift.residual_mse > max_residual_mse:
    reject the complete SAC transaction with policy_residual_mse_limit
else:
    publish the candidate atomically
```

The threshold is optional and must be finite and non-negative. `None` preserves
the current behavior. A threshold of zero means only zero deterministic
behavioral drift is accepted. The existing development-only
`calibrate_drift_threshold()` helper remains the source of frozen thresholds;
this slice does not recompute thresholds during sealed evaluation.

The CLI exposes `--max-policy-residual-mse`. The curriculum command forwards
the value through its target configuration so that the setting is part of the
configuration hash and cannot silently differ between workers.

## Failure handling

Rejection must restore Actor, Critic, target Critic, optimizers, entropy
parameter, and version exactly as before the attempted update. No publisher
call or Actor-update event is emitted. The normal `UpdateStats.rejection_reason`
field carries the reason for audit logs.

## Verification

- Unit tests reject invalid thresholds.
- Unit tests reject an over-limit candidate without publishing or mutating the
  updater state.
- Unit tests accept a candidate below the limit and preserve the existing
  publication order.
- CLI and curriculum command tests verify parsing and forwarding.
- The existing online-adaptation test suite must have no failures other than
  explicitly skipped Isaac-only cases.
