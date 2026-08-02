# Lite3 Direct Reset Command Handoff Design

## Goal

Hand a standing Lite3 environment directly to the frozen policy without erasing the requested locomotion command. Preserve the command term's existing acceleration-limited startup behavior.

## Root Cause

`UniformThresholdVelocityCommand.reset()` initializes the current velocity command to zero, samples the requested target command, and then ramps the current command toward that target. `Lite3DeploymentStartupWrapper.reset()` subsequently clears both `vel_command_b` and `target_vel_command_b`, permanently discarding the sampled target.

## Design

Remove the wrapper's post-reset command clearing. The underlying environment remains the single owner of command reset and slew state:

- The policy receives the observation returned by the same environment reset that initialized the command term.
- The current command starts at zero.
- The target command remains the configured forward command.
- Normal environment steps ramp the current command toward the target.

Standing pose, physics configuration, policy checkpoint, and the `30 rad/s` startup validation threshold remain unchanged.

## Tests

Add a wrapper regression test with a real command-state-shaped fake term. The test must fail on the current implementation because reset clears the target, then pass after the fix by asserting that both the current and target commands are preserved exactly as initialized by the underlying environment.

Run the focused startup tests, the complete CPU test suite, compile checks, and `git diff --check`. Finally run the frozen `model_9999.pt` policy for 500 FlatOnline steps with the flat-retention probe and inspect command, fall, and joint-velocity safety metrics.

## Non-Goals

- No stand-up trajectory or hidden physics steps.
- No safety-threshold relaxation.
- No PPO or residual-policy retraining.
- No changes to `/home/tjx/Lite3_rl_deploy`.
