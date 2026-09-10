# Lite3 SAC action contract

The critic input is the requested residual after `residual_gain` and before
`SafetyFilter.project`. The safety filter is part of the environment transition:
the observed reward and successor state reflect the action it actually executed.
Actor Q, deterministic Actor Q and target Q use requests in those same units.
Entropy remains defined on the actor's squashed distribution, before gain.

Replay schema 2 stores both `requested_residual_action` (Q regression input) and
`residual_action` (executed residual, retained for reward/diagnostics). N-step
returns retain both actions from the first transition, and the successor state
from the last. Safety fallback rows remain excluded from learning. The recurrent
observation interface remains unchanged; the critic still approximates value
under partial observation, including the filter's temporal state.

A schema 1 frozen replay whose residuals are all exactly zero can migrate with
zero requested actions. Nonzero or mutable legacy replay cannot recover lost
requests and is rejected. Historical policies can still be loaded separately,
but an old online experiment must not resume learning through schema 1 replay.
Fresh runs use a new directory and a clean initializer; old artifacts remain intact.

Updater checkpoints now also embed `schema_version=1` and
`critic_action_semantics=requested_residual_before_safety_v1`. All training
loaders, including saved stable rollback states, reject missing or incompatible
contracts before changing parameters. This deliberately rejects earlier v10
checkpoints as well: their external metadata does not make an unversioned model
self-describing. Regenerate the initializer for a new run; do not relabel old
models or modify their checksums. Historical actor weights can be extracted for
evaluation separately, but the training loader has no legacy bypass.

Failed restores retain independent snapshots of parameters and optimizer state.
Agent restores also roll back replay and n-step state; online replay is required
to remain mutable so a failed restore cannot leave a frozen buffer behind.

The runner manifest records action semantics, replay schema and residual rate
limit. Training, flat probes, development controls and sealed evaluation share
the same rate limit. Code/configuration mismatch continues to fail closed on
resume; update code only between runs and preserve the original manifest.

Regression coverage includes rate limits, joint limits and tilt guards through
agent sampling, replay save/load and critic regression, plus n-step first-action
selection, legacy migration/rejection and evaluation parameter propagation.
