# Source PPO L1 Training Preparation

Status: `cpu_initialization_passed_runtime_pending`, 2026-10-30.

The user directed continuation of the existing Source training plan and clarified
that failures of the old flat policy are expected learning targets. All six
families remain in scope. No additional broad failure-analysis stage or new
permission is required for this preparation. This package is not a runnable
training lock: the integration work below still has to be completed.

## Implemented and Verified

- `source_scan/source_training.py` loads the pinned actor mean weights, explicitly
  sets exploration standard deviation to the draft's 1.0, and leaves critic,
  optimizer and iteration state inaccessible to the loader. Hash, schema,
  finite tensors and ELU policy-only architecture are checked before mutation.
- Transition budgets use complete rollouts, rounded down. The helper rejects an
  extra update at the ceiling; a future training loop must call it before each
  rollout. It does not constrain the existing generic `train.py` automatically.
- `scripts/tools/source_training_preflight.py` creates fresh installed RSL-RL
  5.0.1 runners on CPU for seeds 130001, 130002 and 130003. Each actor matches the
  registered frozen policy exactly on 257 inputs. Critic hashes differ across
  seeds, optimizer state stays empty, iteration/storage remain zero and actor
  loading does not consume the RNG stream.
- The source checkpoint standard deviations are approximately 0.2883 to 0.4236.
  Resetting them to 1.0 is explicit, following the existing candidate, and does
  not imply that stochastic action safety has passed simulator qualification.
- 74 targeted tests passed, including initialization rejection without partial
  mutation, mean parity, trainable gradients, budget enforcement, existing safety
  bridge and parallel terrain allocation regressions. See `focused.junit.xml`.

The CPU fixture has two synthetic environments, 45 policy inputs and 48 critic
inputs. It intentionally uses a different critic input width to demonstrate
that the old critic is not restored. It does not validate live critic dimensions,
the 120-environment candidate, contact physics, safety or training throughput.
No simulator steps or PPO updates were performed by the preflight. A unit test
uses a synthetic gradient step only to check that the loaded actor is trainable.

The first pytest attempt failed during third-party plugin auto-loading because
installed `typeguard` expects `typing_extensions.NoExtraItems`. Tests were run
with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; no installed dependencies were changed.

## Candidate Run Contract

The machine-readable candidate and actual audit results are in
[`initialization_audit.json`](initialization_audit.json). Existing training draft
values are carried forward, not represented as measured optimal parameters.

| Field | Candidate |
| --- | --- |
| Scope | All six families, all 15 L1 directional/layout cells; no automatic L2/L3 |
| Reset family counts per 120 assignments | Flat 12, Rough 24, Slope 24, Stairs 30, Slope+Rough 18, Friction 12 |
| PPO seeds | 130001, 130002, 130003 |
| Generator seeds | 30000, 33000, 36000 |
| Reset state IDs | 40000, 43000, 46000; currently bookkeeping, not randomization |
| Sampler seeds | Candidate independent streams 131001, 131002, 131003; not yet wired |
| Throughput candidate | 120 environments x 24 steps = 2,880 transitions/update |
| Numerics gate | 91 updates, 262,080 transitions (64 below ceiling) |
| Midpoint | 364 updates, 1,048,320 transitions |
| Pilot hard stop per seed | 728 updates, 2,096,640 transitions (512 below ceiling) |
| Maximum across three seeds | 6,289,920 training transitions |

The numerics gate counts toward the pilot budget. Validation/smoke interactions
must be logged separately. There is no automatic extension after a failed pilot.
Mean actor weights come from checkpoint SHA-256
`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`.
Fresh critics/optimizers are required, as is iteration zero. The generic
`runner.load()` default restores all old training state and is unsuitable.

The draft's original stairs/flat seed allocations are preserved. All remaining
L1 cells receive five selection pairs and five final-confirmation pairs in the
audit manifest, for 75 episodes per split per evaluated checkpoint. Geometry
50000-50074/state 60000-60074 are selection candidates; geometry 55000-55074/state
65000-65074 are final-confirmation candidates. These are reservations to verify
before execution, not evidence that different IDs yield distinct terrain/reset
realizations. No validation has run. L2 reservations 51000-51009 are unaffected.

## Next Implementation Order

1. Implement the Source PPO environment wrapper and entrypoint. Keep raw sampled
   actions/log probabilities in PPO storage while executing the SafetyFilter
   output. End and reset only affected environments for expected falls/contact/
   bounds events; preserve their failed episodes as training feedback. Nonfinite
   values, corrupt sensor state, cross-environment contamination and split/budget
   violations stop the run. Validate filter state reset and last-action semantics.
2. Add vector completion/50-step hold and the registered 2,000-step episode
   bound. The current Source env disables timeout and completion lives in the
   single-episode scanner. Decide terminal-vs-truncation semantics explicitly;
   never bootstrap from post-reset observations. Installed RSL-RL's timeout
   shortcut uses the stored current-state value, so terminal-observation handling
   requires an explicit adapter or a documented finite-horizon terminal choice.
3. Override the current diagnostic generator seed 10001 with the registered
   train seed. Wire independent sampler RNG, actual reset/transition counts and
   fresh output directories. Keep initial episode lengths at zero. Sampler state
   restoration is not currently implemented; prohibit automatic interrupted-run
   resume until the full state/remaining-budget contract is implemented.
4. Complete bounded moving-environment isolation and resource/throughput checks
   at the selected scale, reusing G1/G2 evidence where applicable. Old-policy
   terrain completion is not an admission condition for learning; motion checks
   test environmental integrity and reset isolation. Shared bank G2 passed at
   eight environments, not 120; G3/G4 remain unqualified.
5. Finish a source-only validation lock covering all 15 cells. Use the existing
   stairs/flat quality thresholds as the starting proposal for the other families
   without tuning to future results. Specify deterministic repeat interpretation,
   per-cell absolute and paired-retention gates, and checkpoint tie-breaking
   before evaluating candidates. Evaluate only at registered budget nodes, keep
   confirmation out of checkpoint selection, and never use development/sealed.
6. Freeze code/configuration/splits and run the 91-update numerics gate. Only
   verified numerics/runtime behavior permits continuation within the registered
   pilot budget. No previously failed terrain family is removed.

Ordinary task failure and infrastructure failure have different consequences.
The existing single-cell baseline records are retained as learning diagnostics;
9 passes/6 failures across 15 fixed conditions are coverage, not an independent
success-rate estimate. No additional baseline success requirement is introduced.

## Reproduction

Use `/home/tjx/anaconda3/envs/env_isaaclab/bin/python`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests/source_scan/test_source_training.py tests/source_scan/test_bridge.py tests/source_scan/test_parallel_terrain_allocation.py
```

Run `scripts/tools/source_training_preflight.py --checkpoint <registered-path>
--output <new-json-path>` with that interpreter. Existing outputs are refused.
The checkpoint path and per-input code digests are in the audit JSON. This is an
initialization audit command, not a simulator/training command.

Master plan: `/home/tjx/rl_training_main/Lite3_v2.3_Fresh_C1a_20261021.md`, sections
3.1-3.3, SHA-256
`00a8c53ef5863fc747c733766b49ebd9d027d460375b3d03c7f386649d0caf9d`.
Historical training and runtime plans remain available; this package updates
the immediate next action, without changing their archived runtime evidence.
