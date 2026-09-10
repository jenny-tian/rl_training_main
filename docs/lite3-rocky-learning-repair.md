# RockyRoad learning repair (2026-09-10)

The v11 run completed 100 episodes but did not establish reliable learning:
46/100 online passes, 35 falls, and 7/20 frozen benchmark passes versus 9/20
for the baseline. Those 20 seeds were already observed; they are a reused
benchmark, not a blind test. A higher actor version is not a gait improvement.

## Changes

- The RockyRoad runner now starts entropy temperature at 0.02 and caps it at
  0.05. With 12 normalized tanh actions the maximum differential entropy is
  `12 * log(2)`, so the maximum expected entropy bonus at the cap is about
  0.416 per step. Entropy stays in normalized action coordinates. The runner
  no longer defaults to clipping targets at 300; existing finite loss, TD-error
  and gradient guards remain. Logs expose raw target mean, clipping fraction
  and entropy bonus. These are experimental defaults, not a convergence guarantee.
- The runner uses Huber critic loss (delta 1). A single terminal transition
  with TD error 300 in a batch of 128 makes twin squared loss about 1406,
  exceeding the previous 800 rejection threshold. Huber loss allows learning
  from such a failure instead of repeatedly rejecting the whole batch.
- Rough-terrain swing shaping no longer uses the step-edge-only reward. Each
  foot's last observed support contact supplies its height reference. The
  reward peaks at 4.5 cm lift, requires forward progress and another support
  foot, and does not reward stationary hopping or arbitrarily high lift.
  Its gate uses online novelty adaptation on rough stages, without exact terrain
  geometry. It measures lift from the previous support surface, **not** clearance
  above the unseen terrain ahead. Invalid/unobserved contact references earn no
  lift credit; references reset at physical environment resets.
- Per-step logs include support-relative foot lift, contact states and physical
  joint-target residuals alongside the stochastic-control flag. These allow
  deterministic gait changes to be distinguished from exploration.
- Every flat-probe interval (10 episodes by default), and at the final episode,
  the current model is frozen and evaluated on the four development seeds other
  than the training seed. Each has a paired frozen base control. These probes do
  not update replay or policy. Stable-success reporting now also requires current
  deterministic validation to pass and improve the success count over the base.
- `experiment.pt` remains the latest continuation state. `best_checkpoint.json`
  points to an immutable episode checkpoint, selected only among flat-retention
  passes by success count, falls, remaining distance, velocity error and tilt.
  Exact ties keep the earlier checkpoint. Final benchmark evaluation uses this
  selected checkpoint; it cannot silently use a worse final policy. Selection
  does not guarantee outperforming the baseline; paired results remain explicit.
- Snapshot hashes, frozen rollout metadata and cache provenance are checked.
  A restart between flat and rough validation completes the missing probe before
  proceeding. Reward/SAC contracts are embedded in new experiment checkpoints;
  training rejects missing or changed contracts. Historical compatible models
  can still be evaluated frozen. Old run directories and replay are preserved.

## Running the corrected experiment

Create a new initializer using `initialize_residual_sac.py` with the same frozen
flat replay, `--frame-dim 49 --initial-alpha 0.02 --alpha-max 0.05` and a new output
directory. The initializer manifest must match the runner temperature settings.
It retains a zero-mean actor and prewarms only the critics from flat replay.

Use the existing verified base checkpoint/manifest, flat replay, novelty baseline
and frozen benchmark as inputs to `run_rocky_road_multiepisode.py`, with the new
initializer and a **new** output directory. The corrected runner defaults to
100 training episodes, 2000 steps each, gains 0.10/0.25/0.25, rate limit 0.05,
initial alpha 0.02, alpha cap 0.05, Huber delta 1, and rough swing weight 1.
Do not reuse the old launcher flags `--target-q-min -500 --target-q-max 300`.
Do not resume v11's replay/critics under the changed objective.

Validation rollouts are additional evaluation interactions and are not counted
as training episodes. A short smoke test verifies execution and restore paths;
only a new bounded training run and frozen evaluation can establish actual gait
learning or improvement within 100 episodes.

## Verification

The complete `tests/online_adaptation` suite passed: **1073 passed, 7 skipped**.
It ran with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2
PYTHONPATH=source/rl_training:.` in the `env_isaaclab` environment. Plugin
autoload was disabled because the installed third-party typeguard plugin has
an unrelated typing_extensions incompatibility. Compilation and
`git diff --check` also passed.

The GPU 1 smoke test used a fresh initializer with 1000 flat critic warmup
updates. Its first 800-step rollout completed 8 actor and 214 critic updates;
the rough-swing gate was active for 286 steps, with a positive rough-swing
reward independently reconstructed from logged signals on 215 steps. Across
the eight actor-update records, target clipping fraction was zero, mean
entropy bonus was about 0.159, and mean Q target was about 14.73.
It did not fall and did not finish
the track within this short horizon. Restoring that checkpoint for 200 steps
retained actor version 8 and completed another 87 critic updates. A separate
200-step frozen rollout on development seed 103 performed zero actor/critic
updates and retained identical actor parameters.

These checks exercise simulator reward wiring, checkpoint handoff and frozen
evaluation. They are not success-rate or gait-learning evidence. The records
are archived under
`artifacts/semester_2026/09/rocky_road_online/v12_fix_validation_20260910/`.
