# SAC recovery and PPO rollback development pilots, 2026-09-20

## Evidence and scope

The SAC run `seed101_edge_slip_v2_100ep_v1` stopped after 70 episodes
(25 completions, 30 falls, 197 actor updates and 55,872 critic updates).
Episode 71's worker raised a training-contract mismatch after the code/reward
contract changed. Its stale running state was not evidence of a live process.
Do not resume this experiment under new reward semantics. Low actor update
frequency is a hypothesis to test, not an established explanation for failures.

The completed PPO v11 anchored candidate passed 4/5 development layouts and
3/3 physical flat-retention pairs. Continuing with constant exploration in v12
passed a frozen check at episode 5 but the latest candidate passed only 3/5
layouts. Decaying exploration also ended at 3/5 and did not demonstrate improved
learning stability. The v12 constant arm's flat evaluation stopped at seed 8
because the remote default ground-plane USD could not be loaded; simulator
shutdown reported exit zero without producing summary.json.

## Implementation

The SAC supervisor exposes the worker's existing control-step update schedule
and binds nondefault schedules to the run manifest. The default wall-clock
manifest remains compatible. Development pilots can explicitly skip sealed
evaluation; default behavior remains enabled.

The PPO arm runner exposes frozen-probe interval and failure policy. Evaluations
require both exit zero and a summary. A missing result is retried once in a fresh
directory while archiving the failed logs. Evaluation-only resume requires the
same training arguments and base/initial checkpoint hashes and completed training;
matching cached evaluations are retained. No training or failed attempt is erased.

## Bounded experiments

All launches use an archive of committed scripts/source and record its revision.
The new experiments retain the original success and safety requirements. Neither
pilot is evidence of a statistically reliable improvement or a fair direct
SAC-versus-PPO comparison (their policy architectures differ).

- SAC pair: fresh online replay, shared verified zero-mean actor with flat-only
  prewarmed critics; development terrain 101, simulation seed 7, up to 20 episodes,
  control-step scheduling, critic budget 0.5/transition, actor budget 0.005 versus
  0.02/eligible transition. Flat probes every 10 episodes; drift cap 1e-5,
  residual gains 0.10/0.25/0.25 and rate limit 0.05 remain unchanged. No sealed
  evaluation is used for this tuning pilot. Baseline uses GPU 0; treatment uses
  GPU 1 after the interrupted PPO flat evaluation is repaired.
- PPO rollback pilot: v11 anchored episode 20 initializer and fixed teacher,
  action std 0.03, actor LR 5e-5, terrain coefficient 100/cap 1e-4, source-flat
  coefficient 100/cap 1e-4, up to 20 episodes. Probe every episode and roll back
  failed candidates. This combines probe frequency and rollback changes and is
  not an isolated single-factor comparison. Runs on GPU 3, followed by the same
  five-layout and physical flat-retention evaluations.

Initializer SHA256:
`1d884d5426a22caae851d8bbfbd346bace20ae544741f7bbe986a80a104b9043`.
PPO initializer SHA256:
`9c7c3bb5a80ea8e8a7766be126b25bf9edfe3edea063137f4fec9e5b44bba980`.

Only launch and initial effective execution are checked interactively; bounded
jobs then run unattended as requested. Performance conclusions await results.

## Validation

Online-adaptation CPU suite: 1463 passed, 7 skipped. Coverage includes schedule
forwarding without weakening the drift guard, manifest binding, bounded retries
for exit-zero/no-summary failures, preserved failed logs, completed-evaluation
reuse and rejection of changed arguments/checkpoint hashes. Physical simulation
validation is performed by the bounded experiments above.
