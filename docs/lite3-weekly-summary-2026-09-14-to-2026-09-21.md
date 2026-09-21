# Lite3 work summary, 2026-09-14 to 2026-09-21

## 2026-09-14 to 2026-09-18

The online-learning boundary and the flat-to-terrain adaptation protocol were
documented. The PPO worker was made auditable, with fixed task/protocol
arguments, forward-command qualification, absorbing-success handling, terminal
failure learning, and strict deadline checks. The command and checkpoint
lineage audits prevented an incomplete or disqualified rollout from being
counted as a success.

The failure audit identified lateral deviation as the dominant historical PPO
failure mode: 116 of 238 failed trials in the v9 review crossed the lateral
boundary. The old actor did not observe cumulative lateral/heading context,
although completion depended on it. Exploration at std 0.10 also produced far
more requested-versus-executed action projection than std 0.03.

## 2026-09-19

PPO gained proprioceptive context and physical flat-retention probes. A fixed
source-flat anchor and then a fixed terrain teacher were added so a candidate
could be rejected or restored when frozen terrain behavior degraded. Paired
context trials showed the anchored arm at 4/5 development layouts and 3/3
flat-retention pairs from its episode-20 latest checkpoint; this became the
current PPO initializer. Exploration decay was then compared with constant
noise. Neither arm established stable improvement; both ended at 3/5 terrain
layouts.

The evaluation supervisor was hardened after a simulator could exit zero while
failing to write a summary. It now requires a valid summary, retries once in a
fresh directory, archives the failed attempt, and can resume only matching
evaluation arguments and checkpoint hashes.

## 2026-09-20

The old SAC run was recovered for analysis. It stopped after 70 episodes with
25 completions, 30 falls, 197 actor updates and 55,872 critic updates. Episode
71 failed the training-contract check because the live code/reward contract had
changed while the old checkpoint was reused. It was not resumed. A clean
49-dimensional proprioceptive initializer with flat-only critic prewarm was
used instead.

The SAC runner gained explicit control-step scheduling and manifest binding,
while retaining the drift cap, residual gains, critic budget and safety checks.
Two 20-episode pilots compared actor budgets 0.005 and 0.02 per eligible
transition. The 0.005 arm finished at 3/20 successes, 7 falls, flat retention
passed, and 3/4 development layouts, but its velocity error worsened from
0.147 to 0.199. The 0.02 arm finished at 2/20 successes, 9 falls, flat
retention failed, and 0/4 development layouts. The higher update rate is
rejected.

PPO was also run for 20 episodes with a frozen probe every episode and rollback
on failure. It recorded 3/20 training successes, 5 falls, 11 probe rollbacks,
and no stable-success window. The selected checkpoint's flat retention passed,
but final terrain validation reached only 1/4 layouts; the latest checkpoint
reached 2/5. This pilot did not establish an improvement.

## 2026-09-21 repair

The SAC supervisor now restores the best frozen-validated checkpoint as the
actual continuation after a failed development probe, updates its checksum
state, and records the rollback. This prevents later episodes from continuing
to optimize a degraded latest checkpoint while the supervisor reports an older
best checkpoint. A regression test covers the copied checkpoint and sidecar.

The change is deliberately conservative: it keeps all safety thresholds and
qualification requirements unchanged. The next bounded SAC pilot will use the
original actor budget with this development-probe rollback enabled; results
will be compared against the clean 0.005 arm above.

The LoRA/PPO episode supervisor was then corrected so a candidate that triggers
a training-time safety violation cannot become a recovery checkpoint merely
because its later frozen probe passes. The probe remains necessary for
promotion, but it no longer erases safety evidence from the training rollout;
the default policy rolls back to the previously verified recovery checkpoint.
This is a supervisor-safety fix and does not claim improved PPO performance.

CPU online-adaptation tests currently pass before the final full-suite run for
this repair. Physical conclusions remain pending that bounded pilot.
