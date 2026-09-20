# Lite3 exploration decay continuation — 2026-09-20

## Evidence and next hypothesis

Both v11 arms completed 20 training episodes with five successes. The fixed
terrain teacher arm fell twice versus six times in the control, and passed its
fourth deterministic probe (the control passed none of four). The successful
probe took 1997 of the allowed 2000 steps, so it has little completion margin.
Transfer and flat retention evaluations were still running when this next
experiment was prepared. This is a promising development observation, not a
stable improvement or a claim that all retention gates passed.

The mean policy can still differ materially from the noisy, changing policy
used in online training. Next we test gradually reducing exploration, while
continuing from the newly probe-verified e20 model and using smaller actor steps.
This tests an exploration mismatch hypothesis; it does not establish noise as
the cause of earlier failures.

## Paired experiment

Root: `artifacts/semester_2026/09/lora_ppo_multiepisode/exploration_decay_pair_v12_20260920/`.

Both arms start from
`terrain_anchor_pair_v11_20260920/anchored/training/episodes/episode-0020/latest.pt`
relative to the multiepisode artifact root, SHA256
`9c7c3bb5a80ea8e8a7766be126b25bf9edfe3edea063137f4fec9e5b44bba980`.
This same checkpoint is the fixed terrain teacher throughout the new experiment.
It passed the strict terrain-101 frozen probe; starting another simulation trial
does not promote it as a fully validated deployment model.

- `constant`, GPU 1: fixed action standard deviation 0.03.
- `decay`, GPU 3: standard deviation falls geometrically from 0.03 in episode 1
  to 0.01 in episode 20.
- Both: actor LR 0.00005, critic LR 0.0003, context enabled, terrain coefficient
  100/cap 0.0001, source-flat coefficient 100/cap 0.0001, maximum 20 episodes,
  2000-step strict deadline, probe every five episodes.

The pair differs only in the exploration schedule. Both use a lower actor LR
than v11 (0.00015); cross-generation comparisons therefore cannot attribute
changes solely to noise. Simulation seeds 7 onward and development terrain 101
are reused, so this remains development tuning, not independent confirmation.
The runner separately evaluates the latest trained model on all five development
layouts and performs physical flat retention on latest and selected models.
Original safety limits and stability/completion criteria remain in effect.

## Resume and queue behavior

The supervisor computes noise from the absolute episode index and records the
schedule in its immutable run manifest. Changing endpoints or duration rejects
a resume. Noise changes only between episode workers, before rollout collection.
The existing runtime configuration override applies it after restoring the
checkpoint, updates active and rollback distributions, preserves actor means and
Adam moments, advances policy version when needed, and republishes the actor.
Frozen probes still execute deterministic means without updates.

Arm CLI exposes initial/final noise, actor LR and episode budget. Optional
`--wait-for-gpu` records `waiting_for_gpu` and blocks on the same exclusive GPU
lock used by the preceding experiment. It does not poll results or interrupt
existing evaluations. The arm owns its output lock while queued and the GPU lock
through all training and evaluations. Defaults retain the previous fixed-noise,
nonblocking behavior.

## Validation

Regression coverage includes endpoint/monotonic decay, rebuilding an episode-12
command on resume, rejecting schedule changes and invalid one-episode decay,
applying scheduled noise to a restored actor and rollback distribution without
changing means or Adam moments, and forwarding arm options/recording queue state
before lock acquisition. Full online-adaptation CPU suite: 1458 passed, 7 skipped.
Simulation outcomes are pending and must be checked separately.
