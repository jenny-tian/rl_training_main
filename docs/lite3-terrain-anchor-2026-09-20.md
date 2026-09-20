# Lite3 fixed terrain teacher pilot — 2026-09-20

The completed context pilot did not establish a stable improvement. Its control
completed 4/20 training episodes and passed 1/4 frozen probes; context completed
5/20 but passed 0/4 frozen probes and fell 6 times (control: 2). Context's final
3/4 transfer result and 3/3 flat retention result belonged to the original e95
recovery checkpoint, not any newly trained context policy.

## Observed failure and hypothesis

Artifacts are under
`artifacts/semester_2026/09/lora_ppo_multiepisode/context_pair_v10_20260919/`.
The context e5/e20 frozen probes stop near world X = 0.59/0.64 m at the rough
terrain entrance; e10 also times out and e15 exits laterally. In the e20 probe's
last 500 rows, executed residual maximum averages 0.05830, identical to the
requested residual maximum, with zero paused/projected/contact rows. Its final
estimated lateral position differs from measured Y by approximately -0.0203 m.
The stalled segments already receive negative rewards (about -1.35/step).
These observations do not support blaming safety suppression, large odometry
drift, or positive idle rewards for this particular stalled probe.

The working hypothesis is cumulative degradation of the verified terrain
behavior while optimizing noisy, changing-policy episodes. The existing source
flat anchor constrains flat observations with zero context; it provides no
explicit reference for terrain observations with nonzero context. This is a
hypothesis to test, not a demonstrated cause. The teacher itself is imperfect
and the added constraint can inhibit useful adaptation.

## Intervention

An optional fixed terrain teacher complements the source flat anchor. For valid
on-policy actor observations, PPO penalizes mean-action MSE to the frozen teacher
and can reject updates above a separate MSE cap. The teacher receives the same
actor inputs, including estimated odometry; no world pose or terrain scan is
introduced. This preserves the requested-action PPO likelihood and all existing
safety, reward, flat-anchor and strict completion rules.

The teacher is explicitly loaded from the experiment's initial e95 checkpoint,
not copied from each episode's resumed training candidate. Legacy 45D teacher
migration zeros its context output. A private frozen copy survives learning
checkpoint restores and rollback. Checkpoint path/hash and settings are recorded
in the supervisor manifest and worker config/checkpoint/summary. Changing the
reference asset or settings prevents a supervisor resume under the old manifest.
Update records include `candidate_terrain_anchor_mse`; a rejected update restores
actor, critic and both Adam states. The cap applies to mean squared action drift
on valid rows of the current rollout, not a global per-state guarantee.

## Bounded comparison

Output root: `artifacts/semester_2026/09/lora_ppo_multiepisode/terrain_anchor_pair_v11_20260920/`.
Both arms use context, start from the same verified checkpoint, run up to 20
episodes with simulation seeds 7 onward on terrain 101, and probe every five
episodes. Control runs on GPU 1 with no terrain anchor. Treatment runs on GPU 3
with terrain coefficient 100 and MSE cap 0.0001. Both retain std 0.03, actor LR
0.00015, critic LR 0.0003, source-flat coefficient 100/cap 0.0001 and the original
success criteria (recent 10 >=80%, >=3 consecutive, no recent falls, passing
frozen probe, accepted learning). These matched seeds are a development pilot,
not evidence of performance across independent training seeds.

Teacher checkpoint:
`seed101_source_v9_std100_anchor100_100ep_20260919_0532/episodes/episode-0095/latest.pt`
relative to the multiepisode artifact root. SHA256:
`f6e34b96b19d455008b3416002e86d251113638b95b9f318fa2f95fd7e4cc80e`.

The arm runner now separately evaluates the actual latest training checkpoint
on all five development layouts (101, 103, 107, 109, 113), regardless of selection.
Physical flat retention seeds 7/8/9 cover both selected and latest checkpoints,
sharing each seed's baseline and deduplicating identical checkpoint hashes.
Reports attach role, checkpoint, hash, and whether it is the original recovery
checkpoint. Selected-model transfer scores cannot stand in for latest-candidate
scores. These remain development evaluations, not held-out test claims.

## Validation

CPU suite: 1453 passed, 7 skipped. Tests cover soft penalty effectiveness,
invalid configuration, fixed teacher preservation across caller mutation and
checkpoint restore, full optimizer/actor/critic restoration on cap rejection,
valid-row masking, legacy-to-context teacher migration, supervisor provenance,
and complete runner flows where a failed latest candidate coexists with a
passing selected recovery model. CPU tests establish implementation behavior;
terrain improvement requires the new physical simulation results.

Launch uses archived committed code and records hashes/commands. Only startup
and initial effective updates are checked interactively; the bounded experiment
then runs unattended, per the user's preference.
