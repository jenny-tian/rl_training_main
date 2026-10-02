# Flat-anchor route-policy pilot

Protocol: `docs/lite3_full_policy_flat_anchor_protocol_20261002.json`. Learning seed283 and frozen evaluation seed293. Same route/symmetry/PPO configuration with flat-only frozen-original action MSE reward weight-2. Fresh original-base initialization and new critic/optimizer; no previously successful learned policy imported. Inference remains a single learned actor with no teacher blending or flat fallback.

Training completed512 updates and2,097,152 charged/confirmed controls. The524,288 flat targets use already charged pre-action training states. Original target model buffers remained unchanged. Final checkpoint SHA256 `5f75f26340f6c98002bb105cd62932c0e6e0542fe12ff936a43cac365867d894`; training report SHA256 `68300e7a640621c2f492922c5fd1eddb28eb9dbf79dbf4e4f035523bd221b382`.

Full action-history audit found0 actor, critic, reset-history and stored-emitted-command mismatches across2,097,152 rows. Full anchor audit recomputed every524,288 flat target from the immutable original actor, verified all recorded masked costs and exact zero rough targets/costs; maximum target numerical error1.19e-6. Audits consumed0 additional environment controls.

Frozen evaluation has started on101/107, seed293,5 task trials and2 flat windows per arm/terrain, maximum48,000 controls. Evaluation lock includes both history and anchor audit hashes. Preserve runtime source until both supervisors finish. Prior failures and costs remain recorded; this new configuration needs a successful pilot and its own independent replication before new-development terrain tests. Training completion and audit success do not establish gait qualification.

## Final frozen result

Candidate5/5 versus base2/5 on101, candidate5/5 versus base0/5 on107. Both paired flat-retention comparisons passed: candidate meanXY0.01511m/s versus base0.01698m/s, full eligible windows and per-axis thresholds passed. All28 trials completed, matched starts and zero-update/checkpoint checks passed. Evaluation38,036 confirmed controls, uncertain0. Cumulative full-policy training/evaluation9,613,671; including registered engineering9,615,359. Supports same-configuration independent replication, not overall goal completion. Descriptive gait summary covers all28 trajectories.

## Reviewable trajectory evidence

`figures/body_paths.png` plots all registered candidate/base task trajectories, with terminal markers and the common lane/finish boundaries. `figures/foot_contacts.png` shows six seconds after rough-section entry for the first registered trial of each arm/terrain. It is descriptive and does not select the best trial or establish a causal gait effect. `figures/manifest.json` binds plotted inputs, generating script and audit hashes. Figures were visually inspected; generation consumed0 environment controls. The plotting tool is `docs/analysis/plot_frozen_policy_evidence.py`.
