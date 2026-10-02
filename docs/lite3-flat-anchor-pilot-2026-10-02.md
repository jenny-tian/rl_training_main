# Flat-anchor route-policy pilot

Protocol: `docs/lite3_full_policy_flat_anchor_protocol_20261002.json`. Learning seed283 and frozen evaluation seed293. Same route/symmetry/PPO configuration with flat-only frozen-original action MSE reward weight-2. Fresh original-base initialization and new critic/optimizer; no previously successful learned policy imported. Inference remains a single learned actor with no teacher blending or flat fallback.

Training completed512 updates and2,097,152 charged/confirmed controls. The524,288 flat targets use already charged pre-action training states. Original target model buffers remained unchanged. Final checkpoint SHA256 `5f75f26340f6c98002bb105cd62932c0e6e0542fe12ff936a43cac365867d894`; training report SHA256 `68300e7a640621c2f492922c5fd1eddb28eb9dbf79dbf4e4f035523bd221b382`.

Full action-history audit found0 actor, critic, reset-history and stored-emitted-command mismatches across2,097,152 rows. Full anchor audit recomputed every524,288 flat target from the immutable original actor, verified all recorded masked costs and exact zero rough targets/costs; maximum target numerical error1.19e-6. Audits consumed0 additional environment controls.

Frozen evaluation has started on101/107, seed293,5 task trials and2 flat windows per arm/terrain, maximum48,000 controls. Evaluation lock includes both history and anchor audit hashes. Preserve runtime source until both supervisors finish. Prior failures and costs remain recorded; this new configuration needs a successful pilot and its own independent replication before new-development terrain tests. Training completion and audit success do not establish gait qualification.
