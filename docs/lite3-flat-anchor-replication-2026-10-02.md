# Independent flat-anchor replication

Protocol: `docs/lite3_full_policy_flat_anchor_replication_protocol_20261002.json`. Learning seed307 and evaluation seed311; same configuration as seed283, fresh original-base actor/critic/optimizer initialization. No previously learned policy imported. Frozen original-policy targets are used only during flat training; deployed execution remains a single actor.

Training completed512 updates,2,097,152 charged/confirmed controls and524,288 flat labels. Target buffers remained unchanged. Final checkpoint SHA256 `05f5fd60e291aae2d919c5f5836c040af6a60292e700c390f9bf867781ce020d`; training report SHA256 `9754428eb8af7de9feb2f93219654361776c65535b0baa1b296fba1eb3606f5d`.

Full action-history audit:0 actor/critic/reset/stored-command mismatches. Full anchor audit recomputed every target and checked every masked cost, with maximum target error9.54e-7 and0 additional environment controls. Both audits are locked before evaluation.

Frozen paired evaluation has launched on101/107 with new seed311 starts,5 task trials and2 flat windows per arm/terrain, cap48,000 controls. Acceptance requires repeated paired gains and flat retention, aggregate>=9/10 per terrain across the two anchored policies with each>=4/5, then registered new-development terrain validation. Prior unanchored failure is preserved and excluded from this configuration's qualification. Goal remains incomplete.
