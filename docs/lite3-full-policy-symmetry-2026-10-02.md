# Lite3 full-policy symmetry pilot

Protocol: `docs/lite3_full_policy_symmetry_protocol_20261002.json`.
Training seed233; frozen evaluation seed239. Original-base actor warmstart, fresh critic/optimizer, no previous pilot parameters imported. Same fixed flat/rough mixture and rewards as the corrected pilot; sagittal mirror policy loss coefficient1 and a longer fixed512-update budget. No augmented PPO returns from unmirrored terrain. This compound change does not isolate the effect of symmetry.

## Training completed

- 512 PPO updates;2,097,152 charged and confirmed control transitions;0 uncertain.
- Final checkpoint: `artifacts/full_policy_symmetry_20261002/train/final.pt`.
- Checkpoint SHA256: `fa0d2596f7c224103dd87c5ebc67b6797c4862cc6d3834876f029795c6ba7bf3`.
- Training report SHA256: `7f33681b3d5d360ee2e2ccace799ce1c9eff5c99ac2a429072a94fc804088cc6`.
- Runtime source SHA256: `717b6aa938f65227aa72f00624b126499a11d61e9c0c856be1c3505984ffabf3`.
- Training episodes:643 success,776 failure,214 deadline. These mixed flat/rough training outcomes are not frozen qualification evidence.
- Full2,097,152-row action-history audit:0 actor nonterminal mismatches,0 critic mismatches,0 reset-history mismatches,0 stored-emitted-action mismatches. Maximum nonterminal history error1.72e-7.
- Prior full-policy program expenditure1,079,544 remains separately preserved. Including this training, expenditure is3,176,696 before this pilot's frozen evaluation.

## Frozen evaluation running

Final checkpoint selected by protocol only. Original development tracks101/107; each arm5 task starts capped at2,000 steps and2 flat windows of1,000 steps. Task and flat phase-local starts are matched against the original base. Evaluation cap48,000 controls. No online learning, exploration or replacement starts. Durable execution manifests and per-step ledgers are under `artifacts/full_policy_symmetry_20261002/`.

Do not claim stable rough-terrain gait from the training report. Complete the frozen audit, retain all failures and costs, and require independent learning-run replication and new-development-terrain/gait evidence before overall qualification.
