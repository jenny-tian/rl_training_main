# Full-policy PPO pilot with emitted-action history

Protocol: `lite3_full_policy_historyfix_protocol_20261002.json`.
Final model SHA256: `18b4eaf02268ea74088a8704775065ef01b966599ad9fbf137e144e0970f8939`.

Training completed 128 PPO updates and 524,288 confirmed controls. The full action-history audit found zero nonterminal emitted-action mismatches. This fixes the compatibility defect, but does not establish task success. The previous incompatible pilot's 524,288 controls remain charged separately.

Frozen evaluation uses original development tracks 101/107, seed 229, five task trials and two flat windows per arm and terrain, matched against the original base. The final checkpoint alone is selected. `docs/analysis/run_full_policy_evaluation.py` maintains durable phase status; `docs/analysis/audit_full_policy_evaluation.py` binds each arm to its own checkpoint hash and checks paired starts, zero updates, zero residual commands, and flat retention. Saved before/after tensors cover the residual actor only; full RSL freezing evidence consists of the execution path and unchanged checkpoint files.

## Preliminary observations while evaluation runs

Candidate terrain 101: 0/5 completions, all lateral violations; progress 6.23, 3.41, 3.50, 3.52, 3.62 m. Both flat windows completed without resets, but lateral MAE was 0.105 and 0.102 m/s. Base terrain 107: 1/5 completions; flat XY MAE 0.01658 and 0.01790 m/s. Final paired audit remains pending.

`artifacts/full_policy_pilot_historyfix_20261002/drift_diagnostic.json` records an offline diagnostic, with no new environment controls. Over the last 16 training updates, mean body velocity was approximately (0.214, 0.0467) m/s. On 4,096 identical candidate-training observations, left/right mirror action RMS was 0.450 for the candidate versus 0.292 for the base. This is descriptive evidence of asymmetry, not a causal result or a selection criterion. Canonical leg pairing is FL/FR and HL/HR; HipX changes sign under sagittal reflection, while HipY/Knee retain sign. URDF joint axes and mirrored joint origins were inspected.

A prospective next experiment should address lateral drift and investigate longer training with mirror regularization. No current runtime source is changed during frozen evaluation. No stability or new-gait qualification is claimed.
