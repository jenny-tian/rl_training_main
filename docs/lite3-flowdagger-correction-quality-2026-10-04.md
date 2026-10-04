# FlowDAgger correction quality and continual preservation

The first FlowDAgger pilot did not establish an adaptation benefit. This change makes the corrective data criterion explicit and adds preservation of already learned behavior. It does not assume that an expert action is better merely because it differs from the student's action.

## Paired corrective rollouts

For every collection round, run the current student twice from the same registered initial state and actuator delays:

1. An unassisted reference, with no teacher takeover.
2. The same student with the existing action-disagreement-triggered expert takeover.

No simulator snapshots or free counterfactual steps are used. Both complete trajectories, including inactive vector padding and failures, are charged. Check the observation/root-state prefix up to the first intervention (maximum absolute discrepancy≤1e-4). A prefix mismatch cannot provide a qualified corrective label.

A corrected episode must complete. It qualifies if it recovers an actual reference failure/deadline, or if both complete and all of the following hold:

- Body lateral RMS improves by at least10% and5mm.
- The maximum body lateral deviation does not increase.
- Episode duration increases by no more than5%.

A reference collector cutoff is not counted as a demonstrated task failure. These criteria establish a benefit for the **intervention sequence**, not a causal benefit for every individual action in that sequence.

Only intervention steps from qualifying episodes undergo inverse-label admission. The independent unassisted-success trajectories populate the autonomous buffer. Sampling remains50/50. At least32 rows in each buffer and two qualifying episodes are required before optimization. Insufficient data is explicitly recorded as a skipped update; unchanged weights are never described as successful new learning.

## Continual adaptation and preservation

V3 began with a zero-initialized adapter. V4/v5 instead initialize from the pinned prior v2 adapter, with a fresh optimizer and buffers. They do not import v3/v4 labels or evaluation trajectories. The flow stays frozen.

The loss is the existing mixed noise regression plus an emitted-action preservation term on the autonomous half:

`L = MSE(z_pred,z_label) + 25*MSE(clip(G(o,z_pred)),clip(G(o,z_actual)))_autonomous`

Gradients through the frozen flow reach the adapter; targets and flow weights remain fixed. Learning rate is3e-5 and each of two rounds has at most1000 updates. The actual generated noise of successful reference actions is retained, rather than reconstructed from a clipped action.

Skipped rounds now also save qualified buffers, episode IDs, model/optimizer state and RNG. This fixes loss of accumulated data when no optimizer update took place. It does not add an automatic resume/retry path to managed experiments.

## Recorded experiments

- [Read-only diagnosis](lite3_flowdagger_correction_diagnosis_20261003.json): the old replay can measure fitting errors, but cannot establish the benefit of an unobserved counterfactual action.
- [V3 protocol](lite3_flowdagger_v3_protocol_20261004.json), [audited result](lite3_flowdagger_v3_result_20261004.json):17 qualifying sequences,115 accepted labels,5000 updates,733184 physical controls. On paired seed491, new v3 completed61/64, prior v2 completed64/64, and unadapted flow61/64. V3 did not qualify for promotion or replication.
- [V4 protocol](lite3_flowdagger_v4_protocol_20261004.json), [audited result](lite3_flowdagger_v4_result_20261004.json):5 qualifying sequences but only11 accepted labels; both updates skipped,735232 controls. The candidate exactly retained v2 weights and its63/64 result on seed509. This is a null learning result, not an improvement.
- [V5 protocol](lite3_flowdagger_v5_protocol_20261004.json): use independently trained PPO307 as the candidate corrective expert, retaining all v4 qualification, initialization and loss settings. The teacher must still demonstrate paired benefit. Its status and full raw evidence live under `artifacts/flowdagger_20261004/pilot-v5/`.

V4/v5 incremental acceptance additionally requires no per-terrain completion regression versus both v2 and unadapted flow, at least5% lower rough RMS than flow **and**2% lower than inherited v2, and no higher worst rough deviation than either comparator. Changed weights and actual updates are required. Passing one development run would still require separately registered independent continual-learning replication.

## Audit and execution

The existing runner accepts the new protocol options. For v5:

```bash
PYTHONPATH=.:source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/run_lite3_flowdagger.py \
  --protocol docs/lite3_flowdagger_v5_protocol_20261004.json \
  --flow-directory artifacts/flowdagger_20261003/flow-v1 \
  --output artifacts/flowdagger_20261004/pilot-v5 --headless --device cuda:3
```

Registered output directories cannot be overwritten. Runtime source and protocol must remain fixed during each physical run. `docs/analysis/audit_lite3_flowdagger.py` reconstructs every transition, paired qualification decision and inverse-label eligibility, and replays sampled deployed actions without loading a teacher. `docs/analysis/summarize_flowdagger_improvement.py` evaluates the declared gates without selecting checkpoints.

The39 relevant tests cover the decoder/inverse, failed/prefix-mismatched/nonbeneficial correction rejection, recovery versus cutoff semantics, action preservation gradients, skipped-checkpoint retention, fixed commands, action history, and physical-step accounting. All work remains simulation-only with explicit route localization, a Gaussian-distilled flow prior and single-action decoding.
