# Lite3 FlowDAgger implementation

This is a single-action, scripted-expert adaptation of the FlowDAgger mechanism for Lite3. It implements a learned conditional flow, inversion of the discrete decoder, expert intervention at learner-visited states, dataset aggregation, and balanced noise-policy regression. It does not reproduce the original vision/language manipulation experiments, and does not claim human feedback or a newly discovered gait family.

The [completed pilot report](lite3-flowdagger-result-2026-10-03.md) records63/64 adapted versus64/64 unadapted flow completions. That initial run did not establish adaptation benefit. Later v8 passed all fixed development gates but failed independent replication (61/64 versus64/64 comparators). Stable reproducible adaptation benefit remains unestablished; see the [v8 result](lite3-flowdagger-v8-result-2026-10-04.md).

## Version scope

The sections below describe the original v2 pilot unless stated otherwise. Current v8 uses a widened1024/512 adapter initialized from pinned v2, fresh optimizer/buffers, paired full-episode qualification, and50/50 correction/autonomous regression with emitted-action preservation weight25 (lr3e-5, two rounds of1000 updates). The frozen flow foundation is unchanged.

Correction supervision is restricted to lateral error>0.10m, yaw error>5 degrees, or tilt>10 degrees; autonomous preservation uses the complementary region of successful unassisted reference episodes. These regions select training data only. A correction must improve its paired reference episode (RMS by at least10% and5mm without increasing peak or duration beyond5%), or recover a physical failure/deadline. At least32 labels per buffer and two qualified episodes are required to update. Skipped rounds retain buffers and RNG state.

The v8 scripted teacher uses PPO283 with a transformed heading query: actual yaw + atan2(actual lateral displacement,1m). All learner observations remain actual simulator observations. Intervention additionally requires emitted-action disagreement>0.02. This declared geometric teacher uses route localization, not terrain foresight. Deployment uses only the learned adapter plus frozen flow. An untrained widened comparator distinguishes training gains from numerical effects of widening. See the immutable [v8 protocol](lite3_flowdagger_v8_protocol_20261004.json) and [experiment registry](lite3_experiment_registry.md) for later outcomes and replication.

## Model and data contract

- Inputs: the existing 48D `sim-localization-v1` observation, including simulator lateral position and yaw. No height scan. Both expert and learner receive the same observation.
- Expert: frozen PPO283, checkpoint SHA256 `5f75f26340f6c98002bb105cd62932c0e6e0542fe12ff936a43cac365867d894`.
- Foundation data: observations from the last64 chunks of that PPO's **training** trajectory; evaluation trajectories are excluded. Every environment stream whose ID is divisible by4 is held out across all chunks, avoiding adjacent-step train/validation splitting within a stream. This is an observation holdout, not new-terrain validation.
- Flow: learned ELU MLP `(48+12+1) →512→256→128→12`; state and time are explicit inputs. Its weights initially copy the PPO mean with zero extra input columns, then undergo6000 flow-matching updates. All parameters freeze before DAgger.
- Distillation: Gaussian shared-noise coupling `a0=z`, `a1=mu_PPO(o)+sigma_PPO*z`, `x_t=(1-t)*a0+t*a1`, target velocity `a1-a0`. The auxiliary center-path target is `v(o,t*mu,t)=mu`. This inherits a **unimodal Gaussian prior**, not a pretrained multimodal gait generator.
- Decoder:8 Euler steps from noise at time0 to requested joint action at time1. Twelve-dimensional noise represents one action at20ms control intervals, not an action chunk.
- Adapter: `48→256→128→12`, bounded by `6*tanh(output/6)`, initialized to zero noise. Only this network changes during DAgger. No PPO/SAC update is used in this phase.

## Corrective learning

For a forward decoder step `x[k+1]=x[k]+h*v(o,x[k],t[k])`, inversion solves

`x[k] = x[k+1] - h*v(o,x[k],t[k])`

by fixed-point iteration, from the last step back to the first. It inverts the actual discrete computation, rather than merely integrating the continuous ODE backward. Convergence is not guaranteed: every label must pass forward reconstruction (maximum requested-action error≤1e-4) and the noise bound. Rejected labels are preserved in the inverse audit and excluded from learning; targets are never silently clipped into the adapter range.

The scripted expert is queried at the learner's current observation. When emitted-action RMS disagreement exceeds0.08 policy units, its action is executed instead of the learner's action. Thus these are states from the mixed, intervened rollout, not entirely unassisted learner trajectories. The expert is not assumed infallible: only episodes that actually complete qualify for replay. Failed episodes remain in the trajectory and interaction ledger.

After an episode completes:

1. Expert-intervened segments contribute `(observation, inverted expert noise)` when inversion passes.
2. Non-intervened segments of successful episodes contribute `(observation, actual learner noise)`; the original noise is retained, not guessed from a clipped action.
3. Training batches draw exactly50% from each aggregated buffer. If either buffer is empty, the runner fails explicitly; it does not substitute unrelated flat data or change the mixture.

Successful non-intervened segments can come from an episode that also needed correction. They must not be described as fully autonomous successful episodes. Every failed episode and rejected inversion still counts toward physical cost.

Inversion targets are the expert's **requested** actions, before clipping. The common Lite3 clipping function is applied once before physics, and the emitted action is checked against the next observation's action history. This avoids treating an irreversible clipping operation as invertible.

## Execution and evaluation

The [foundation protocol](lite3_flowdagger_pilot_protocol_20261003.json) fixes the training data and flow optimization. The corrected immutable [v2 pilot protocol](lite3_flowdagger_v2_protocol_20261003.json) pins that foundation checkpoint and fixes two aggregation rounds,2500 adapter updates per round, and final-checkpoint-only selection. The physical cap is640000 transitions, counting every vector row even after its first episode has ended. No failed output directory can be overwritten or silently resumed.

After training, evaluate the expert, zero-noise frozen flow, and final adapted flow on the same new seed463 initial states, including paired actuator delays. Each uses64 vector slots, the original flat/101/107 course mixture, and at most2000 controls per first episode. This is a development pilot, not a replication of the earlier59/60 result. Lateral deviation is reported explicitly; the existing1.65m body/feet boundary is not a10cm precision-tracking criterion. The terminal root pose is captured after physics and before automatic reset.

The first physical attempt inherited extra zero/one-axis command overrides from the learning environment, despite its nominal0.3m/s range. It was stopped and excluded from task qualification. Raw logs remain intact; the [interruption audit](lite3_flowdagger_v1_interruption_20261003.json) preserves424000 confirmed plus64 uncertain transitions. V2 uses the shared `configure_forward_command`, checks actual command observations on every step, and starts a fresh adapter and buffers. The existing flow foundation is retained. Kit's default SIGINT handler was also replaced by a request flag handled before the next physical transition, allowing Python reports/ledgers to finalize.

Teacher queries/inversion/adapter updates belong to training. At deployment only the noise network and frozen flow decoder generate actions, without teacher takeover. The runner queries the expert additionally for evaluation diagnostics, but those diagnostic actions never drive the flow/adapter evaluation arms.

## Commands

Run from the policy-learning worktree with the existing IsaacLab Python environment. Select a free GPU; GPU3 was used for this pilot.

```bash
export PYTHONPATH=.:source/rl_training
PYTHON=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
$PYTHON scripts/reinforcement_learning/train_lite3_flow.py \
  --protocol docs/lite3_flowdagger_pilot_protocol_20261003.json \
  --output artifacts/flowdagger_20261003/flow-v1 --device cuda:3
$PYTHON scripts/reinforcement_learning/run_lite3_flowdagger.py \
  --protocol docs/lite3_flowdagger_v2_protocol_20261003.json \
  --flow-directory artifacts/flowdagger_20261003/flow-v1 \
  --output artifacts/flowdagger_20261003/pilot-v2 --device cuda:3 --headless
```

These paths belong to a registered run. Do not rerun them or change parameters while its managed physics jobs are active. New experiments require distinct outputs and a separately recorded budget/protocol.

Core code: [flowdagger.py](../source/rl_training/rl_training/online_adaptation/flowdagger.py). The implementation uses PyTorch already installed in this project; it does not import the external FlowDAgger package. Algorithm reference checked against [official implementation, commit3426f24](https://github.com/microsoft/FlowDAgger/tree/3426f246510fd67a9e39c3c08bd5b02d586ad688), especially the per-step fixed-point inverse and dual-buffer loop. Time convention here is noise0→action1, the reverse of pi0's convention.

The deployment checkpoint contains both networks. A compatible controller can load it without the teacher:

```python
import torch
from rl_training.online_adaptation.flowdagger import ConditionalFlow, NoisePolicy
from rl_training.lite3_joint_conventions import clip_lite3_policy_action

checkpoint = torch.load("artifacts/flowdagger_20261003/pilot-v2/final.pt",
                        map_location="cuda:3", weights_only=False)
assert checkpoint["schema"] == "lite3_flowdagger_v1"
flow = ConditionalFlow(**checkpoint["flow_config"]).to("cuda:3")
flow.load_state_dict(checkpoint["flow"])
noise_policy = NoisePolicy(bound=checkpoint["noise_bound"],
                           hidden=tuple(checkpoint.get("noise_hidden", (256, 128)))).to("cuda:3")
noise_policy.load_state_dict(checkpoint["actor"])
flow.eval().requires_grad_(False)
noise_policy.eval().requires_grad_(False)
with torch.inference_mode():
    emitted_action = clip_lite3_policy_action(flow(observation48, noise_policy(observation48)))
# Send emitted_action to the canonical Lite3 joint-action interface and retain
# that same emitted_action in the next observation's action history.
```
