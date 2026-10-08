# Lite3 FlowDAgger implementation

This is a single-action, scripted-expert adaptation of the FlowDAgger mechanism for Lite3. It implements a learned conditional flow, inversion of the discrete decoder, expert intervention at learner-visited states, dataset aggregation, and balanced noise-policy regression. It does not reproduce the original vision/language manipulation experiments, and does not claim human feedback or a newly discovered gait family.

The [completed pilot report](lite3-flowdagger-result-2026-10-03.md) records63/64 adapted versus64/64 unadapted flow completions. That initial run did not establish adaptation benefit. Later v8 passed all fixed development gates but failed independent replication (61/64 versus64/64 comparators). Stable reproducible adaptation benefit remains unestablished; see the [v8 result](lite3-flowdagger-v8-result-2026-10-04.md).

## Version scope

The sections below describe the original v2 pilot unless stated otherwise. Current v8 uses a widened1024/512 adapter initialized from pinned v2, fresh optimizer/buffers, paired full-episode qualification, and50/50 correction/autonomous regression with emitted-action preservation weight25 (lr3e-5, two rounds of1000 updates). The frozen flow foundation is unchanged.

Correction supervision is restricted to lateral error>0.10m, yaw error>5 degrees, or tilt>10 degrees; autonomous preservation uses the complementary region of successful unassisted reference episodes. These regions select training data only. A correction must improve its paired reference episode (RMS by at least10% and5mm without increasing peak or duration beyond5%), or recover a physical failure/deadline. At least32 labels per buffer and two qualified episodes are required to update. Skipped rounds retain buffers and RNG state.

The v8 scripted teacher uses PPO283 with a transformed heading query: actual yaw + atan2(actual lateral displacement,1m). All learner observations remain actual simulator observations. Intervention additionally requires emitted-action disagreement>0.02. This declared geometric teacher uses route localization, not terrain foresight. Deployment uses only the learned adapter plus frozen flow. An untrained widened comparator distinguishes training gains from numerical effects of widening. See the immutable [v8 protocol](lite3_flowdagger_v8_protocol_20261004.json) and [experiment registry](lite3_experiment_registry.md) for later outcomes and replication.

## Recovery experiments (2026-10-05)

The [v9/v10 report](lite3-flowdagger-recovery-2026-10-05.md) records two unsuccessful follow-up candidates; neither is promoted. V9 moves tilt>10-degree states to preservation from successful unassisted episodes, with25% of the autonomous half sampled from recovery states (at least32 rows in each stratum). This optional `tilt_role="preservation"` retains the legacy correction-region semantics by default. V10 adds a weight25 action penalty against the frozen initial widened actor on correction rows. It is soft regularization, not a hard safety constraint or a new teacher label. Both changes are training-only; no reference actor or state fallback is needed at deployment. Failure traces now retain contact/orientation/lateral terms individually.

## Training perturbations (v11/v12)

The [perturbation report](lite3-flowdagger-perturbation-2026-10-05.md) adds deterministic training-only COM velocity impulses with explicit state/observation audits and charging before mutation. V11 failed reliability gates; v12 collected no successful tilt>20-degree recovery samples and correctly skipped all updates. Neither is promoted. Deployment and evaluation have no impulses or extra recovery controller.

## Recovery teacher development

A separately trained [PPO recovery teacher](lite3-recovery-teacher-2026-10-05.md) achieved64/64 on the registered full-course development evaluation (rough mean lateral RMS4.56cm, peak13.77cm). It recovered29/48 held-out kinematic-reset trials versus0/48 baselines, below the80% requirement with two source cases still at zero. It is not integrated into FlowDAgger or promoted as reproducibly stable. The flow and adapter results above remain separate.

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

## 2026-10-07 教师开发复现更新

冻结PPO教师五种子320/320（旧v2为314/320），但原采样独立重训恢复0/48，未复现。固定32恢复槽消融两学习种子分别恢复0→37/48、29→39/48；后者横偏精度保持失败，两者仍有同一来源case零恢复。教师资格和完整收益复现均未通过，FlowDAgger新教师集成尚未启动。详见 [P1复现](lite3-p1-replication-result-2026-10-06.md)、[P2配对消融](lite3-p2-sampling-result-2026-10-06.md)。当前这些成绩属于PPO教师，不能写成FlowDAgger成绩。

2026-10-07倾角课程首组已完成并否决：同seed固定槽控制恢复37/48，课程36/48；普通通关同为64/64但横偏精度退化。教师资格仍未满足，未进行新的FlowDAgger集成。见[课程结果](lite3-curriculum-result-2026-10-07.md)。

2026-10-08按来源固定槽PPO教师实验已完成：困难来源曝光增加但留出恢复仍37/48，普通通关62/64，完整收益与教师资格未过，未启动新FlowDAgger集成。见[动态诊断与来源分配结果](lite3-source-stratified-result-2026-10-08.md)。

2026-10-08受约束短动作搜索找到TRAIN case26的恢复正例，经延迟分组筛选后两个新启动种子均24/24恢复并完成路段，原PPO均0/24。该结果属于固定0.24秒修正＋冻结PPO，横偏仍大；不是新的神经策略，也不计入原留出教师资格，未接入FlowDAgger。见[搜索与跟进报告](lite3-recovery-sequence-search-result-2026-10-08.md)。

随后新采集示范并完成两次48D完整Actor监督微调，case26自主恢复均24/24，不再使用固定前缀。保持损失权重1→10使普通61→64/64，RMS16.01→8.77cm、峰值79.20→41.95cm；普通精度仍劣于同初态原PPO，恢复峰值71.30cm亦超过70cm工程门槛，两版pilot均未通过。原7来源资格验证未触发，未启动新FlowDAgger集成；该监督Actor成绩不算作FlowDAgger更新收益。见[监督迁移与消融报告](lite3-prefix-imitation-result-2026-10-08.md)。

新增普通状态覆盖后，同配置Actor监督拟合在eval2251达到普通64/64、RMS7.91cm、峰值18.03cm，普通精度保持门槛通过；case26自主恢复24/24，但恢复峰值71.62cm仍超过70cm，完整pilot未过，原7来源验证未触发。本轮采用原PPO在候选新访问状态上给保持标签，属于一次监督数据聚合消融，没有flow反演或噪声策略更新，不能算作新的FlowDAgger收益。下一项为带保持约束的PPO回线工程准备，尚未启动。见[普通状态覆盖结果](lite3-retention-coverage-result-2026-10-08.md)。
