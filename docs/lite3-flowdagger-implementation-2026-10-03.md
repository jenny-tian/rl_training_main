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

2026-10-09接续更新：带普通/恢复前段动作奖励保持的PPO配对实验已完成；原横偏惩罚−4与主组−8都未改善，主组普通63/64、RMS13.39cm、峰值35.97cm，case26恢复24/24但峰值75.94cm。未晋升或触发原7来源验证，未进入新FlowDAgger集成，仍保留原监督候选6a470cec…。新增920000步，累计charged27760704/confirmed27760640/历史uncertain64。下一项是直接均值动作保持损失。见[完整PPO配对报告](lite3-recovery-return-ppo-result-2026-10-08.md)。

直接均值保持PPO消融随后完成：普通64/64、RMS8.43cm/峰值19.04cm，恢复24/24但峰值70.62cm仍未通过70cm。它是PPO加监督辅助损失，没有flow反演或噪声策略更新；原7来源验证未触发，新FlowDAgger集成仍未开始。详见[直接均值保持](lite3-mean-anchor-result-2026-10-09.md)。

256轮直接均值保持PPO长度消融已完成：普通64/64、RMS7.04cm/峰值15.84cm，case26恢复24/24但峰值70.46cm仍未过70cm。原7来源资格验证未触发，新FlowDAgger集成继续未启动；该PPO结果不计为flow噪声策略收益。见[长度消融](lite3-mean-anchor-long-result-2026-10-09.md)。

2026-10-09纠正示范目标消融完成：只把correction回放目标换为成功TRAIN执行动作，其余256轮PPO设置不变；普通64/64、RMS6.32cm，但峰值22.16cm超原19.27cm；case26恢复24/24、RMS62.44cm/峰值75.94cm，超70cm。六门槛过四项，新候选不采用，原7来源验证及新FlowDAgger集成未触发。示范验证MSE下降40.2%，恢复后状态的均值变化却为旧对照3.12倍，下一项先查尾段保持不足；新消融尚未启动。已修复参考输出view导致标签替换误改输入的问题，40项测试及全部审计通过。代理未检查首轮失败测试就提前启动的中止成本104576也完整计入；正式训练524288+pilot179200，阶段总808064。累计charged29712576/confirmed29712512/历史uncertain64。见[目标消融结果、两张图及错误记录](lite3-demo-target-result-2026-10-09.md)。

2026-10-09尾段保持消融完成：先核对5681条TRAIN尾段覆盖和辅助梯度，再只加25×尾段均值MSE，原四组系数/分母/采样不变。固定256轮及eval2251结果：普通64/64，但RMS9.89cm/峰值33.10cm；TRAIN case26恢复24/24、RMS61.54cm/峰值73.68cm。原六门槛仅过三项，不采用新候选，不触发原7来源验证或新FlowDAgger集成。离线尾段误差降低45%、实际普通访问状态的均值误差也降低18%，但普通路线退化；env44约−1.18cm/s世界横移在26.58秒累积至−33.10cm，提示应检查持续方向偏差与路线奖励，而非继续堆叠保持损失。42项测试及全部审计通过，无物理重试；新增524288+183936=708224，累计charged30420800/confirmed30420736/历史uncertain64。后续奖励/课程消融尚未启动。见[完整报告与两张图](lite3-tail-anchor-result-2026-10-09.md)。
