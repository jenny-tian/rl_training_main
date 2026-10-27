# Lite3 台阶 L1 Source PPO 执行锁定清单

版本：v0.1 draft，2026-10-27
状态：`draft_not_for_run`
执行授权：`false`

本清单把台阶 L1 PPO 训练方案中的待锁字段整理成可审阅的候选配置。带“候选”或“待确认”的值尚未锁定；它们不能直接传给训练入口。当前仿真安全桥、完整指标链路和台阶校准仍有资格阻塞。本文件不授权训练、资格扫描、Development/Sealed 访问或 checkpoint 发布。

## 1. 决策摘要

| 项目 | 当前建议 | 状态 |
|---|---|---|
| 训练范围 | 六类 Source L1；台阶 3 级、约 2 cm 高、0.5 m 踏面；上行/下行分别保留 | 候选；待 Source 资格化 |
| PPO 训练重复 | 3 个不同 PPO RNG seed；共同初始化自 `model_1998.pt` | 候选 |
| Source episode/reset 配比 | 120 次新 episode 分配为一个平衡块（含初始化）：Flat 12、Rough 24、Slope 24、Stairs 30、Slope+Rough 18、Friction 12；Stairs 上/下各 15 | 单环境 sampler 和并行分配离线原型已有；并行运行适配待验证 |
| PPO rollout | `num_envs=120`、`num_steps_per_env=24`，每次更新 2,880 transitions | 候选；吞吐/显存未验证 |
| 每 seed 预算 | 不超过 2,097,152 transitions；先行数值门计入总上限 | 候选；待用户确认 |
| 初始策略 | 只加载 `model_1998.pt` 的策略权重；不假定恢复旧 PPO optimizer 状态 | 候选；需 checkpoint 审计 |
| L1 晋级建议 | 每个 PPO seed、每个方向均达到至少 4/5 完整完成，并通过运动质量与安全门 | 数值候选；非已锁阈值 |
| L2 | 仅在 Source-trained policy 通过双向 L1 后另做可学习性诊断；不随本次训练自动加入 | 保持阻塞 |

第 5 节只给 Stairs 和 Flat retention 列出量化候选门槛。因为 PPO 仍采样其余四类 Source 家族，在为 Rough、Slope、Slope + Rough、Friction Variation 登记完整的 source-only 资格和质量门槛前，本清单不能转成多源 PPO 的可执行 lock。

阶段说明：`num_envs=120` 仅是 Source PPO 基础技能训练的吞吐候选。C1-a 在线适应当前使用单环境、完整回合交互，不因 Source 并行数改变而改成 120 环境；正式实验配置仍须分别锁定。

## 2. 训练与评估种子清单（候选）

以下 ID 均落在准备清单预留 split 内。它们是建议分配，写入正式 lock 前须确认没有被其它任务占用，并核实实际训练器会消费这些 seed。几何 seed 和状态 seed 属不同命名空间；状态 seed 只有在 reset 代码实际使用它时才代表状态随机化。

### PPO RNG 与 Source-train 场景

| Run | PPO/PyTorch/Numpy RNG 根 seed | Source terrain-generator seed | Source initial-state/reset seed |
|---|---:|---:|---:|
| train-1 | 130001 | 30000 | 40000 |
| train-2 | 130002 | 33000 | 43000 |
| train-3 | 130003 | 36000 | 46000 |

每个 run 的其他 RNG 应由该根 seed 以有记录的固定偏移派生；不能让同一进程的 CUDA、CPU、NumPy、环境 reset 和地形生成器意外复用同一随机流。三个训练 run 共用策略初始化 SHA `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`，各自建立全新的 PPO optimizer、scheduler 和训练状态。

### Source-validation：checkpoint selection

| 单元 | Geometry seed | Initial-state seed | Episode 数 |
|---|---|---|---:|
| L1 上行 | 50000–50004 | 60000–60004 | 5 |
| L1 下行 | 50005–50009 | 60005–60009 | 5 |
| Flat 保持 | 50010–50014 | 60010–60014 | 5 |

这组 seed 可在预登记的三个 PPO transition 节点上重复使用做 checkpoint selection；只允许按预登记规则选择 checkpoint，不能更新 PPO 权重。每个方向和 Flat 的失败都要保留。

### Source-validation：final confirmation

| 单元 | Geometry seed | Initial-state seed | Episode 数 |
|---|---|---|---:|
| L1 上行 | 55000–55004 | 65000–65004 | 5 |
| L1 下行 | 55005–55009 | 65005–65009 | 5 |
| Flat 保持 | 55010–55014 | 65010–65014 | 5 |

Final-confirmation split 在 checkpoint selection 完成前封存不读；三个 PPO seed 各自选定一个 checkpoint 后，都用同一 final-confirmation seed 列表评估并逐个报告。不得依据 final-confirmation 的结果再挑选三个模型中的赢家、调整阈值或追加训练。

### L2 后续诊断保留 seed（本次不使用）

| 单元 | Geometry seed | Initial-state seed | Episode 数 |
|---|---|---|---:|
| L2 上行可学习性诊断 | 51000–51004 | 61000–61004 | 5 |
| L2 下行可学习性诊断 | 51005–51009 | 61005–61009 | 5 |

此组只在 L1 复核通过、L2 诊断方案另行锁定并获批准后使用；不能用于 L1 PPO 更新或 L1 checkpoint 选择。

**样本独立性限制：** 规则楼梯的固定几何不会因 geometry seed 改变；现有 Source 准备清单也说明初始状态 seed 目前是 bookkeeping key，当前 reset 是确定的。若 rollout 使用确定性策略且仿真/reset 可复现，同一方向的 5 次重复可能实际只有一个不同轨迹。正式锁定前必须选择并验证下列一种解释：

1. 保持固定几何与确定性部署评估，把重复回合当作可复现性检查，不把 5 个 seed 当作独立地形实例或独立统计样本；或
2. 预先登记 L1 范围内的可审计几何/初态变化，验证 seed 确实改变被评估因素，再按变化后的实例定义解释样本量。

不得仅凭不同 seed ID 宣称台阶几何泛化。Wilson 区间可按现有分类提案计算，但在固定几何/确定性重复下只能作为描述，不能冒充独立实例不确定性。

## 3. 地形抽样与并行环境

候选 Source 配比沿用主方案，按 120 次新 episode/reset 组成一个可复现平衡块：

| 家族/子类 | 每 120 次 reset 数 | 比例 |
|---|---:|---:|
| Flat | 12 | 10% |
| Multi-scale Rough | 24 | 20% |
| Slope：uphill/downhill/cross-left/cross-right | 各 6，共 24 | 20% |
| Stairs：ascending/descending | 各 15，共 30 | 25% |
| Slope + Rough：uphill/downhill/cross-left/cross-right | 6/6/3/3，共 18 | 15% |
| Friction Variation：三种布局 | 各 4，共 12 | 10% |

每块内部使用对应 PPO run 的独立 RNG 做确定性洗牌；新 reset 分配下一项，不因失败、 episode 长短或中途指标重抽。每次更新同时报告家族/方向的 reset 次数和 transition 数，报告实现比例及目标比例的差异。

**采样单位与并行限制：** 地形 `proportion` 只在生成时把子地形静态分配到列，不是 episode/reset 采样概率。reset sampler 目前只支持显式开启的 `num_envs=1` 配置：每次 reset 选取一列并更新该环境的 terrain type、level 和 spawn origin；每 120 次 reset 形成一个包含全部 120 列的一次性平衡块。Sampler 默认关闭，启用时必须显式提供非负 seed。不得把静态 tile 比例写成每回合采样概率。

**并行训练阻塞项：** 主表中的 `num_envs=120` 仍是未验证候选，但与当前单环境 reset sampler 不兼容，不能直接作为可执行配置。多环境下重复设置相同列会把多个机器人映射到同一世界 origin，导致实体重叠；当前实现对 `num_envs != 1` 明确报错。若要保留 120 环境吞吐候选，必须另行设计和验证无碰撞的同时占用/列分配机制、所有方向的 terrain-type 映射、reset 行为和实测 transition 占比，或在锁定方案中正式改为单环境采样单位并重新评估吞吐预算。

**2026-10-27 离线设计进度：** [并行地形分配草案](lite3-source-parallel-terrain-allocation-v0.1-draft-2026-10-27.md)及独立 Python 原型使用每环境一行、15 子类各一列的预留布局；120 环境对应 1,800 tiles，地图容量等权，回合配比仍由 120 张票据控制。原型校验异步 reset 的索引唯一性、非 reset 环境不变及每块方向数量，但尚未接入 importer/reset，也没有验证实际碰撞或资源。地图规模、启动独占 origin、边界终止与状态恢复仍阻塞运行；不能仅将当前 `num_envs` 改成 120。

## 4. PPO 配置与预算（候选）

### 运行器参数

| 参数 | 候选值 | 来源/说明 |
|---|---:|---|
| 初始化 checkpoint | `model_1998.pt`，SHA-256 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c` | 已登记冻结基线；运行时仍需复核 |
| `num_envs` | 120 | 未验证；当前 reset sampler 只支持 1 env。启用 120 env 前需完成不共享 spawn origin 的分配器及 GPU/terrain-origin 吞吐资格 |
| `num_steps_per_env` | 24 | 沿用现有 Lite3 Flat PPO rollout 长度候选 |
| PPO iterations 上限 | 728 | transition 硬上限下向下取整 |
| Actor/Critic hidden dims | `[512, 256, 128]` | 现有 Lite3 PPO 候选配置 |
| Activation | ELU | 现有 Lite3 PPO 候选配置 |
| Initial action noise std | 1.0，log parameterization | 现有 Lite3 PPO 候选配置；需确认训练/部署 action scale |
| Learning rate | `1e-3`, adaptive schedule | 现有 Lite3 PPO 配置候选；需通过数值 Pilot |
| PPO clip / epochs / minibatches | `0.2` / 5 / 4 | 现有 Lite3 PPO 配置候选 |
| `gamma` / `lam` | `0.99` / `0.95` | 现有 Lite3 PPO 配置候选 |
| Desired KL / max grad norm / entropy coef | `0.01` / `1.0` / `0.01` | 现有 Lite3 PPO 配置候选 |
| Empirical normalization | false | 与现有 Lite3 Flat 配置一致；须确认 checkpoint observation contract |
| Safety / termination / observation | 使用已资格化并哈希锁定的 Source 契约 | 当前安全桥尚未资格化，阻塞运行 |

当前 Lite3 Source scene 配置仍设 `scene.num_envs=1`，Lite3 Flat PPO 的 `max_iterations=1500` 不是此处的批准预算；不得直接启动现有配置。训练器需证明只做策略初始化、不恢复不匹配的旧 optimizer，并使用新的、不可覆盖的 run 目录。

### Transition 预算计算

候选每次完整 PPO 更新采集：`120 envs × 24 steps/env = 2,880 transitions`。

| 节点 | 完成 PPO 更新数 | 累计 transitions | 作用 |
|---|---:|---:|---|
| Throughput/numerics gate | 91 | 262,080 | 核对 finite loss/gradient、reset sampler、动作过滤与日志；低于 262,144 候选约 64 transitions。 |
| 中期 Source-only review | 364 | 1,048,320 | 运行 selection split，按锁定规则判定课程是否有可学习反馈。 |
| 每 seed 硬上限 | 728 | 2,096,640 | Pilot 终点，低于 2,097,152 候选上限。 |

因此，262,144 的数值门计入每 seed 上限；三 PPO seed 总 transition 硬上限候选为 `6,289,920`。三个节点均按完整 rollout 结束，不向上补足，不因失败追加预算。若用户希望严格达到主方案候选的准确 transition 数，则须另选可整除的并行环境/rollout 组合并重做吞吐资格。

### 训练停止规则（数值候选）

- **完整性硬停：** 任一 NaN/Inf 观测、动作、loss、gradient 或模型权重；checkpoint/代码/manifest 哈希不一致；访问未登记 split/seed；transition 超额；接触传感器或 termination trace 与锁定契约矛盾。保留当时日志和失败状态，不替换 seed。
- **安全事件：** SafetyFilter emergency、invalid-state 或非足端接触终止时立即结束该环境 episode 并记录。若同一方向的 5 个只读评估 episode 中出现至少 2 次 safety/body-contact termination，暂停该训练阶段并审查，不能把它们改记成普通任务失败。普通跌倒/超界仍是完整失败样本；是否暂停以本候选阈值执行。
- **学习反馈回退：** 若任一方向在两个连续 validation 节点中至少 4/5 episode 的 `completed_step_index=0`，暂停当前 L1 设置，检查 sampler/几何和是否要设计中间课程；不得自动增加预算或删除楼梯家族。
- **预算停机：** 每 seed 不超过 728 updates / 2,096,640 transitions。达到上限仍未过 L1 门槛，则报告未达标，回到方案审查，不自行续训。
- **候选 early stop：** 如果同一 run 在 91 和 364 updates 两个连续节点都通过下节的全部 L1 selection gates，可在 364 updates 后停止该 run，并进入 final-confirmation；两个节点的判断必须只用 selection split。

## 5. L1 晋级与停止阈值（全为待审阅候选）

### L1 checkpoint-selection 与 final-confirmation 门槛

每个训练 RNG seed 独立判定，不把三个模型 seed 合并成一个更大的样本量。checkpoint-selection 与 final-confirmation 都采用上行、下行、Flat 保持分项结果。

| 指标 | 候选通过线 | 说明 |
|---|---:|---|
| 上行完整完成 | 每个训练 seed 至少 4/5 | 完成定义是达到 local x=10.5 m、无安全违规并稳定保持 50 control steps。 |
| 下行完整完成 | 每个训练 seed 至少 4/5 | 单独判定；不能用上行成绩补足。 |
| Flat 保持 | 每个训练 seed 至少 4/5 完整完成 | 同一 Source-validation protocol 下相对冻结基线评估。 |
| 每方向运动质量 | 至少 4/5 episode 同时满足绝对质量上限 | mean velocity error ≤0.20 m/s；RMS ≤0.25 m/s；roll/pitch peak ≤0.40 rad；lateral peak ≤0.30 m；foot-slip contact fraction ≤0.10；SafetyFilter projection fraction ≤0.10。 |
| 每个已完成 episode 的阶梯进度 | `completed_step_index=3` | 未完成 episode 保留其实际逐级进度，部分进度不计完整成功。 |
| 卡阶/卡滞 | 0 次 | 任何事件都作为停审信号，确认指标映射后才可继续。 |
| 前足台阶碰撞 | 每个方向最多 1/5 episode 有事件 | 使用当前 20 N rising-edge detector；阈值/时间窗需 runtime qualification。 |
| SafetyFilter emergency、invalid state、非足端接触终止 | 0 次 | 任一发生即该 checkpoint 不过安全门；诊断事件仍完整记录。 |
| 相对 `model_1998.pt` 的退化候选上限 | 五个配对 episode 的 mean/RMS velocity error、lateral peak、slip fraction、SafetyFilter projection fraction 平均变化各 ≤+0.05；roll/pitch peak 平均变化各 ≤+0.10 rad | 摘自 `source_candidate` 分类提案。这里将“变化”暂按绝对加法差值解释，不是百分比；需在同一 validation seed、同一评估配置上运行冻结基才可计算，且须先审阅此解释。 |

同一训练 seed 必须在 selection split 通过全部门槛才可选为候选；final-confirmation 也须独立通过相同门槛。最终报告逐个列出三组 PPO seed 的结果，不能用 pooled 12/15 替代每 seed 判定。Wilson 95% 区间按准备清单报告；固定几何/确定性复测不声称实例泛化，也不能把形式上的区间当成有效独立样本证据。4/5 是当前提案的小样本操作门槛，置信区间会很宽；它只能作为继续课程的筛选条件，不构成精确的总体成功率估计。

### L1 到 L2 的建议闸门

只有三个 PPO seed 的最终候选均通过双向 L1、Flat 保持和安全门后，才可提交独立 L2 诊断锁定申请。建议 L2 诊断仍按 5 episode/方向：至少 3/5 episode 越过首级台阶，且不得出现 SafetyFilter emergency、invalid state 或非足端接触终止，才算出现初步的 L2 可学习进展信号。它只允许另行审议 L2 训练，不自动晋级、不授权增加 L2 采样权重。

## 6. 执行前阻塞核对

以下每一项完成并记录前，执行状态必须保持 `draft_not_for_run`：

- [ ] 冻结基线 Source qualification scan 已按独立 manifest 完成；扫描 seed 从未用于训练或后续验证。
- [ ] 六个 Source 家族的 L1 几何、完整 episode 和质量/安全 qualification 完成；当前数值门槛仅覆盖楼梯与 Flat，另四家族不能留空。
- [ ] Source 训练环境、terrain-importer 的 120 env 对应关系及 episode/reset sampler 通过实测；六类和方向数量/比例正确，reset 后地形/种子语义清楚。
- [x] 完成 120 环境预留行分配的离线原型及不变量测试；此项只证明索引调度，不替代上一项实测。
- [ ] 上/下行 L1 生成几何、Lite3 步幅/足端净空、3 m 入口、12 m 轨道、10.5 m 完成点和 50-step 保持完成审计。
- [ ] policy/SafetyFilter action bridge、base-height scan、termination 时序、接触映射和 emergency 行为通过完整 episode qualification。
- [ ] L1 指标 collector 在完整上行/下行回合验证：逐级完成、足端净空、碰撞/卡阶、滑移、姿态、侧偏、动作延迟和 reset 捕获均有效。
- [ ] 验证同一 geometry seed 是否真的产生新 L1 几何、initial-state seed 是否影响 reset；据此确定重复 episode 的统计解释。
- [ ] 用 262,080 transitions 的候选 Pilot 检查 120 env × 24 rollout 的吞吐、显存、数值稳定性和准确计数；未通过不得扩展到正式预算。
- [ ] 预先批准第 5 节所有绝对阈值、相对阈值、sample-count/CI 解释和失败处置；不得查看正式结果后再改阈值。
- [ ] 固定并哈希模型、runner、PPO 配置、terrain/reset sampler、观察/动作/normalizer、SafetyFilter、termination、指标和 runtime 环境；当前相关代码仍有未提交改动，不能作为锁定版本。
- [ ] 生成单独 `locked_for_run` manifest，列出三个训练 run seed、所有 source split seed、预算、输出目录、执行代码和 hash；经用户单独明确批准后才可执行。

## 7. 依据文件

- 多源域 L1 训练方案：[lite3-stairs-l1-source-training-plan-v0.1-draft-2026-10-26.md](lite3-stairs-l1-source-training-plan-v0.1-draft-2026-10-26.md)
- 并行分配离线设计：[lite3-source-parallel-terrain-allocation-v0.1-draft-2026-10-27.md](lite3-source-parallel-terrain-allocation-v0.1-draft-2026-10-27.md)
- 总体方案：[Lite3_v2.3_Fresh_C1a_20261021.md](../../../Lite3_v2.3_Fresh_C1a_20261021.md)
- 课程草案：[source_curriculum_manifest_v0.1_draft.json](lite3-source-scan-preparation-v0.2-2026-10-22/source_curriculum_manifest_v0.1_draft.json)
- Source 准备清单：[source_qualification_scan_manifest_v0.2_preparation.json](lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json)
- 分类阈值提案：[classification_proposal.json](lite3-source-scan-preparation-v0.2-2026-10-22/classification_proposal.json)
- 终止契约候选：[termination_config.json](lite3-source-scan-preparation-v0.2-2026-10-22/termination_config.json)
- Safety 配置候选：[safety_config.json](lite3-source-scan-preparation-v0.2-2026-10-22/safety_config.json)
