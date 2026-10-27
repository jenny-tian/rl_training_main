# Lite3 多级台阶 L1 Source PPO 技能训练方案

版本：v0.1 draft，2026-10-26

状态：`draft_not_for_run`

执行授权：`false`

本文为 Lite3 多源域 PPO 方案中的多级台阶 L1 技能训练子方案。它不授权 PPO 更新、仿真训练、Source 资格扫描、Development/Sealed 访问或 checkpoint 发布。数值候选、训练 seed 清单、代码/环境哈希和晋级阈值须在单独的执行锁定文件中登记并经用户确认后，才可运行。

## 1. 目标与边界

目标是在 Source PPO 中学习连续上、下多级台阶所需的摆动足抬升、足端落点、支撑转换和连续高度变化控制，同时维持平地基础步态。台阶是预先声明的 Source 技能家族；冻结平地策略未通过某级台阶，不构成删除该技能家族的理由。

本阶段只训练台阶 L1，不训练 L2/L3。台阶 L1 的当前候选几何为 3 级、单级高度约 0.02 m、踏面深度 0.5 m；上行和下行分别采样并平衡。几何数值仍是准备清单中的候选，需在执行前确认实际生成几何、步幅/足端间隙校准和安全配置。

本方案定位为六类 Source Terrain PPO 课程中的台阶家族专项定义，不把它改成独立的楼梯专用策略训练。建议保持主方案的 episode/reset 抽样配比：Flat 10%、Multi-scale Rough 20%、Slope 20%、Stairs 25%、Slope + Rough 15%、Friction Variation 10%。Stairs 家族内部上、下行各占一半，对应总 Source episode/reset 抽样的候选 12.5%/12.5%。同时报告 episode/reset 抽样占比和实际 transition 占比。

上述配比只有在训练环境确实按 episode/reset 对地形家族重新采样时，才能称为 episode/reset 概率。当前 Isaac terrain 配置中的 sub-terrain `proportion` 可能只控制生成地形 tile 或并行环境的分配；执行前必须检查实际训练调度。如果每个并行环境固定绑定一个地形，需预先选择并资格化跨 reset 家族调度，或准确报告固定环境分配及其实际 transition 权重，不能把静态 tile 比例描述成每回合地形采样概率。

训练先从 `model_1998.pt` 的策略权重开始；每个 PPO 训练 seed 使用独立 RNG 与独立 PPO 优化器状态。该设计评估共同初始化下的训练稳定性，不代表独立随机初始化泛化。checkpoint 是否包含可恢复 PPO 优化器状态，须先审计；没有证据时只把它当作策略初始化权重。

## 2. 已有诊断如何约束课程

现有 L1 运行报告各只记录了一个 episode，虽然候选诊断 cell 预留了五组 seed，因此它们是单样本冻结策略诊断，不是五 seed 资格结果，也没有学习到新技能：

| 方向 | 已记录的诊断结果 | 对训练方案的含义 |
|---|---|---|
| 上行 | 3/3 级通过，前进约 8.58 m；未完成全程，在 1600 步诊断上限结束；没有记录到前足台阶碰撞或卡滞 | 保留为 L1 学习目标；训练和复核要覆盖全程完成与侧向漂移，不能只以“越过三阶”为目标。 |
| 下行 | 3/3 级通过且完成轨道；记录到 1 次前足台阶碰撞，无卡滞 | 保留为独立方向目标；除完成外，持续监测碰撞和摆动足净空。 |
| L2 上行 | 五 seed bounded diagnostic 均未越过首级台阶；2 次姿态跌倒、3 次达到诊断步数上限 | L2 晋级保持阻塞；先训练并验证 L1 双向技能，再另行制定 L2 可学习性诊断。该结果不删除 Stairs 家族。 |

冻结诊断使用的 checkpoint SHA 为 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。这些记录仅用于课程设计，不可替代后续 Source 训练或 Source-validation 证据。

## 3. 建议训练阶段

**准备与资格门。** 先固定 Source 训练环境、完整 12 m 赛道、3 m 入口平地、3×2 cm×0.5 m 台阶几何、10.5 m 完成点、50 控制步稳定保持、0.3 m/s 速度命令、终止规则和 SafetyFilter。核对台阶踏面/步幅、足端净空、接触传感器映射、结束点后的缓冲、指标采集与复位。此门未通过时不启动 PPO。

**技能获取。** 用 `model_1998.pt` 初始化的 Source PPO 训练六类 L1 候选；保持候选家族配比，Stairs 内部上、下行等量采样。禁止以单方向表现代替双方向训练。第一轮不加入未单独标定的控制延迟、外力推扰或额外动力学随机化；后续加入时须先登记其范围并独立资格化。

**L1 双向巩固。** 继续在相同 L1 候选内训练，同时保留 Flat 和其他已登记 Source L1 家族。周期性保存只用于恢复的训练 checkpoint；固定评估节点使用未参与梯度更新的 Source-validation seeds，分别评估上行、下行和平地保持。checkpoint 选择规则须在开跑前登记，不能按单一上行奖励挑选模型。

**L2 后续复评。** 只有 Source-trained checkpoint 在双向 L1 的独立 Source validation 达到预先锁定的完成、安全和运动质量标准后，才提出 L2 复评计划。L2 首次动作应是受限的可学习性诊断，记录逐级进度、第一阶接触/碰撞、摆动足高度、姿态、命令跟踪和安全过滤；该诊断不自动等于晋级或扩大 L2 训练占比。若 L2 仍近零进度，先检查是否需要注册并审计中间难度，不能据此移除楼梯技能目标。

## 4. Seed 与数据隔离

使用准备清单已经划定的 split 区间，不在本草案里分配具体 seed ID：

| Split | Geometry seed 区间 | Initial-state seed 区间 | 用途 |
|---|---:|---:|---|
| 已有资格扫描 | 10000–19999 | 20000–29999 | 只读基线/环境诊断；不得进入 PPO 更新。 |
| Source train | 30000–39999 | 40000–49999 | PPO rollout 与梯度更新。 |
| Source validation：checkpoint selection | 50000–54999 | 60000–64999 | 固定节点只读验证和预登记的 checkpoint 选择。 |
| Source validation：final confirmation | 55000–59999 | 65000–69999 | 最终候选的一次性确认，不回流训练，也不用于在多个候选 checkpoint 中再次挑选。 |
| Development | 70000–79999 | 80000–89999 | 保持隔离；本方案不访问。 |
| Sealed | 90000–99999 | 100000–109999 | 保持隔离；本方案不访问。 |

上表把准备清单的 Source-validation 保留区间进一步建议拆成两个互斥子区间；具体 seed ID 仍须由 lock 文件登记。每个训练 split 内为各训练 seed 和方向分配不重叠子集，并登记配对规则。不得用资格扫描中的 10180–10184/20180–20184、10190–10194/20190–20194 或任何诊断 seed 做更新或 validation。

当前规则楼梯的固定几何不会因更换 geometry seed 自动改变；准备清单也明确 regular stairs 的不同 seed 是重复试验而非新几何实例。因此，seed 隔离可防止训练与评估 episode 重用，但不能单独证明几何泛化。若目标包含踏面、入口或布局变化，须另行预注册 L1 内变化范围、审计生成结果，并确保 train/validation 的几何实例分离。初始状态 seed 也不等于随机化初态；须审计 reset 实现后才能作此主张。

## 5. PPO 训练预算候选

主方案给出的 Source Pilot 候选为：先做 262,144 transitions 的吞吐/数值检查；每个训练 seed 的 Pilot 硬上限为 2,097,152 transitions。本文建议把前述 262,144 计入每 seed 的 2,097,152 上限，避免形成未登记的额外训练预算；此记账方式仍须在正式 lock 时确认。

建议用 3 个独立 PPO RNG seed 评估共同 `model_1998.pt` 初始化下训练结果的稳定性。每个 seed 最多 2,097,152 transitions，三 seed 总硬上限候选为 6,291,456 transitions。262,144-transition 检查是每 seed 的先行阶段；未通过运行完整性检查时停止相应 run，不把失败预算转成临时追加训练。训练 seed 数和所有预算当前仍为候选，尚未授权。

候选预算节点为每 seed 262,144、1,048,576 和 2,097,152 transitions：首节点检查训练管线与数值完整性，中间节点做 Source-only 课程诊断，末节点达到 Pilot 硬上限。技术安全故障之外，不在看到中途验证表现后临时追加预算、改采样配比或移动节点。节点和相应评估次数仍须锁定。

预算按实际环境 transition 计数，不用 PPO iteration 数替代：

`transitions = num_envs × num_steps_per_env × completed_rollout_updates`

锁定 `num_envs`、`num_steps_per_env` 和更新次数后，runner 必须证明不会越过每 seed 上限；按完整 rollout 停止时，如无法刚好命中预算，记录实际 transition 数且不得超过上限。现有 Lite3 PPO 配置中的 `num_steps_per_env=24`、平地配置 `max_iterations=1500` 以及 Source env 配置的 `scene.num_envs=1` 都不能直接视为本方案的运行值。学习率、优化器、并行环境数、rollout 长度、保存/评估间隔和学习率调度均待资格化并写入 lock 文件。

本预算表不自动允许从 2.1M 延长到 4.2M 或 8.4M/seed；任何后续增量都须单独预注册、审阅并授权，不能因当前表现不理想临时续训。

## 6. 评估指标与候选评估流程

每个 seed、每个方向分别报告，不把方向或训练 seed 合并掩盖差异。至少记录：

- 全程完成、终止原因、步数、前进距离、逐级完成数与每级进度；
- 线速度均值/RMS 跟踪误差、roll/pitch 均值/RMS/峰值、侧向偏移均值/峰值；
- 摆动足净空、前足台阶碰撞数、卡阶/卡滞数、足端滑移和非足端接触；
- SafetyFilter 投影比例/原因、拒绝或紧急停止数、无效状态数及动作计算/执行延迟；
- 方向、台阶数/高度/踏面、轨道与入口/出口几何审计数据、地形/初态 seed、模型和环境代码哈希。

checkpoint-selection Source-validation 在预登记的训练节点执行；每个方向候选 5 个 episode，另对 Flat 保持域按准备清单协议评估。最后只对按事先规则选定的候选运行 final-confirmation split，每个方向候选 5 个 episode；这组结果不得再用于在 checkpoint 之间选模型或调整阈值。固定几何 seed 的规则楼梯重复试验应按 repeated trials 汇报，不能声称独立几何实例泛化。完整成功仍依照 10.5 m 完成点、无安全违规及 50 控制步稳定保持；部分前进不算成功。

晋级与选模需同时看双向完整完成、速度与姿态质量、侧向偏移、足端碰撞/卡滞/滑移、平地保持及 SafetyFilter；不能只最大化 reward 或首级台阶进度。所有数值上下限、置信区间方法、最小样本数、评估节点及模型选择规则目前均为 `TBD`，须在任何 PPO 更新前锁定。未锁定前只能讨论候选结果，不得称 L1 已资格或 L2 可晋级。

## 7. 停止、回退与检查点规则

**立即终止该 run 并保留日志：** 出现非有限观测/动作/权重、checkpoint/hash 不一致、超出预算、未登记的 split/seed 访问、环境终止与接触传感器证据冲突、SafetyFilter 紧急停止或其他预注册安全终止。不得跳过失败 episode 或用重跑替换不利 seed。

**按预算停止：** 达到本 seed 的 transition 硬上限即停止更新，不追加样本。除预先登记的安全/运行完整性停止外，不因单次失败或短期 reward 波动临时提前截断不利训练段；到达封闭预算仍未满足预先登记的 L1 条件，就报告未达标并回到方案审查。

**回退课程：** 若 L1 的某方向持续没有可学习的足端进度，或碰撞、姿态、侧向漂移或安全指标恶化，保持 Stairs 家族不变，降低该具体方向/几何难度或提出经审计的中间课程；不得通过提高更新预算、复用 validation seed 或直接删除该家族来处理。

**检查点：** 所有训练检查点写入新的、不可覆盖的 run 目录，记录 PPO RNG seed、输入 checkpoint SHA、当前 transition 数、PPO/环境配置和代码哈希。只按预登记的 Source-validation 规则选择最终候选；Development/Sealed 不能用于训练预算、课程或 checkpoint 选择。发布或锁定 source checkpoint 需要独立授权。

## 8. 开跑前必须补齐的 lock 项

1. 完成主方案要求的冻结基线 Source 扫描和 Source 环境资格化；完成双向 L1 几何、Lite3 步幅/净空、接触/终止映射、SafetyFilter 和全程指标检查。当前准备清单仍列有楼梯校准、策略安全桥接和全条件运行资格阻塞。
2. 固定 3 个（或另行批准的数量）PPO RNG seed 与互斥 train/validation seed 清单；说明固定几何 repeated trials 的解释边界。
3. 确认 262,144 是否计入每 seed 上限、是否采用 2,097,152/seed 硬上限及三 seed 总预算。
4. 固定 PPO runner、`num_envs`、rollout 长度、优化器、学习率、评估/保存间隔、家族/方向抽样实现和确定性的 checkpoint 选择规则；不能仅照搬现有 Flat PPO iteration 配置。
5. 预先登记双向 L1、平地保持和安全资格的绝对阈值、相对 `model_1998.pt` 的退化上限、统计不确定性和失败样本规则。
6. 固定地形生成器、环境配置、模型、观测/动作/归一化、SafetyFilter、reset、终止和指标实现的内容哈希；本草案对应工作区仍有未提交代码改动，因此当前不绑定执行哈希。
7. 稳定并提交本方案所依赖的运行代码，重新绑定环境、训练入口与全部接口哈希；当前工作区有未提交源代码和准备清单改动，不能直接作为冻结执行版本。
8. 单独形成运行 manifest 并由用户明确批准。批准该草案不等于批准运行。

## 9. 参考文件

- 主方案：[Lite3_v2.3_Fresh_C1a_20261021.md](../../../Lite3_v2.3_Fresh_C1a_20261021.md)
- Source 课程草案：[source_curriculum_manifest_v0.1_draft.json](lite3-source-scan-preparation-v0.2-2026-10-22/source_curriculum_manifest_v0.1_draft.json)
- Source 资格扫描准备清单：[source_qualification_scan_manifest_v0.2_preparation.json](lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json)
- L1 上行诊断：[multi_step_stairs_L1_ascending_headless_diagnostic_retry1.json](lite3-source-scan-preparation-v0.2-2026-10-22/family_runtime_20261025/multi_step_stairs_L1_ascending_headless_diagnostic_retry1.json)
- L1 下行诊断：[multi_step_stairs_L1_descending_headless_diagnostic_retry1.json](lite3-source-scan-preparation-v0.2-2026-10-22/family_runtime_20261025/multi_step_stairs_L1_descending_headless_diagnostic_retry1.json)
- L2 上行五 seed 诊断：[progress.json](lite3-source-scan-preparation-v0.2-2026-10-22/family_runtime_20261025/l2_ascending_multiseed_retry1/progress.json)
