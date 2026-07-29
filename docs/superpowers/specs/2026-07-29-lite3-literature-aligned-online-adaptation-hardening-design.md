# Lite3 近期文献对齐的在线适应强化设计

## 1. 背景

当前 Lite3 方案建立在 SAC、Residual RL、RMA、AWAC 和 DayDreamer 等 2018 至 2022 年工作之上。2023 至 2026 年的研究进一步揭示了部署期适应中的四个关键问题：策略在线漂移、offline-to-online Critic 失配、真实机器人探索安全，以及“隐状态适应”和“权重更新”的评测混淆。

项目实测也出现了对应症状：将 Actor 更新频率从 1 Hz 提到 5 Hz 后，更新次数从 14 增加到 145，但相对 frozen residual 的速度改善从约 5.6% 降到约 3.5%；45k checkpoint 的 online 2000 步运行虽无安全违规，但 4 次 Actor 更新均被性能回滚，最终参数未变化。

本设计把近期文献结论转化为分阶段工程要求。它不要求一次性复现所有论文，而是先建立可归因、可回滚、可比较的更新链路。

## 2. 文献结论与项目缺口

| 研究方向 | 代表工作 | 当前已有 | 当前缺口 |
|---|---|---|---|
| 隐式本体适应 | DreamWaQ, 2023 | GRU 历史编码、frozen residual 对照 | 未复现 privileged reconstruction；基线名称需避免夸大 |
| offline-to-online RL | RLPD、Cal-QL, 2023 | 预训练 Actor/Critic、在线 replay | 无源 replay 混合、无 Q 校准监控 |
| 真实机器人持续 RL | APRL, 2023 | residual 限幅、低频更新、回滚 | 无稳定策略锚定、无策略漂移指标 |
| 实时参数可塑性 | Synaptic Motor Adaptation, 2023 | 同一运行中更新网络权重 | 使用通用 SAC 更新规则，未针对快速适应元优化 |
| 长期 sim-real 适应 | LoopSR, 2024/2025 | 独立 held-out 仿真 | 无真实数据回流数字孪生流程 |
| 自适应安全 | BAS, 2025 | 硬阈值、安全投影、基础 PPO 回退 | 无恢复策略和 reach-avoid value |
| 腿式 SAC 稳定性 | Sabatini 等, 2026 预印本 | 预训练初始化、Twin Critic | 无 timeout-aware target、multi-step return |

## 3. 方案选择

考虑三种推进方式：

1. **只完成当前性能回滚修复。** 改动最小，但无法判断后续失败来自策略漂移、Critic 失配还是数据分布，研究证据不足。
2. **分阶段强化并保留单变量验证。** 先对齐回滚因果边界和策略漂移观测，再处理 replay/target，最后评估 learned safety 与 sim-real 循环。本设计采用该方案。
3. **一次性引入 RLPD、Cal-QL、multi-step 和 learned safety。** 理论覆盖最全，但会同时改变 Actor、Critic、数据和安全层，仿真结果无法归因，且超出当前分支范围。

## 4. 子项目拆分

### 4.1 子项目 A：回滚对齐与策略漂移观测

本子项目是下一实施计划的唯一代码范围：

- 按已确认设计，只在当前 adaptation episode 首次成功 Actor 更新后启用性能退化回滚。
- 更新前完整窗口只维护最近基线；首次更新时清除跨边界部分窗口。
- 在进入 adaptation episode 时冻结一份参考 Actor。
- 每次成功 Actor 更新后，在固定诊断 batch 上记录当前策略与参考策略的确定性残差差异和参数范数差异。
- 本阶段只记录漂移，不把新指标接入 loss、更新拒绝或安全门控。
- 继续保留动作限幅、变化率限制、数值拒绝和硬安全回滚。

选择“先观测、后正则化”是为了获得 Lite3 实际漂移尺度。没有该尺度时直接选择 KL 或动作锚定系数只是在增加新的调参自由度。

### 4.2 子项目 B：offline-to-online Critic 与数据语义

子项目 A 验证完成后单独设计：

- 源地形训练过程持久化一个有版本和配置元数据的 source replay 样本集。
- 在线 Critic batch 明确由 source 和 current-run transition 按配置比例组成；adaptation Actor batch 仍只来自当前 novelty episode。
- transition 将 `done` 拆分为真实 termination 和 truncation/bootstrap mask。
- 控制超时和安全暂停 transition 继续排除，不作为 timeout-aware bootstrap 的样本。
- 先加入 Q 分布、TD target 和源/在线 batch 差异日志，再决定是否采用 Cal-QL 类校准项。
- 一步 target 稳定后，multi-step return 作为独立实验，不与 replay 混合同时上线。

### 4.3 子项目 C：基线、真机循环与 learned safety

该子项目不进入当前仿真实现计划：

- 将 `residual_frozen` 明确为“冻结权重、历史条件隐式适应”基线，不宣称完整复现 DreamWaQ。
- 如需论文级对照，另行训练带 privileged reconstruction 的 latent adaptation 基线。
- 真机持续改进优先采用真实轨迹回流仿真或 hard-to-sim objective model，而不是无保护地持续更新 Actor。
- learned recovery policy 或 reach-avoid value 只能作为硬安全控制器之外的附加层。
- 未具备吊架、急停、独立 watchdog 和操作员监督前，不授权真机在线梯度更新。

## 5. 子项目 A 数据流

```text
进入 ADAPTING
  -> 保存完整稳定 SAC 快照
  -> 冻结 reference actor
  -> 收集更新前性能基线
  -> adaptation replay 达到门槛
  -> 候选 actor update
  -> 发布前计算并校验 policy drift
  -> 成功发布首次 actor update
  -> 清除跨边界性能窗口并启用更新后评估
  -> 完整更新后窗口
       -> 改善：保存稳定快照
       -> 连续退化：回滚
       -> 硬安全事件：立即回滚
```

策略漂移诊断至少包含：

- 当前 Actor 与 reference Actor 在同一 adaptation batch 上的确定性残差均方差；
- 每关节绝对残差变化的最大值和均值；
- Actor 参数向量相对 reference 的 L2 范数；
- 当前 adaptation episode、Actor 版本和更新步数。

诊断计算放在后台更新线程的候选 Actor 发布之前，不进入 50 Hz 控制关键路径。非有限诊断值拒绝该次候选更新；有限的诊断标量随成功发布的 Actor 版本记录。日志不保存可无限增长的 batch。

## 6. 并发与失败处理

- reference Actor 由更新线程在 `start_episode` 命令下、现有更新锁内创建或刷新。
- 控制线程只读取按 episode 汇总的成功 Actor 更新计数，不直接访问 reference Actor。
- 漂移诊断失败不能阻止硬安全回滚；非有限诊断值必须记录更新错误并拒绝发布候选版本。
- 旧 episode 的迟到更新和诊断不能修改当前 episode 的回滚资格或日志快照。
- 回滚恢复 Actor、Critic、target Critic、优化器、熵参数和版本，并清除对应 reference Actor。
- 未发生 Actor 更新的 episode 不产生“更新后性能退化”判定。

## 7. 测试与验收

### 7.1 自动化测试

- 更新前退化窗口不触发性能回滚。
- 首次成功 Actor 更新启用回滚并清除部分窗口。
- 两个完整更新后退化窗口仍触发回滚。
- 旧 episode 更新不影响当前 episode。
- reference Actor 在 episode 内保持冻结，进入新 episode 时刷新。
- 漂移指标对相同 Actor 为 0，对已知参数变化为正且有限。
- Critic-only 更新不产生 Actor 漂移事件。
- 安全事件在首次 Actor 更新前仍立即回滚。

### 7.2 GPU 验证

使用 seed 7、0.3 m/s、`residual_00045000.pt`：

1. 750 步 online 探针确认无首帧超时、novelty 前 0 次 Actor 更新。
2. 2000 步 online 运行确认无跌倒、无重置、无安全违规。
3. 每次性能回滚前必须存在首次 Actor 更新以及两个完整更新后窗口。
4. 报告 Actor 漂移与速度误差、倾角、非足接触之间的时间关系。
5. 不以降低安全阈值、放宽残差限幅或删除失败 seed 获得通过结果。

## 8. 决策门槛

子项目 A 完成后按证据选择下一步：

- 漂移在少量更新后快速增大且性能恶化：设计稳定策略动作锚定或 KL 约束。
- 漂移较小但 Q/TD target 异常：优先进入子项目 B 的 replay 与 Q 校准。
- 45k online 安全且优于 frozen：扩大到至少 3 个具有 held-out 覆盖的种子，再决定算法扩展。
- 安全事件仍早于有效更新：停止调 Actor，优先处理 residual checkpoint 和安全基线。

## 9. 非目标

- 当前分支不复现完整 RLPD、Cal-QL、DreamWaQ、BAS 或 LoopSR。
- 不因为引用新论文就改变验收阈值。
- 不把 2026 预印本结论描述为已被充分复现或同行评审确认。
- 不把仿真中的安全过滤等同于真机安全认证。

## 10. 完成定义

- 原理文档包含 2023 至 2026 文献、方法分类及与代码的明确差距。
- 子项目 A 有独立实施计划，且不混入子项目 B/C。
- 代码实现和验证遵守单变量顺序。
- 实验报告同时给出成功和失败结果。
- 每次有效改动提交并推送到 `jenny-tian/rl_training_main` 私有仓库。
