# Lite3 在线地形适应：当前技术方案

**执行入口更新：2026-09-29；下文工程历史版本：2026-09-23**

> 现行执行方案为 [v2.2（2026-09-28）](docs/lite3_execution_plan_v2_2_2026-09-28.md)；它细化并覆盖 v2.1 的执行顺序、预算、验收门槛和候选路线。
> 历史决策与证据仍见 [v2.1（2026-09-25）](docs/lite3_online_adaptation_plan_v2_2026-09-24.md)。
> 本文保留为已有工程能力和历史实验记录，不再决定训练优先级。
> v2.1 暂停高度扫描教师路线，优先 G0 记录/协议和 G1 学习信号诊断；合格教师不再是直接 SAC 或无教师方法的前置条件。
> 执行前需同时阅读 [2026-09-28 审核](docs/lite3-plan-audit-2026-09-28.md)，其中区分了可立即执行的 G0 与待修订的算法门槛。
> 新增 [FlowDAgger / USR 与跨领域文献复核（9/28）](docs/lite3-literature-and-algorithm-review-2026-09-28.md)：包含在线来源、开源状态和迁移边界；其候选建议已细化为 v2.2 执行方案，尚未表示算法已实现。
> 第一阶段进展：[G0 runner 配对修复（9/28）](docs/lite3-g0-runner-parity-2026-09-28.md)。已定位 reset 执行器延迟差异并补齐 SAC 主路径轨迹。
> 后续完成：[G0.3 共同任务奖励/终止（9/28）](docs/lite3-g0-task-objective-2026-09-28.md)。五地形基线配对、成功/跌倒/deadline/cutoff和真实replay验证均已完成；下一步是G0.4预算、session恢复与封存守卫。
> G0.4：[G0.4账本/恢复与冻结评估（9/28）](docs/lite3-g0-session-ledger-2026-09-28.md)。逐步预算、跨进程优化器/replay续接和冻结零写入已验证；最终协议锁定前仍需初态扰动与统一探针/平地调度。
> G0.5：[G0.5初态、探针与平地调度（9/28）](docs/lite3-g0-initial-state-schedule-2026-09-28.md)。真实两入口初态/轨迹配对与冻结零写入通过；开发协议检查表已生成，正式封存与G1 warmup仍未完成。
> G0.6：[G0.6出生几何连续范围审核（9/29）](docs/lite3-g0-spawn-geometry-2026-09-29.md)。22个碰撞体、中心及8角点通过，证书绑定模型/reset源码；下一步进入G1 warmup与学习信号诊断。
> G1.1：[G1.1目标域预热与恢复（9/29）](docs/lite3-g1-target-warmup-2026-09-29.md)。已接入4000步冻结base采样、critic-only预热与新样本actor更新；完整回合留出诊断、控制超时问题和路线晋级仍待完成。
> 最新：[G1.2控制时延与冻结执行修复（9/29）](docs/lite3-g1-control-latency-2026-09-29.md)。减少GPU同步与固定边界分配；修复SAC冻结评估增益为0的问题。修复后9696步无控制超时，下一步是完整回合留出价值诊断。

本文是当前代码和实验协议的技术基线。它描述已经存在的工程能力，并把潜变量残差方案作为扩展路线；任何尚未完成的训练或实验均不写成已验证结论。

## 1. 目标和约束

给定一个已经在平地训练的 Lite3 PPO 控制器，在仿真中的未见地形上进行有限在线适应，同时满足：

- 基础 PPO 权重默认冻结；
- 适应模块只拥有受限的参数或动作自由度；
- 实际执行动作经过统一安全投影；
- 真实终止、时间截断和 reset 前末帧严格区分；
- 候选策略通过冻结探针和安全审计后才能发布；
- 训练、开发、封存测试数据相互隔离。

## 2. 统一动作语义

控制器输出请求动作 `a_req`，安全模块输出执行动作：

```text
a_exec = P_safe(a_req)
```

环境状态转移由 `a_exec` 产生。PPO 的 old log-prob、概率比和 SAC Critic 的动作输入仍对应保护前的请求动作；每条 transition 额外保存执行动作和两者差异。该约定避免把动作过滤器造成的变化错误地当成策略采样变化。

## 3. 两条当前适配路线

### 3.1 Direct LoRA-PPO

基础 Actor 指定线性层写为：

```text
W_i = W_i,0 + (alpha_i / r) B_i A_i
```

`W_i,0` 冻结，`A_i` 和 `B_i` 可训练，`B_i` 零初始化。在线更新只作用于 LoRA 参数和独立 Critic。每个连续 rollout 绑定一个策略版本，更新完成后生成候选检查点；近似 KL、有限值、参数范围、源平地保持和安全事件均可触发拒绝或回滚。

### 3.2 冻结 PPO + Residual SAC

Residual SAC 的 Actor 根据本体历史、当前帧和基础动作输出小残差。残差经历幅值、变化率、关节范围、姿态/接触和传感器有效性检查。新异门控允许时更新 Actor；Critic 可以在更宽的安全回放上更新。控制步调度用于算法公平配对，墙钟调度用于实时吞吐和 deadline 研究。

## 4. 潜变量残差扩展

潜变量扩展由以下可独立验收的组件组成：

1. 本体历史和高度扫描摘要编码为潜变量；
2. 教师残差经动作限幅归一化后用截断 SVD拟合固定解码矩阵；
3. 对教师动作求低维 steering code；
4. 监督训练观测编码器预测该 code；
5. 低维 code 经有界解码生成残差；
6. 风险升高时缩小残差增益，硬安全事件时置零并暂停更新。

这些组件可以用于后续 RGLS 类实验。当前项目没有把它们合并成已经完成的端到端主结果，也没有把 PPO 直接称作 Flow policy。

## 5. 训练阶段

### 阶段 A：基础策略资格审查

冻结基础检查点，运行平地和源地形资格审查，确认成功判据、终止协议和动作接口正确。资格不通过时停止适配实验。

### 阶段 B：适配初始化

LoRA 路线从零低秩增量开始。潜变量路线可以使用安全教师动作、动作反演和固定解码器做离线初始化。初始化数据不得来自封存测试集合。

### 阶段 C：受限在线更新

只更新当前路线允许的参数。rollout 或 replay 必须写入策略版本、动作语义、终止字段和安全状态。违反训练期安全条件的候选立即失去晋升资格。

### 阶段 D：冻结候选验证

候选依次经过参数/漂移检查、源平地保持、开发地形冻结探针和安全审计。通过后登记完整状态、校验哈希和版本信息，成为可恢复检查点。

### 阶段 E：封存测试

通过独立 sealed 入口加载预先选定 checkpoint，关闭梯度、优化器和参数写入，只生成独立测试报告。封存结果不能回流候选选择或继续训练。

## 6. 终止和价值计算

真实失败终止使用末帧状态但不 bootstrap；时间截断可以用 reset 前末帧做价值引导，但优势递推在回合边界切断。reset 后新初态不能作为上一条失败 transition 的后继状态。所有路线都要在 summary 中分开统计 terminated 和 truncated。

## 7. 验证优先级

当前优先级是：

1. 固定动作/终止/安全协议；
2. 在同一训练、开发、封存划分上运行 Frozen PPO、Direct LoRA-PPO、Direct Residual SAC；
3. 至少使用三个独立 simulation seeds；
4. 报告成功率、位移、速度误差、姿态/接触违规、投影差异、平地保持和回滚；
5. 再比较潜变量初始化、风险门控和动作反演的独立贡献。

FlowDAgger 只作为潜空间动作反演的参考思想。除非另行实现 flow policy 和对应数据闭环，不能在报告中写成“本项目采用 FlowDAgger”。

## 8. 代码与文档对应

- 公共组件：`source/rl_training/rl_training/online_adaptation/`
- LoRA-PPO：`scripts/reinforcement_learning/online_adaptation/train_lora_ppo.py`
- 多回合候选监督：`run_lora_ppo_multiepisode.py`
- Sealed 测试：`run_lora_ppo_sealed_test.py`
- 高度扫描教师：`train_height_scan_teacher.py`
- 教师解码器：`fit_teacher_decoder.py`
- 当前完整项目说明：`docs/lite3-project-documentation-20260923.md`

## 9. 当前限制

当前证据主要来自仿真和组件测试。潜变量残差方案、完整特权教师轨迹、端到端高度扫描编码器以及实体机器人部署仍需独立验证。任何数值性能结论都必须绑定具体 checkpoint、协议、地形划分和随机种子。

## 10. Height-scan teacher 当前进度（2026-09-23）

正式 train split 固定为 `101/103/107`，由 `train_height_scan_teacher.py` 顺序训练。训练阶段允许在物理 reset 后继续更新；训练结束后，最终 checkpoint 在三个 train seed 上分别进行最多 4,000 步的冻结 probe，只有三个 probe 全部通过完成、安全和零 reset 条件才发布 teacher。2,000 步以下只能使用 `--smoke`，其 manifest 明确为未认证结果。

32 步三 seed smoke 已验证 Isaac 管线和 provenance contract，但不认证 teacher，结果位于 `artifacts/height_scan_teacher_smoke_qualified_20260923/`。正式多回合训练 `artifacts/height_scan_teacher_formal_multiepisode_20260923/` 已结束：三个 train seed 各 20,000 步，最终冻结 probe 在 seed 101 的第 683 步跌倒；同协议冻结 PPO 在第 675 步跌倒，性能没有可见提升，因此没有合格 teacher。

失败审计发现扫描布局被转置：Isaac Lab `GridPatternCfg(ordering="xy")` 的射线 reshape 顺序是 `[lateral_y, forward_x]`，rough 配置为 15 行、23 列；v1 却按 23 行、15 列解释，并把横向统计用作前向统计。高度扫描摘要与教师权重因此使用了错误的空间语义。代码已修正为 schema/teacher contract v2，多回合运行 manifest 也随之升级；旧 v1 检查点不得续训或与新观测混用。

v2 修复验证（2026-09-24）：online adaptation 测试 `1506 passed, 7 skipped`，`compileall` 和 `git diff --check` 通过。Isaac Sim 5.1 pip wheel 使用当前 worktree 和 Isaac Lab 源码路径成功启动 CPU smoke；Lite3 FlatOnline 任务注册、reset/step 及 45 维 policy / 393 维 critic（含 345 维 height scan）观测均正常。当前执行容器看不到 NVIDIA 驱动（NVML driver not loaded、CUDA no device），因此该 CPU smoke 不能替代 GPU 训练 smoke。v2 三 seed GPU smoke 和后续独立正式训练仍待在可用 GPU 会话运行；此前失败产物保留作审计证据。未得到合格 teacher 前，不采集 teacher 数据，也不启动长 SAC/RGLS 训练。
