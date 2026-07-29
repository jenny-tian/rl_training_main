# Lite3 从严格平地策略在线学习低台阶与不平路设计

## 1. 目标

建立一个可归因、可复现的 Isaac Lab 实验，证明 Lite3 的初始化策略只接受过平地训练，随后仅依靠受保护的在线交互，在有限预算内学会通过 4 至 6 cm 单级台阶和约 +/-4 cm 不平路面。

本阶段只完成仿真闭环。真机仅定义安全状态机和接口边界，不向 Lite3 发送在线学习动作。

## 2. 已确认的成功标准

- 基础 PPO 只在严格 plane 环境训练，训练 provenance 可验证。
- residual Actor 初始输出严格为零，且初始化阶段未接触台阶、rough 或 slope 数据。
- 在线训练最多使用 100 个受保护 episode 或 200,000 个控制步，任一预算先耗尽即停止。
- 每个训练 episode 最多 2000 个控制步；提前结束仍按实际交互步数计入总预算。
- 训练按五个难度阶段推进，每阶段最多 20 个 episode。
- 最终在 20 个封存地形 seed 上，`learned_frozen` 通过率至少为 80%；每个 seed 生成一条同时包含 4 至 6 cm 单级台阶和约 +/-4 cm 不平路段的组合道路。
- `learned_frozen` 通过率相对 `base` 和 `initial_zero_residual` 都至少提高 30 个百分点。
- 最终 20 个确定性目标地形评测 episode 零跌倒。
- 返回平地后，完成率不得比基础 PPO 低 5 个百分点，速度误差和倾角 RMS 均不得恶化超过 5%，且不得增加跌倒或非足端碰撞。

通过定义为：机器人在规定步数内到达道路终点，期间没有 reset、跌倒、安全终止或人工式中止。训练中的安全中止必须计入样本预算和失败统计，不能删除。

## 3. 非目标

- 不在线微调整个 PPO。
- 不使用来源不明的现有 PPO 或见过 rough/slopes 的 45k residual 作为正式 flat-only 基线。
- 不在封存 seed 上调参、继续训练或选择 checkpoint。
- 不用 learned safety 替代硬阈值、动作投影、watchdog 和回滚。
- 不在本阶段执行真机在线梯度更新、自动起身或自动返回起点。
- 不一次性引入 diffusion/flow policy、完整 Cal-QL、learned recovery policy 或数字孪生循环。

现有 checkpoint 和历史实验仍可用于在线更新基础设施回归，但不得用于本实验的能力结论。

## 4. 方案选择

考虑了三种路线：

1. **冻结平地 PPO，加零初始化 residual SAC。** 保留稳定基础控制器，只让受限 residual 在线学习。本设计采用该方案，因为它最符合“只会平地，随后在线学会目标地形”的因果问题，也能复用现有发布、漂移监控和回滚基础设施。
2. **在线微调整个 PPO。** 表达能力更强，但灾难性遗忘、回滚粒度和安全风险都更高，不适合作为第一阶段。
3. **RMA/DreamWaQ 类部署时隐变量适应。** 样本效率高，但主要改变推断出的环境表征，不满足本实验对在线权重学习的严格定义。

## 5. 研究依据与采用边界

| 机制 | 依据 | 本设计中的采用方式 |
|---|---|---|
| off-policy 腿式在线学习 | Sabatini、Li、Hutter, 2026, *Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged Locomotion* | 分别消融策略初始化、timeout-aware target 和 multi-step return；该工作是预印本，不把结论视为已充分复现 |
| 源数据与在线数据混合 | Ball 等, ICML 2023, RLPD | 使用只读 `flat_replay` 与跨 episode `online_replay` 分层采样，不让少量在线失败样本完全覆盖平地分布 |
| Q 值校准 | Nakamoto 等, NeurIPS 2023, Cal-QL | 第一阶段先记录源/在线 Q 尺度和 TD target；只有独立消融证明需要时才加入 conservative/calibrated loss |
| 受约束真实机器人探索 | Smith、Cao、Levine, 2023, APRL | residual 权限渐进开放，并对稳定策略加入动作锚定和漂移约束 |
| 稳定性对齐 residual | Jayasinghe 等, 2026, *Residual Control for Fast Recovery from Dynamics Shifts* | 冻结名义策略，限制 residual 幅度、变化率和候选漂移；不声称复现其 inference-only 方法 |
| 独立恢复与安全切换 | Zhong 等, 2025, BAS | 硬安全监督器独立于学习器；恢复策略仅作为未来真机附加层 |
| 长期 sim-real 适应 | Wu 等, IROS 2025, LoopSR | 作为未来真实数据回流仿真的候选路线，不进入当前仿真实现 |

采用顺序必须保持单变量：单步 SAC 基线、termination/truncation 修正、3-step、5-step、双 replay、Actor 锚定。某项机制未在相同开发 seed 上产生稳定收益时，不进入后续正式配置。

## 6. 系统架构

### 6.1 Flat Baseline Trainer

负责训练正式基础 PPO，并在环境构建和训练启动时执行正向断言：

```text
terrain_type == "plane"
terrain_generator is None
```

仅检查任务名称不够，因为名称不能证明实际场景是平地。训练产物必须包含 manifest，至少记录：

- 环境注册名称和解析后的地形配置；
- PPO 配置和归一化统计；
- Git SHA、Python/Isaac Lab/RSL-RL 版本；
- 训练 seed、环境数量、控制频率和训练步数；
- checkpoint、配置和 manifest 的 SHA-256；
- 训练开始/结束时间和运行目录。

### 6.2 Zero Residual Initializer

residual Actor 的均值输出层权重和偏置初始化为零。由于随机 SAC 即使均值为零也可能采样非零动作，`initial_zero_residual` 和 `BASE_WARMUP` 还必须把采样后的 residual gain 固定为 0，从执行路径上保证 residual 逐元素精确为零。在线新异门开启后才允许从小增益逐级开放探索。初始化验收要求 `base` 与 `initial_zero_residual` 在相同状态序列上的最终动作最大绝对差不超过 `1e-6`。

Critic 可以使用平地 transition 预热，但 residual Actor 在新异门开启前不得更新。任何目标地形 transition 都不能进入初始化 checkpoint。

### 6.3 Persistent Online Learner

跨环境 reset 保存：

- Actor、双 Critic、target Critic；
- Actor/Critic/温度优化器和熵参数；
- `flat_replay`、`online_replay` 和 n-step 队列；
- 当前稳定快照、Actor 版本、策略 reference；
- 课程阶段、episode/step 预算、早停统计；
- CPU/CUDA/replay 随机数状态。

环境 reset 不是训练进程重启。只有显式新实验才能清除上述状态。

### 6.4 Independent Safety Supervisor

安全监督器位于学习策略和仿真执行器之间，拥有最终动作裁决权。它负责残差限幅、变化率限制、关节目标投影、姿态/高度/关节速度/非足端碰撞检查、控制超时和安全中止。

学习器不能修改安全阈值。安全事件触发时立即清除 learned residual、关闭 Actor gate、停止前进指令并请求完整 SAC 回滚。

### 6.5 Curriculum and Experiment Coordinator

协调器负责地形阶段、预算、reset、checkpoint 和评测模式。它不向 Actor 暴露地形高度、道路坐标、课程级别或 seed。

## 7. 控制与更新数据流

```text
本体状态与 25 帧历史
  -> 冻结 flat PPO
  -> base action
  -> 当前 residual Actor
  -> 独立安全投影
  -> 仿真执行
  -> transition + termination/truncation/safety mask
  -> flat/online replay
  -> 后台 SAC 候选更新
  -> 数值、episode、漂移和性能检查
  -> 原子发布候选 Actor 或恢复稳定 SAC 快照
```

控制线程不得等待梯度更新。候选 Actor 只有在后台事务完整成功后才可发布；发布或回滚未确认期间，控制线程执行冻结基础 PPO 和安全投影。

## 8. Replay 与 SAC 设计

### 8.1 Flat Replay

`flat_replay` 是由正式 flat PPO 在严格 plane 环境中生成的只读数据集。数据集具有独立 manifest 和 hash，不能在在线运行时被覆盖。安全暂停、控制超时和无效观测不进入该数据集。

### 8.2 Online Replay

`online_replay` 保存目标地形上的有效 transition，并跨 reset 持久化。失败前的正常 transition 可以训练 Critic；安全暂停、控制超时以及安全监督器接管后的 transition 不进入 Actor 有效 batch。

每条 transition 明确保存 `terminated`、`truncated`、bootstrap mask、safety mask、curriculum stage 和 episode id。stage/id 只用于采样审计和统计，不进入策略观测。

### 8.3 分层采样

- 在线数据不足时，Critic batch 使用约 75% flat、25% online。
- online replay 达到配置门槛后，过渡到 50% flat、50% online。
- 实际比例、两类 Q 均值、TD target 和 TD error 必须逐次记录。
- Actor 的 SAC 主损失只使用有效 online batch；flat batch 只用于 residual 零动作锚定。

这些比例是初始可检验配置，不是论文结论。必须与 online-only Critic 做相同 seed 消融。

### 8.4 Actor 锚定与 residual 权限

Actor loss 由目标地形 SAC loss、平地零 residual 锚定和候选漂移惩罚组成。residual gain 档位固定为 `0.0 / 0.25 / 0.5 / 0.75 / 1.0`：`BASE_WARMUP` 使用 0，新异门开启后从 0.25 开始；连续 5 个 episode 无跌倒、无安全回滚且性能成本不退化时提高一档；任何安全或性能回滚降低一档。gain 只缩放安全限幅内的 residual，不能提高硬安全上限。

候选发布至少检查：

- 所有参数、loss、Q 值和漂移指标有限；
- 候选属于当前 adaptation episode；
- 固定诊断 batch 上的动作变化和参数 L2 未超过配置阈值；
- 无待确认的安全回滚；
- 发布事务失败时可恢复稳定快照。

漂移阈值先用开发 seed 上的 zero-residual 和首个安全档位运行校准，取已接受安全候选漂移分布的第 99 百分位乘以 1.25，并写入冻结配置；封存评测前不得再次修改。校准日志、样本数、分位数和最终阈值必须进入 manifest。

### 8.5 Return 语义

单步 TD 是首个基线。随后先修正真实 termination 与时间 truncation 的 bootstrap 语义，再分别评估 3-step 和 5-step return。控制超时和安全接管 transition 始终无效，不能被误当作普通 timeout bootstrap 样本。

## 9. 奖励与信息隔离

在线奖励只使用机器人可部署状态和任务命令：

- 前进速度跟踪和有效位移；
- 姿态、机身高度和非足端碰撞；
- 关节速度、扭矩、动作变化率；
- residual 幅度和 residual 变化率；
- 跌倒或安全终止惩罚。

奖励、更新门控和性能回滚不得读取目标区域坐标、台阶高度、地形类型、课程阶段或封存 seed。道路终点和 held-out mask 只用于 episode 结束及离线评测统计。

## 10. 课程与预算

五个阶段按顺序执行：

1. 2 cm 单级台阶；
2. 4 cm 单级台阶；
3. 6 cm 单级台阶；
4. 约 +/-2 cm 随机不平路；
5. 约 +/-4 cm 随机不平路。

每阶段最多 20 个训练 episode。总预算最多 100 episode 或 200,000 控制步。当前阶段最近 10 个 episode 通过率至少 80%、零跌倒且平地保留检查通过时，可以提前晋级。连续 10 个 episode 没有改善时暂停实验并报告失败，不自动放宽安全阈值、残差权限或训练预算。

每个阶段使用固定的开发 seed 集合，默认 5 个几何 seed，训练 episode 在其中均匀采样。开发 seed 可以用于调参和消融，封存 seed 不可以。

## 11. Reset、失败与恢复

仿真状态机为：

```text
RESET
  -> BASE_WARMUP
  -> ADAPTING
  -> SAFE_ABORT
  -> ROLLBACK
  -> 等待人工式 reset
  -> BASE_WARMUP
```

`BASE_WARMUP` 期间 residual 为零，只有冻结 flat PPO 控制。达到稳定窗口后才恢复 residual gate。

本阶段的“人工式 reset”由实验协调器模拟操作员确认：环境完全终止、稳定快照和日志落盘、机器人回到规定起点后才开始下一 episode。它不尝试在失稳位置继续行走，也不训练自动起身。

checkpoint 使用临时文件、校验和与原子替换形成事务。写入失败时保留上一份完整 checkpoint，并停止新的 Actor 发布。

## 12. 数据集与封存规则

- 开发地形和封存地形使用不同的生成 seed 范围，并保存生成配置与几何 hash。
- 目标几何 seed 随评测 seed 改变，不能在环境配置内硬编码同一个 generator seed。
- 每个封存 seed 对应一条组合道路，包含一个由 seed 决定的 4、5 或 6 cm 单级台阶和一个约 +/-4 cm 不平路段；通过必须覆盖整条道路。
- 封存清单在正式评测前生成并冻结；只有路径和 hash 可用于启动，具体结果不能用于 checkpoint 选择。
- 正式评测只执行一次。基础设施错误导致的无效运行必须保留原日志，并在修复后以新实验编号重跑，不能静默覆盖。

## 13. 对照与消融

正式能力对照：

| 模式 | PPO | residual | 在线更新 | 目的 |
|---|---|---|---|---|
| `base` | 冻结 flat PPO | 0 | 否 | 测量零样本能力 |
| `initial_zero_residual` | 同一 PPO | 严格零初始化 | 否 | 验证组合路径不改变基础动作 |
| `online_learning` | 同一 PPO | 在线 residual | 是 | 执行有限预算学习 |
| `learned_frozen` | 同一 PPO | 学习后的 residual | 否 | 排除评测期间继续更新的影响 |

开发 seed 上按顺序执行：

1. 单步 TD 基线；
2. timeout-aware target；
3. 3-step return；
4. 5-step return；
5. online-only replay 对比双 replay；
6. 无 Actor 锚定对比有 Actor 锚定。

每次只改变一个因素，并使用相同初始 checkpoint、seed、课程和预算。最终配置必须由开发结果预先确定，不能根据封存结果回选。

## 14. 指标与日志

每控制步至少记录：

- episode、stage、step、seed 和模式；
- base/residual/final action 范数；
- Actor 版本和 residual gate 状态；
- 速度误差、倾角、机身高度和碰撞；
- terminated、truncated、safety mask 和回滚状态；
- replay 大小和实际源/在线采样比例。

每次 Actor 更新独立记录：

- Actor/Critic/温度 loss、Q、TD target、TD error 和梯度范数；
- policy residual MSE、每关节变化和参数 L2；
- flat anchor loss、n-step 长度和候选接受/拒绝原因；
- adaptation episode、候选版本和最终发布版本。

每个 episode 输出不可变 summary；整个实验输出课程曲线、失败原因分布、学习预算、平地保留和最终封存统计。

## 15. 测试策略

### 15.1 单元测试

- plane-only 正向断言拒绝 generator、rough 和 slope 配置；
- zero residual 初始化与 base action 一致；
- 双 replay 的比例、只读边界和有效 mask；
- termination/truncation 与 1/3/5-step target；
- Actor 平地锚定和 residual 权限上限；
- 事务 checkpoint 的完整保存、恢复和损坏回退；
- 课程预算、早停、晋级和连续无改善停止；
- 封存 seed 与开发 seed 不相交。

### 15.2 集成测试

- reset 后 SAC、replay、RNG、版本和课程预算连续；
- 安全事件立即产生零 learned residual，并在回滚确认前保持基础策略；
- 旧 episode 候选不能发布到新 episode；
- 评测模式关闭所有梯度更新且不改变 checkpoint；
- 目标地形坐标、高度和 seed 不进入 Actor 输入、奖励或更新门控。

### 15.3 GPU 验证

1. flat PPO 训练和固定 seed 资格测试；
2. `base` 与 `initial_zero_residual` 动作一致性；
3. 单阶段 2 cm 台阶短探针；
4. 安全中止、reset、跨 episode 恢复探针；
5. 五阶段开发训练和消融；
6. 配置冻结后执行 20 个封存 seed 正式评测；
7. 回到平地执行保留测试。

## 16. 真机边界

仿真通过不等于真机授权。未来 Lite3 第一阶段采用人工复位：风险预警时 residual 归零、速度命令归零并切换到已验证站立/阻尼控制；已经摔倒时进入阻尼或急停，由操作员复位。策略回滚只能恢复网络参数，不能恢复物理姿态。

真机实验还需要吊架、物理急停、独立 watchdog、控制权仲裁、人员安全距离和操作员全程监督。自动退回平地和自动起身属于独立后续项目。

## 17. 风险与决策门槛

- **flat PPO 零样本就能通过目标地形。** 提高目标几何难度，但仍从 2 cm 课程开始；正式目标必须让 base 低于在线策略至少 30 个百分点，不能故意破坏基础策略。
- **100 episode 仍无改善。** 停止并分析 reward、Q 校准、更新延迟和探索权限，不扩大预算掩盖失败。
- **双 replay 阻碍目标适应。** 保留 online-only 消融，根据开发 seed 证据调整比例。
- **n-step 引入偏差或不稳定。** 回到最近通过的单步/短步配置，不把多个 target 变化同时保留。
- **Actor 学会目标地形但遗忘平地。** 提高 flat anchor 或降低 residual 权限；若仍失败，该 checkpoint 不具备晋级资格。
- **安全回滚频繁早于有效更新。** 停止 Actor 调参，先检查基础策略、课程首级和安全状态定义。

## 18. 实施分期

### 18.1 阶段 A：严格初始化与地形隔离

实现 plane-only 断言、flat PPO provenance、flat replay manifest、zero residual 执行门和开发/封存组合道路生成。门槛是 flat PPO 资格测试通过，且 `base` 与 `initial_zero_residual` 动作一致。

### 18.2 阶段 B：跨 reset 学习状态与 target 语义

实现完整 checkpoint、replay/RNG/课程恢复、termination/truncation 和可配置 1/3/5-step target。门槛是 CPU 集成测试和 GPU 安全 reset 探针通过。

### 18.3 阶段 C：双 replay、Actor 锚定与课程协调

实现分层 Critic batch、flat anchor、residual gain 档位、预算和早停。门槛是在开发 seed 的 2 cm 阶段出现可重复改善，且平地保留检查通过。

### 18.4 阶段 D：消融、完整课程与封存评测

按预定顺序完成单变量消融，冻结最终配置和 hash，再运行五阶段训练、20 seed 组合道路评测及平地保留测试。封存结果不反向修改配置。

## 19. 交付物

- 可追溯的 flat PPO checkpoint 和 manifest；
- flat replay 数据集和 manifest；
- zero residual 初始化 checkpoint；
- 跨 reset 在线学习 checkpoint 与恢复工具；
- 五阶段课程配置；
- 对照、消融和 20 seed 封存评测脚本；
- 原始 JSONL/TensorBoard 日志、summary 和统计报告；
- 明确区分通过、失败和未验证结论的项目文档。

## 20. 完成定义

- 自动化测试、静态检查和 GPU 因果探针通过。
- 正式配置在查看封存结果前冻结并记录 hash。
- 100 episode/200,000 step 预算和所有失败 episode 完整可审计。
- 达到第 2 节全部成功标准，或如实形成未达标报告和下一步根因分析。
- 每次有效改动提交并推送到 `jenny-tian/rl_training_main` 私有仓库。

## 21. 参考资料

1. Philip J. Ball 等，*Efficient Online Reinforcement Learning with Offline Data*，ICML 2023，<https://arxiv.org/abs/2302.02948>。
2. Mitsuhiko Nakamoto 等，*Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning*，NeurIPS 2023，<https://arxiv.org/abs/2303.05479>。
3. Laura Smith、Yunhao Cao、Sergey Levine，*Grow Your Limits: Continuous Improvement with Real-World RL for Robotic Locomotion*，2023，<https://arxiv.org/abs/2310.17634>。
4. Peilin Wu 等，*LoopSR: Looping Sim-and-Real for Lifelong Policy Adaptation of Legged Robots*，IROS 2025，<https://arxiv.org/abs/2409.17992>。
5. Yichao Zhong 等，*Bridging Adaptivity and Safety: Learning Agile Collision-Free Locomotion Across Varied Physics*，2025，<https://arxiv.org/abs/2501.04276>。
6. Nethmi Jayasinghe 等，*Residual Control for Fast Recovery from Dynamics Shifts*，2026 预印本，<https://arxiv.org/abs/2603.07775>。
7. Gianluca Sabatini、Chenhao Li、Marco Hutter，*Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged Locomotion*，2026 预印本，<https://arxiv.org/abs/2605.24975>。
