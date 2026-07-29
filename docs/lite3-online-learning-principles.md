# Lite3 单次连续行走在线学习原理与实现

## 1. 文档目的

本文说明当前项目中 DeepRobotics Lite3 在线地形适应功能的算法原理、数据闭环、工程实现、参考资料、安全边界与验证状态。

当前方案的核心是：保留已经训练好的 PPO 步态控制器，并在其动作上叠加一个受严格约束的残差 SAC 策略。残差策略在一次不重置的连续行走过程中采集新经验、执行梯度更新，并把更新后的网络参数用于同一次行走的后续控制。

需要特别说明：截至 2026-07-29，NVIDIA 驱动与 Isaac Lab 已恢复可用，环境创建、冻结 PPO 推理、残差 SAC 梯度更新和单次连续行走脚本均已通过短程仿真验证。当前候选基础策略 `lite3_rl_deploy_policy_candidate.pt` 已通过五种子源地形资格测试：0 次跌倒，平均速度误差 0.103，小于 0.35 门槛。novelty-gated 在线适应机制已经实现并具备自动化测试，但 held-out 方块道路正式对照尚未通过最终验收。2023 至 2026 年的相关工作表明，当前方案还需要进一步处理策略漂移、offline-to-online 数据混合、Q 值校准和 timeout-aware target；这些缺口在第 12 节和第 14 节中明确列出。

## 2. 什么是这里的“真正在线学习”

这里的在线学习不只是：

- 在多个预训练策略之间切换；
- 根据最近观测更新一个隐藏状态；
- 使用固定网络推断环境参数；
- 在多次环境重置后离线重新训练。

当前实现会在一次连续行走中执行以下闭环：

1. 从 Lite3 当前本体传感器状态构造观测历史；
2. 冻结 PPO 产生基础动作；
3. 残差 Actor 产生小幅动作修正；
4. 安全过滤器投影并执行最终动作；
5. 根据机器人响应计算奖励并写入回放池；
6. 后台 SAC 更新线程从本次行走产生的数据中采样；
7. Critic 持续学习，Actor 只在本体感知信号持续显示新异状态后更新；
8. 通过稳定快照、性能回滚和安全回退保护已发布 Actor；
9. 将更新后的 Actor 原子发布给控制线程；
10. 后续控制周期使用新的网络参数继续行走。

因此，进入演示前和离开演示后保存的残差 Actor 参数应当不同；运行脚本也会显式保存并检查 `residual_before.pt` 和 `residual_after.pt`。

## 3. 总体架构

```text
本体感知历史
       │
       ├────> 冻结 PPO ──────> 基础动作 a_base
       │
       └────> GRU + SAC Actor ─> 原始残差 a_res
                                      │
                         幅度/变化率/关节限位投影
                                      │
                                      v
最终动作 a = a_base + a_res ───────> Lite3
                                      │
                         奖励 + 下一时刻状态
                                      │
                                      v
                                 Replay Buffer
                                      │
                           后台 SAC 梯度更新
                                      │
                           双缓冲发布新 Actor
```

核心编排位于 [`agent.py`](../source/rl_training/rl_training/online_adaptation/agent.py)。

### 3.1 冻结基础 PPO

基础 PPO 负责名义步态、速度跟踪和基本稳定性。加载检查点后，代码会：

- 获取 RSL-RL 推理策略；
- 对基础策略执行 `requires_grad_(False)`；
- 检查不存在仍允许梯度更新的参数；
- 始终把 PPO 输出作为最终动作的基础部分；
- 在安全事件或残差模块异常时退回纯 PPO 控制。

在线学习不会原地修改基础 PPO 检查点，也不会对 PPO 执行反向传播。

### 3.2 残差控制

最终归一化动作定义为：

```text
a_final = a_base + a_residual
```

残差策略只需要学习基础 PPO 在未见路况上处理不好的部分，例如：

- 某条腿落足过早或过晚；
- 身体在不平整方块上产生额外横滚或俯仰；
- 速度跟踪因碰撞和打滑恶化；
- 某些关节需要小幅增加或减少目标位置。

相比直接在线修改完整步态策略，残差结构显著缩小了探索空间，并保留了随时退回基础控制器的能力。

## 4. 本体感知输入与历史编码

在线策略不读取地形高度图、仿真地形类别、方块区域标志、世界坐标或未来地形信息。每个控制帧由以下 49 个量组成：

| 信号 | 维度 |
|---|---:|
| 机身角速度 | 3 |
| 重力在机身坐标系的投影 | 3 |
| 12 个关节相对默认位置 | 12 |
| 12 个关节速度 | 12 |
| 4 个足端接触状态 | 4 |
| 上一时刻最终动作 | 12 |
| 线速度与偏航速度指令 | 3 |
| 合计 | 49 |

适配器实现位于 [`isaaclab_adapter.py`](../source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py)。

控制频率为 50 Hz，历史缓冲区保存最近 25 帧，即约 0.5 秒本体感知历史：

```text
25 / 50 Hz = 0.5 s
```

一个带 LayerNorm 的 GRU 将 `25 × 49` 的历史序列压缩成 64 维隐变量。这个隐变量不直接表示“方块高度”或“地形类别”，而是从身体最近的动态响应中提取有助于控制的时序信息，例如姿态变化、关节受扰、接触顺序和动作响应。

历史缓冲区和网络定义分别位于：

- [`history.py`](../source/rl_training/rl_training/online_adaptation/history.py)
- [`networks.py`](../source/rl_training/rl_training/online_adaptation/networks.py)

## 5. 残差 Actor 与双 Critic

### 5.1 Actor

Actor 的输入为：

```text
64维历史隐变量 + 49维当前帧 + 12维基础动作
```

网络输出 12 维高斯分布的均值和对数标准差。训练时使用重参数化采样：

```text
u = μ(s) + σ(s) · ε
a_residual = tanh(u) · action_limit
```

Actor 均值输出层使用接近零的初始化，使未训练残差最初接近 0，降低其破坏基础步态的风险。

当前归一化动作限制为：

```text
[0.24, 0.20, 0.20] × 4 条腿
```

结合 Lite3 的动作缩放：

```text
[0.125, 0.25, 0.25] × 4 条腿
```

对应的最大物理关节目标修正约为：

- HipX：`0.24 × 0.125 = 0.03 rad`；
- HipY/Knee：`0.20 × 0.25 = 0.05 rad`。

### 5.2 Critic

Critic 的输入为：

```text
历史隐变量 + 当前帧 + 基础动作 + 残差动作
```

实现包含两个独立 Q 网络 `Q1` 和 `Q2`。更新和 Actor 优化都使用二者的较小值，以降低 Q 值过估计。

## 6. SAC 在线更新

SAC 实现位于 [`sac.py`](../source/rl_training/rl_training/online_adaptation/sac.py)。当前主要超参数为：

| 参数 | 数值 |
|---|---:|
| 折扣因子 `gamma` | 0.99 |
| 目标网络软更新系数 `tau` | 0.005 |
| Actor 学习率 | 3e-4 |
| Critic 学习率 | 3e-4 |
| 熵系数学习率 | 3e-4 |
| 目标熵 | -6.0 |
| 最大梯度范数 | 1.0 |

### 6.1 回放数据

每一步保存以下 transition：

```text
(history,
 current_frame,
 base_action,
 residual_action,
 reward,
 next_history,
 next_frame,
 next_base_action,
 done)
```

回放池容量为 10,000。大部分状态和动作使用 CPU `float16` 保存，采样时转回设备上的 `float32`。奖励和 `done` 使用 `float32`。

回放池实现位于 [`replay_buffer.py`](../source/rl_training/rl_training/online_adaptation/replay_buffer.py)。

### 6.2 更新启动条件

默认至少积累 256 条 transition 后才允许 Critic 更新，每次采样 128 条。单环境 50 Hz 下，积累 256 条数据约需要：

```text
256 / 50 = 5.12 s
```

测试道路包含平地热身段，使 Critic 能在进入 held-out 区域前获得安全 transition。Critic 更新启动只依赖全局回放池大小和安全状态，不依赖方块区域标志。

Actor 更新使用更严格的门槛：只有 `NoveltyGate` 进入 `ADAPTING`，本轮新异回放池达到默认 128 条，且没有安全暂停时才允许更新。这样普通路面阶段可以训练价值函数，但不会提前改变控制策略。

### 6.3 Critic 目标

SAC 的目标 Q 值为：

```text
y = r + gamma · (1 - done) ·
    [min(Q1_target(s', a'), Q2_target(s', a')) - alpha · log pi(a'|s')]
```

Critic 损失为：

```text
L_Q = MSE(Q1(s, a), y) + MSE(Q2(s, a), y)
```

### 6.4 Actor 与熵系数

Actor 损失为：

```text
L_actor = E[alpha · log pi(a|s) - min(Q1(s, a), Q2(s, a))]
```

熵系数 `alpha` 通过可训练的 `log_alpha` 自动调节，使探索强度接近目标熵。在线演示默认使用确定性 Actor 输出，以避免单机连续行走展示被控制噪声主导；需要显式探索时可加 `--stochastic-control`。

### 6.5 目标网络更新

每次成功更新后，目标 Critic 使用 Polyak 平均软更新：

```text
theta_target = (1 - tau) · theta_target + tau · theta
```

### 6.6 新异状态门控

新异检测实现位于 [`novelty.py`](../source/rl_training/rl_training/online_adaptation/novelty.py)。状态机为：

```text
CALIBRATING -> MONITORING -> ADAPTING -> COOLDOWN -> MONITORING
```

输入只包含未来真机可获得或可估计的信号：平面速度误差、横滚/俯仰对应的投影重力范数、非足端接触，以及稠密在线奖励相对正常基线的下降。`heldout_mask`、世界坐标、地形类别和方块区域标志只用于日志分段和评测统计，不进入 `NoveltyGate`、SAC 更新或回滚判断。

校准期建立正常路面基线；监控期要求最近窗口内持续超过阈值才进入适应，单帧尖峰不会启动 Actor 更新；适应期只把当前 episode 的 transition 写入新异回放池并低频更新 Actor；恢复或 rollback 后进入冷却期。安全暂停期间状态机计数不前进，Actor gate 关闭。

## 7. 在线奖励

奖励实现位于 [`reward.py`](../source/rl_training/rl_training/online_adaptation/reward.py)。当前奖励由以下部分组成：

```text
+ 线速度指令跟踪
+ 偏航角速度指令跟踪
- 横滚/俯仰倾斜
- 垂直速度与振动
- 足端滑动
- 落足冲击
- 关节功率
- 残差幅度
- 残差变化率
- 非足端碰撞
```

线速度和偏航跟踪项采用指数核，使接近目标时得到较高奖励。其他项作为连续代价，最终奖励裁剪到 `[-10, 5]`，避免异常信号产生极端训练目标。

奖励接口只依赖未来真机理论上可以测量或估计的状态。当前仿真直接读取机身真实速度；真机部署时必须替换为状态估计器输出，而不应修改 SAC 算法本身。

## 8. 异步更新、双回放与双缓冲发布

控制周期约为 20 ms，而一次神经网络反向传播可能超过单个控制周期。因此在线运行将控制和训练分离：

- 控制线程以 50 Hz 执行动作推理和环境步进；
- 更新线程默认以约 5 Hz 执行 Critic 更新；
- Actor 默认最多约 1 Hz 更新，且只在新异适应 episode 中启用；
- 更新线程修改训练 Actor；
- 控制线程只读取冻结的推理 Actor 副本；
- 成功更新后，参数先写入未使用的备用 Actor；
- 写入完成后再原子切换活动索引。

当前实现包含两个回放池：

- `global_replay` 保存所有安全 transition，供 Critic 持续更新；
- `adaptation_replay` 只保存 `ADAPTING` 状态下的 transition，供 Actor 和熵参数更新。

Critic-only 更新不会改变 Actor 参数、不会增加 Actor 版本，也不会发布新策略。完整 Actor 更新会同时执行本周期的 Critic 更新，避免配置的 Critic 频率被隐式放大。

双缓冲发布器位于 [`publisher.py`](../source/rl_training/rl_training/online_adaptation/publisher.py)，运行线程位于 [`play_online_adaptation.py`](../scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py)。

这种结构避免控制线程读取到只更新了一部分的网络参数，也避免控制循环等待反向传播。

## 9. 安全过滤与参数回滚

安全过滤实现位于 [`safety.py`](../source/rl_training/rl_training/online_adaptation/safety.py)。

### 9.1 动作级保护

每个残差动作依次经过：

1. 每关节残差硬限幅；
2. 相邻控制周期变化率限制，默认不超过 0.02 个归一化动作单位；
3. 与基础动作相加；
4. 转换为物理关节目标；
5. 投影到关节软限位内，并额外保留 0.02 rad 边界；
6. 重新计算最终接受的残差。

### 9.2 状态级保护

以下任一情况会使残差清零并暂停更新：

- 重力 XY 投影范数大于 0.65，即机身倾斜过大；
- 机身高度低于 0.20 m；
- 任一关节速度超过 30 rad/s；
- 非足端部件连续接触地面超过宽限窗口，默认 10 个控制周期；
- 控制动作计算超过配置的超时时间。

发生危险后，需要连续稳定 25 个控制周期，即约 0.5 s，才允许恢复残差和在线更新。

此外，安全层在接近危险边界时会提前禁用残差，但不立即暂停整机：重力 XY 投影范数超过 0.55 或检测到瞬时非足端接触时，当前步残差强制为 0，动作退回到经过关节软限位投影的基础 PPO 动作。这是为了避免残差在接触或大姿态角附近继续放大失稳。

### 9.3 数值级保护

每次 SAC 更新前后都会检查：

- transition 是否包含 NaN 或 Inf；
- loss 是否有限；
- 梯度是否有限；
- 梯度范数是否超过限制；
- Actor、Critic、目标 Critic 和熵参数是否有限。

如果任一检查失败，会恢复此次更新前的完整 SAC 状态，拒绝发布该参数版本。

### 9.4 性能回滚

在线模式会在每个适应 episode 内比较固定长度性能窗口。代价为：

```text
mean(velocity_error + tilt_rms) + 0.25 * nonfoot_contact_rate
```

第一个完整窗口建立 episode 参考；后续窗口比当前最佳值改善至少 5% 时保存完整 SAC 稳定快照；连续两个窗口比最佳值差 10% 以上时回滚。回滚恢复 Actor、Critic、目标 Critic、优化器、熵参数和 Actor 版本号，只影响残差模块，基础 PPO 不变。

2026-07-28 的长时探针发现，性能窗口从进入 `ADAPTING` 时就开始累计，但 Actor 需要等新异回放达到 128 条后才首次更新。这样会把更新前的地形自然变化计入退化耐心，导致 Actor 刚开始学习便被回滚。已确认的修正规则是：更新前只维护最近完整基线；当前 adaptation episode 首次成功 Actor 更新时清除跨越更新边界的部分窗口；只有更新后的完整窗口才允许累计性能退化。详细设计见 [`2026-07-28-lite3-post-update-performance-rollback-design.md`](superpowers/specs/2026-07-28-lite3-post-update-performance-rollback-design.md)。

性能退化判定只在控制线程观察到当前 adaptation episode 的首次成功 Actor 更新后启用。首次更新边界保留最近一个完整更新前基线，清除跨越边界的部分窗口；随后只有完整更新后窗口可以保存稳定快照或累计退化。每个成功发布的 Actor 版本还记录固定诊断 batch 上的确定性残差 MSE、每关节绝对变化的均值与最大值、相对 episode reference Actor 的参数 L2，以及 episode、版本和更新步数。这些指标仅用于观测，不进入 loss、阈值或安全门控；旧 episode 候选和非有限诊断在发布前被拒绝。

安全事件、控制超时或环境终止会立即关闭 Actor gate、清空新异回放池并请求回滚。安全阈值不会为了让某次演示通过而放宽；触发保护的 seed 必须如实计入报告。

## 10. 源地形预训练与未见地形隔离

如果残差 Actor 和 Critic 在进入方块道路时完全随机，单次连续行走中的几百条经验通常不足以获得稳定改善。因此采用两阶段流程。

本节描述的是当前历史实验流程，不等价于“只会平地行走，再通过在线学习获得不平路能力”的严格实验。审计确认当前基础 PPO 的训练地形来源无法从 checkpoint 元数据追溯；45k residual 明确在随机粗糙地面和正反坡面上预训练，并且 seed 7 的 held-out 结果参与过 checkpoint 选择。因此这些产物只能用于验证在线更新基础设施，不能用于证明 flat-only 到台阶能力的因果增益。

### 10.1 源地形预训练

残差 SAC 先在与基础 PPO 相同的源地形分布上预训练：

- 随机粗糙地面；
- 正坡面；
- 反坡面。

预训练脚本位于 [`train_residual_sac.py`](../scripts/reinforcement_learning/online_adaptation/train_residual_sac.py)。预训练阶段增加更强的残差幅度和残差变化率惩罚，使残差保持较小。

### 10.2 地形泄漏保护

源训练环境会显式检查以下地形比例必须为 0：

- boxes；
- pyramid stairs；
- inverse pyramid stairs。

同时，预训练脚本拒绝加载名称中包含 `HeldoutBoxes` 的任务。泄漏检查位于 [`terrain_guards.py`](../source/rl_training/rl_training/online_adaptation/terrain_guards.py)。

### 10.3 未见方块道路

评测使用独立的单环境道路配置：

- 3 m 平地前缀；
- 0.4 m 网格宽度；
- 方块高度随机范围约为 ±0.04 至 ±0.12 m；
- 固定 0.5 m/s 前进指令；
- 禁用课程学习、外力扰动和初始化随机化；
- 不通过自动重置积累多回合经验。

配置和地形生成器位于：

- [`heldout_boxes_env_cfg.py`](../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/heldout_boxes_env_cfg.py)
- [`heldout_boxes.py`](../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/heldout_boxes.py)

### 10.4 flat-only 目标的下一阶段隔离要求

最终目标实验必须重新建立初始化和评测边界：基础 PPO 从 `Flat-Deeprobotics-Lite3-v0` 训练并携带可验证 provenance；residual 只能在 plane-only 环境预训练或保持零初始化；plane 断言采用 `terrain_type == "plane"` 且 `terrain_generator is None` 的正面检查。开发地形 seed 与封存测试 seed 必须分离，目标几何 seed 随评测 seed 改变。在线模式还需要跨 reset 保存完整 SAC、回放池和稳定快照，使机器狗能够通过多个受保护 episode 累积低台阶经验，并在回到平地后验证能力保持。

## 11. 三组对照实验

相同基础策略、道路种子、初始状态和速度指令下运行三种模式：

| 模式 | 基础 PPO | 残差 Actor | 在线梯度更新 |
|---|---|---|---|
| `base` | 冻结 | 输出为 0 | 关闭 |
| `residual_frozen` | 冻结 | 加载预训练残差，确定性推理 | 关闭 |
| `online` | 冻结 | 加载预训练残差，受限随机探索 | 开启 |

三组对照的意义是区分：

- 基础 PPO 本身的能力；
- 预训练残差本身带来的改善；
- 同一次行走中的梯度更新额外带来的改善。

评测脚本会保存速度误差、姿态 RMS、非足端碰撞、安全回退、更新次数、参数版本和前后参数差异。批量比较入口位于 [`compare_online_adaptation.py`](../scripts/reinforcement_learning/online_adaptation/compare_online_adaptation.py)。

### 11.1 当前可复现实验记录

2026-07-27 使用 `/home/tjx/Lite3_rl_deploy/policy/ppo/policy.onnx` 导入的基础 PPO、源地形预训练残差 `lite3_deploy_base_residual_pretrain_256env_2000_20260727/latest.pt`，在 `HeldoutBoxes-Deeprobotics-Lite3-v0`、seed 7、`--forward-speed 0.3`、2000 步单次连续行走中得到：

| 模式 | run 目录 | 步数 | reset | 参数变化 | heldout samples | heldout 速度误差前半 | heldout 速度误差后半 |
|---|---|---:|---:|---|---:|---:|---:|
| `base` | `heldout_deploy_base_fwd03_seed7_20260727` | 2000 | 0 | 否 | 485 | 0.318 | 0.248 |
| `residual_frozen` | `heldout_deploy_residual_frozen_fwd03_seed7_20260727` | 2000 | 0 | 否 | 0 | N/A | N/A |
| `online` | `heldout_deploy_online_guard_fwd03_seed7_20260727` | 2000 | 0 | 是 | 804 | 0.252 | 0.213 |

该结果说明：当前推荐演示速度下，在线版本在同一次连续行走中执行了 485 次 SAC 更新，Actor 版本从 742 增至 1227，`residual_before.pt` 与 `residual_after.pt` 不同，并且 heldout 后半段速度误差低于前半段。冻结残差对照虽未摔倒，但该 seed/速度下没有进入 heldout 统计区间，不能证明未见地形适应。

上述记录早于 novelty gate、held-out 边界修正和控制路径预热，只能作为历史证据，不能作为当前配置的正式验收。2026-07-28 在修正后的 seed 7、0.3 m/s、100 ms 控制期限下得到：

| 模式/检查点 | 步数 | reset | safety violation | heldout samples | 速度误差前/后半 | 倾角前/后半 | Actor 更新后次数 | 最终参数变化 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| `base` | 2000 | 0 | 否 | 1562 | 0.297 / 0.284 | 0.040 / 0.107 | 0 | 否 |
| `residual_frozen` / 50k | 1577 | 1 | 是 | 1121 | 0.298 / 0.198 | 0.048 / 0.055 | 0 | 否 |
| `online` / 50k | 1202 | 1 | 是 | 746 | 0.287 / 0.283 | 0.038 / 0.077 | 15 | 是 |
| `residual_frozen` / 45k | 2000 | 0 | 否 | 1554 | 0.301 / 0.279 | 0.062 / 0.150 | 0 | 否 |
| `online` / 45k | 2000 | 0 | 否 | 1554 | 0.295 / 0.291 | 0.068 / 0.073 | 4 | 否，更新均被回滚 |

50k residual 自身不能通过 2000 步 frozen 安全基线。45k residual 可使 frozen 和 online 都安全跑满 2000 步，但 online 速度误差比 frozen 后半程高约 4.5%，两次性能回滚使最终参数恢复到初始值。因此当前结果证明了安全改进和在线更新路径有效，尚未证明在线学习带来最终性能收益。

2026-07-29 在修正首次 Actor 更新因果边界、候选事务发布、episode 并发隔离和独立 Actor 事件流后，使用同一个 45k residual、seed 7、0.3 m/s、100 ms 控制期限得到：

| 运行 | 步数 | reset / fall / safety violation | Actor 更新 / 事件 | episode 更新分布 | 性能回滚 | heldout 速度误差前/后半 | 最终参数变化 |
|---|---:|---|---:|---|---:|---:|---|
| `online_seed7_750_postupdate_drift_eventstream_20260729_cuda2` | 750 | 0 / 否 / 否 | 10 / 10 | 1: 10 | 0 | 0.282 / 0.287 | 是 |
| `online_seed7_2000_postupdate_drift_eventstream_20260729_cuda2` | 2000 | 0 / 否 / 否 | 62 / 62 | 1: 25, 2: 37 | 2 | 0.293 / 0.334 | 是 |

750 步运行中首次 Actor 更新发生在第 643 步，控制线程从第 644 步才启用性能评估。2000 步运行的两个 episode 分别从第 644 和 1196 步开始更新，在 rollback 前分别积累了 299 和 349 个更新后评估步，超过两个完整 50 步窗口。`metrics.jsonl` 严格保持一控制步一行；`actor_updates.jsonl` 对每次成功发布记录完整且有限的漂移标量，事件行数与 summary 更新计数逐项一致。

这组结果通过了安全、更新因果、回滚时序和事件可审计性门槛，但没有通过性能门槛：2000 步 heldout 后半速度误差比前半高约 14.2%。它不能证明机器狗已通过在线学习获得台阶或不平路能力，也不能替代 flat-only 初始化下的 base / frozen / online 对照。

## 12. 参考资料及其与本实现的关系

### 12.1 直接采用的算法思想

1. **Soft Actor-Critic Algorithms and Applications**  
   Tuomas Haarnoja 等，2018。  
   <https://arxiv.org/abs/1812.05905>

   当前实现直接采用其最大熵 Actor-Critic、双 Critic、目标网络、重参数化采样和自动熵温度调节思想。

2. **Residual Reinforcement Learning for Robot Control**  
   Tobias Johannink 等，2018。  
   <https://arxiv.org/abs/1812.03201>

   当前实现采用“基础控制信号 + 学习残差”的结构。区别在于原论文主要以传统反馈控制器作为基础，而本项目使用冻结 PPO 作为基础控制器。

### 12.2 结构启发与对照

3. **RMA: Rapid Motor Adaptation for Legged Robots**  
   Ashish Kumar 等，RSS 2021。  
   <https://arxiv.org/abs/2107.04034>

   当前实现借鉴了通过本体感知历史推断环境变化的思想。区别是 RMA 部署时主要快速推断适应隐变量，而本项目在同一次行走中通过 SAC 真正更新网络权重。

4. **AWAC: Accelerating Online Reinforcement Learning with Offline Datasets**  
   Ashvin Nair 等，2020。  
   <https://arxiv.org/abs/2006.09359>

   AWAC 强调使用先验数据或策略减少在线探索成本。本项目同样采用源地形预训练后再在线微调的暖启动思路，但没有使用 AWAC 的优势加权行为克隆更新，实际更新算法仍为 SAC。

5. **DayDreamer: World Models for Physical Robot Learning**  
   Philipp Wu 等，2022。  
   <https://arxiv.org/abs/2206.14176>

   DayDreamer 展示了世界模型在真实机器人在线学习中的样本效率。本项目没有实现世界模型或想象规划，而是选择计算和安全边界更简单的残差 SAC 路线。

### 12.3 2023 至 2025 年的直接相关工作

6. **DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning**
   I Made Aswin Nahrendra 等，ICRA 2023。
   <https://arxiv.org/abs/2301.10602>

   DreamWaQ 通过本体感知历史形成隐式地形表示，并在真实室外环境中验证长距离行走。它在部署时不更新策略权重，因此与本项目的 `residual_frozen` 历史条件策略属于重要对照，而不是“真正在线梯度更新”的证据。本项目现有 GRU frozen residual 已覆盖部分隐式适应能力，但尚不是对 DreamWaQ 训练方法的完整复现。

7. **Efficient Online Reinforcement Learning with Offline Data (RLPD)**
   Philip J. Ball 等，ICML 2023。
   <https://arxiv.org/abs/2302.02948>

   RLPD 表明，离线初始化后进行在线 RL 时，已有数据与新数据的混合方式会显著影响稳定性和样本效率。本项目目前只加载预训练 SAC 状态，不携带源地形 replay；在线 Critic 只从当前运行的全局 replay 采样。这是当前方案与成熟 offline-to-online 流程之间的明确缺口。

8. **Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning**
   Mitsuhiko Nakamoto 等，NeurIPS 2023。
   <https://arxiv.org/abs/2303.05479>

   Cal-QL 强调离线 Critic 的数值尺度和校准对在线微调至关重要。本项目还没有 conservative/calibrated Q 初始化，也没有专门监控源地形 Q 值与在线 Q 值的尺度偏移。第一步应先记录 Q 分布和 target 偏差，再决定是否引入 Cal-QL/CQL 类目标。

9. **Synaptic Motor Adaptation: A Three-Factor Learning Rule for Adaptive Robotic Control in Spiking Neural Networks**
   Samuel Schmidgall、Joe Hays，2023。
   <https://arxiv.org/abs/2306.01906>

   该工作研究四足机器人实时参数可塑性，是少数比 RMA 更接近“运行中改变参数”的腿式适应方法。它采用元优化的局部突触学习规则和脉冲网络，不适合直接替换当前 SAC，但证明部署时权重适应需要专门设计更新规则，不能仅依赖通用离线超参数。

10. **Grow Your Limits: Continuous Improvement with Real-World RL for Robotic Locomotion**
    Laura Smith、Yunhao Cao、Sergey Levine，2023。
    <https://arxiv.org/abs/2310.17634>

    APRL 在真实四足机器人上使用策略正则化调节探索，使策略能够持续训练和适应动力学变化。它直接支持在 residual Actor 在线更新中加入相对稳定策略的行为锚定，并记录策略漂移，而不是只依靠事后性能回滚。

11. **LoopSR: Looping Sim-and-Real for Lifelong Policy Adaptation of Legged Robots**
    Peilin Wu 等，2024，IROS 2025。
    <https://arxiv.org/abs/2409.17992>

    LoopSR 使用真实轨迹估计数字孪生参数，再回到仿真中继续训练。它不执行同一控制回合内的在线梯度更新，但为真机阶段提供了更保守的长期适应路线：真实机器人负责采集，策略改进在可重放的仿真环境中完成。

12. **Bridging Adaptivity and Safety: Learning Agile Collision-Free Locomotion Across Varied Physics**
    Yichao Zhong 等，2025。
    <https://arxiv.org/abs/2501.04276>

    BAS 使用物理参数估计、恢复策略和 reach-avoid value 管理未知动力学下的安全切换。当前项目的硬阈值过滤和冻结 PPO 回退仍应保留为独立安全底线；learned safety 可以作为未来附加层，但不能替代硬安全控制器。

13. **Fine-Tuning Hard-to-Simulate Objectives for Quadruped Locomotion: A Case Study on Total Power Saving**
    Ruiqian Nai 等，ICRA 2025。
    <https://arxiv.org/abs/2502.10956>

    该工作先用真实数据学习难以仿真的目标模型，再在仿真中改进策略。对于未来 Lite3 功耗、噪声和硬件磨损目标，这比直接在真机上用稀疏、安全敏感奖励更新 Actor 更合适。

### 12.4 2026 年预印本观察

14. **Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged Locomotion**
    Gianluca Sabatini、Chenhao Li、Marco Hutter，2026 预印本。
    <https://arxiv.org/abs/2605.24975>

    该工作直接针对腿式 SAC，提出策略初始化、timeout-aware Critic target 和 multi-step return。当前实现已有预训练初始化，但仍使用一步 TD target，transition 只保存单个 `done`，没有明确区分 termination 与 truncation。由于该工作仍是新预印本，本项目应先独立验证这些机制，而不是把其结果视为已确立标准。

15. **Residual Control for Fast Recovery from Dynamics Shifts**
    Nethmi Jayasinghe 等，2026 预印本。
    <https://arxiv.org/abs/2603.07775>

    该工作保持名义策略冻结，通过受限残差通道和稳定性对齐门控制恢复。它不在部署时重新训练名义策略，但其残差幅度、方向一致性和性能条件门控与本项目安全 residual 架构高度相关，可作为后续残差发布约束的研究参考。

### 12.5 文献对当前实现的分阶段结论

近期工作不会被一次性照搬。当前项目按以下顺序吸收：

1. 先修正首次 Actor 更新与性能回滚窗口的因果边界，并记录策略相对稳定快照的漂移。
2. 再设计源地形 replay 持久化、源/在线批次混合、termination/truncation 区分和 timeout-aware target。
3. 在单步 target 基线稳定后，独立评估 multi-step return 和 Q 值校准，避免同时改变多个 Critic 因素。
4. 真机阶段优先考虑 LoopSR 和 hard-to-sim objective 的“真实数据回流仿真”路线；任何 learned safety 都只能叠加在硬安全控制器之上。

完整分阶段设计见 [`2026-07-29-lite3-literature-aligned-online-adaptation-hardening-design.md`](superpowers/specs/2026-07-29-lite3-literature-aligned-online-adaptation-hardening-design.md)。

### 12.6 工程基础

- **Isaac Lab**：仿真环境、传感器、地形和任务配置。  
  <https://github.com/isaac-sim/IsaacLab>
- **RSL-RL**：现有 Lite3 PPO 的训练和检查点加载。  
  <https://github.com/leggedrobotics/rsl_rl>
- **DeepRobotics RL Training**：Lite3 模型、动作缩放、基础 PPO 配置和原始地形任务。  
  <https://github.com/DeepRoboticsLab/rl_training>

## 13. 当前实现状态

已经完成：

- 冻结 PPO 与残差 SAC 的组合结构；
- 49 维本体感知接口和 25 帧历史缓冲；
- GRU 残差 Actor、双 Critic 和目标 Critic；
- CPU 回放池和 SAC 更新；
- NaN/Inf 拒绝、梯度裁剪和参数回滚；
- 动作限幅、变化率限制、关节限位和安全暂停；
- 双缓冲 Actor 发布；
- 源地形泄漏检查；
- 独立方块道路和三模式对照脚本；
- 运行日志、前后参数保存和多种子汇总入口；
- RSL-RL 5 配置转换和冻结 Actor/Critic 兼容层；
- 新异状态门控、双回放、Critic/Actor 更新解耦和 episode-local 性能回滚；
- Isaac Lab 环境创建、PPO 加载和单步控制集成验证；
- 源地形残差 SAC 正式预训练，256 环境、2000 步，产物位于 `lite3_deploy_base_residual_pretrain_256env_2000_20260727/latest.pt`；
- 冻结残差连续行走验证，参数保持逐位不变；
- 在线适应模块测试（157 通过，2 个 Isaac-only 用例在普通测试中跳过；使用当前 worktree `PYTHONPATH` 并关闭无关第三方 pytest 插件自动加载）；
- 五种子资格入口改为每个种子独立 Isaac 进程，避免仿真单例重建卡死；
- `/home/tjx/Lite3_rl_deploy` 官方部署 PPO 导入为 RSL-RL 兼容检查点，并通过五种子源地形资格测试；
- seed 7、0.3 m/s heldout 单次连续行走在线演示曾跑满 2000 步并产生残差参数变化；当前 novelty-gated 正式对照仍在调试，不能视为最终通过结果。
- held-out 指标边界已与实际轨道边界对齐，控制路径已在计时循环前预热，三种模式不再因 CUDA/GRU 首次调用产生伪 `control_timeout`。
- 45k residual checkpoint 已在 seed 7 的 frozen 和 online 模式下分别安全跑满 2000 步；online 最终性能仍未达到验收门槛。
- 性能退化窗口已改为首次成功 Actor 更新后启用；每个成功发布版本记录固定诊断 batch 上的动作与参数漂移标量，并在发布前拒绝旧 episode 候选和非有限诊断。
- Actor 候选的 episode 检查、gate 状态和双缓冲发布已形成原子边界；回放清空由 updater 命令线程串行执行，旧 episode 命令不会破坏新 episode。
- `metrics.jsonl` 保持一控制步一行，`actor_updates.jsonl` 逐次持久化并 flush 成功发布事件；750/2000 步 GPU 回归中的事件数与 Actor 更新数完全一致。

尚未完成实证验证：

- 同种子三组演示录像；
- 十个未见方块种子的统计对照；
- novelty-gated 配置下 held-out 方块道路的正式通过结果；
- 真机部署与硬件安全验证。

## 14. 已知限制与后续修正

当前代码仍有以下需要在仿真验证阶段处理的细节：

1. 0.5 m/s、seed 7 的 heldout 压力测试仍会在 300 步左右因姿态角过大失败；当前可演示配置为 0.3 m/s。
2. 安全暂停 transition 已通过 `valid_mask` 排除在 replay 训练之外，但 safety pause 仍会计入日志，需要在展示时解释它是保护事件，不是在线更新样本。
3. 方块区域坐标不会进入 Actor、奖励、更新启动条件或性能回滚；`heldout_mask` 保留为评测统计字段，必须继续防止它流入学习决策。
4. 网络使用 LayerNorm，但尚未单独保存完整的输入归一化统计。
5. 更新线程不能强制中断一次正在执行的反向传播；需要通过计算预算、独立进程或 watchdog 进一步加强超时隔离。
6. 当前残差限幅和奖励权重只完成了单种子连续道路验证，尚未通过十种子统计。
7. 仿真奖励使用真实机身速度，真机部署前必须验证状态估计误差下的稳定性。
8. 预训练 checkpoint 不包含源地形 replay，在线 Critic 无法执行 RLPD 式源/在线数据混合。
9. Actor 更新已记录相对 episode reference Actor 的动作和参数漂移，但这些指标仅用于观测，尚未加入 KL、动作或参数锚定；当前仍只能通过限幅和事后回滚控制策略漂移。
10. SAC target 使用一步 return，transition 只保存 `done`，尚未显式区分真实终止和时间截断。
11. 当前安全层是硬阈值、动作投影和基础 PPO 回退，没有 reach-avoid value 或独立恢复策略；这在仿真阶段是明确取舍，不应被描述为完整 learned safety。
12. 当前基础 PPO checkpoint 缺少平地训练 provenance，45k residual 已见过 rough 和 slopes；它们不满足 flat-only 初始化要求。
13. seed 7 的 HeldoutBoxes 表现参与过 45k/50k checkpoint 选择，且当前目标 terrain generator 固定 seed 7；需要独立开发/封存几何 seed 才能形成严格 held-out 结论。
14. 当前脚本在环境终止后结束，尚未跨 reset 持久化完整 SAC、回放池和稳定快照；因此还不能通过多个受保护 episode 累积低台阶学习。

## 15. 真机迁移边界

当前功能只授权 Isaac Lab 仿真验证，不授权无保护的真机在线探索。未来部署到真实 Lite3 时至少需要：

- 吊架或保护绳；
- 物理急停；
- 独立于学习器的硬件/软件安全控制器；
- 控制周期 watchdog；
- 通信中断后自动退回冻结 PPO；
- 更低的残差限幅；
- 默认使用确定性 Actor；
- 分阶段低能量、低速度验证；
- 操作员全程监督。

在完成上述措施和独立安全验证前，不应在真实 Lite3 上启用持续在线随机探索或梯度更新。
