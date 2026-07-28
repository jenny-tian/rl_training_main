# Lite3 新异状态触发在线适应与性能回滚设计

## 1. 背景

当前 Lite3 单次连续行走演示会在回放池达到 256 条有效 transition 后立即开始完整 SAC 更新。实测中，actor 在约第 294 步开始变化，而机器人要到约第 1357 至 1564 步才持续进入未见方块区域，因此策略会在普通路面上提前改变。

现有性能回滚每累计 250 个 held-out 样本检查一次，并要求连续退化 5 次。一次 2000 步演示通常只有 461 至 648 个 held-out 样本，所以该回滚路径在当前演示长度内无法触发。此外，它依赖仿真世界坐标生成的 held-out 标签，不能直接迁移到真机。

本设计将在线学习改为：critic 持续学习，actor 只有在本体感知信号持续显示新异路况时才学习和发布；性能判断与回滚只依赖真机可获得的信号。

## 2. 目标与非目标

### 2.1 目标

- 单次连续行走，不重置环境积累多回合经验。
- 普通路面阶段冻结 actor，避免在遇到未知路况前改变步态。
- critic 持续利用全部安全 transition 学习价值函数。
- 用速度误差、机身倾斜、非足端接触和在线奖励检测持续新异状态。
- actor 只使用近期新异状态数据进行低频更新。
- 在性能连续退化或触发安全保护时恢复最近稳定参数。
- 保持 `heldout_mask` 仅用于评测统计，不进入触发、学习或回滚逻辑。
- 保留未来在 Lite3 真机上运行所需的观测接口和安全边界。

### 2.2 非目标

- 不在线更新冻结的基础 PPO。
- 不使用世界坐标、地形类别、方块区域标记或未来地形信息。
- 不在本次修改中引入视觉、激光雷达或地形高度扫描。
- 不通过提高残差动作上限换取短期指标改善。
- 不改写残差 SAC 的离线预训练流程。

## 3. 方案选择

考虑过以下方案：

1. 所有 SAC 更新都由新异状态触发。实现简单，但进入新路况后 critic 仍是冷启动，响应较慢。
2. critic 持续学习，actor 由新异状态触发。普通路面不会提前改变控制策略，同时 critic 能提前建立价值估计。本设计采用该方案。
3. 单独用 TD-error 触发。该信号依赖 critic 校准，训练早期容易把价值估计误差误判为地形新异，不作为第一版主触发器。

## 4. 运行状态机

新增独立的 `NoveltyGate`，状态为：

```text
CALIBRATING -> MONITORING -> ADAPTING -> COOLDOWN -> MONITORING
                                  |            ^
                                  +--rollback--+
```

### 4.1 CALIBRATING

- 收集前 256 个未暂停的有效控制步。
- 建立速度误差和倾斜的指数移动均值及尺度下限。
- transition 写入全局回放池，critic 达到最小回放量后可以更新。
- actor、熵参数和已发布 actor 版本保持不变。

### 4.2 MONITORING

- 持续更新缓慢变化的正常状态基线。
- 对每个有效步计算新异分数。
- 最近 20 个有效步中至少 10 个超过进入阈值时进入 `ADAPTING`。
- 单次异常接触、单帧姿态尖峰或传感器噪声不能直接启动 actor 更新。

### 4.3 ADAPTING

- 暂停更新正常状态基线，避免将未知路况迅速吸收到基线中。
- 全局回放池继续接收数据并供 critic 更新。
- 新异回放池只接收当前适应段的安全 transition。
- 新异回放池达到 128 条后，actor 和熵参数最多每秒更新一次。
- 每次进入该状态都创建一个适应 episode，并保存完整稳定快照。
- 新异分数持续低于恢复阈值 100 个有效步后进入 `COOLDOWN`。

### 4.4 COOLDOWN

- actor 停止更新，critic 继续更新。
- 冷却期固定为 100 个有效步。
- 冷却结束后返回 `MONITORING`，重新出现持续新异状态才允许下一轮 actor 更新。

安全暂停期间状态机计数不前进，任何 actor 更新都被禁止。

## 5. 新异信号

所有触发输入都来自现有真机可替换接口：

- 平面线速度指令跟踪误差；
- 投影重力横向分量的范数，代表横滚/俯仰倾斜；
- 非足端接触；
- 已有稠密在线奖励相对正常基线的下降。

连续信号使用带尺度下限的标准化超额值，避免校准期方差过小时产生极大分数：

```text
velocity_excess = max(0, (velocity_error - baseline_mean) / max(baseline_std, 0.05))
tilt_excess     = max(0, (tilt_rms - baseline_mean) / max(baseline_std, 0.02))
reward_drop     = max(0, (baseline_reward - reward) / max(baseline_std, 0.10))
```

新异分数取三个连续超额值的最大值，并将非足端接触作为附加证据。默认进入阈值为 2.5，恢复阈值为 1.0。接触必须在最近 10 步中至少出现 3 次，除非安全过滤器已判定为危险状态；危险状态走立即回滚路径，不走普通新异触发路径。

阈值全部通过配置暴露并写入运行日志，但第一版不做自动阈值搜索。

## 6. 双回放与更新数据流

新增两个用途明确的回放来源：

- `global_replay`：保存所有未暂停、安全的 transition，供 critic 更新。
- `adaptation_replay`：只保存 `ADAPTING` 状态下的 transition，进入新的适应 episode 时清空，供 actor 和熵参数更新。

控制与学习数据流为：

```text
控制步 -> 安全 transition -> global_replay -> critic update (持续)
                    |
                    +-> NoveltyGate == ADAPTING
                              |
                              +-> adaptation_replay -> actor/alpha update (1 Hz)
```

`SACUpdater` 接受 critic batch 和可选 actor batch。critic-only 更新不改变 actor、不发布参数，也不增加 actor 版本。actor 更新成功后才增加版本并通过双缓冲 publisher 原子发布。

第一版不使用优先经验回放；新异专用回放池已经解决普通路面样本淹没适应样本的问题。

## 7. 稳定快照与回滚

稳定快照必须包含：

- actor、critic 和 target critic；
- actor、critic 和 alpha 优化器；
- 熵参数；
- actor 版本号。

进入 `ADAPTING` 时保存初始快照。适应期间每 50 个有效步计算一次最近 50 步的综合代价：

```text
cost = mean(velocity_error + tilt_rms) + 0.25 * nonfoot_contact_rate
```

比较只发生在同一个适应 episode 内：

- 第一个完整窗口建立 episode 参考值。
- 新窗口比当前最佳值改善至少 5% 时保存新的稳定快照。
- 新窗口比当前最佳值恶化超过 10% 时记一次退化。
- 连续两个完整窗口退化时立即回滚，并进入 `COOLDOWN`。
- 在 `ADAPTING` 状态发生安全暂停、控制超时、动作保护性降级或环境终止时立即回滚，不等待窗口。
- 在其他状态发生安全事件时不重复加载参数，只保持 actor gate 关闭、由基础 PPO 接管并进入 `COOLDOWN`。

回滚后重新发布快照中的 actor，清空本轮新异回放池和退化计数。基础 PPO 始终保持冻结且不参与回滚。

## 8. 并发与一致性

控制线程拥有 `NoveltyGate` 状态、滚动指标和 transition 路由决策。更新线程只能读取线程安全的状态快照。

- `actor_update_enabled` 使用线程安全事件传递。
- SAC 更新、快照保存和回滚继续共用更新锁，防止加载状态时并发修改优化器。
- 发布器只接收完整、有限值检查通过的 actor。
- 回滚完成前控制线程继续使用上一个已发布的完整 actor，不读取半更新参数。
- 停止事件、异常传播和线程 join 继续沿用现有运行时机制。

## 9. 配置与日志

演示 CLI 增加以下配置，默认值对应本设计：

- `--novelty-calibration-steps 256`
- `--novelty-enter-threshold 2.5`
- `--novelty-exit-threshold 1.0`
- `--novelty-enter-count 10`
- `--novelty-enter-window 20`
- `--novelty-recovery-steps 100`
- `--adaptation-min-replay-size 128`
- `--actor-updates-per-second 1.0`
- `--rollback-window 50`
- `--rollback-patience 2`
- `--rollback-degradation-ratio 0.10`
- `--stable-improvement-ratio 0.05`
- `--adaptation-cooldown-steps 100`

现有 `--updates-per-second` 改为 critic 更新频率，并保持兼容。

每步日志增加：

- `adaptation_state`；
- `novelty_score` 和各分量；
- `actor_update_enabled`；
- `critic_updates`、`actor_updates`；
- `adaptation_replay_size`；
- `adaptation_episode`；
- `rollback_reason`。

摘要增加状态进入次数、actor 首次更新步数、进入新异状态前的 actor 更新数、性能回滚次数、安全回滚次数和各状态持续步数。

`in_heldout` 继续存在，但 `config.json` 必须保留 `"heldout_mask_usage": "metrics_only"`。

## 10. 测试策略

### 10.1 单元测试

- 校准完成前 actor gate 始终关闭。
- 短暂尖峰不能触发，持续超阈值能触发 `ADAPTING`。
- 恢复计数和冷却期状态转换正确。
- 安全暂停冻结状态机计数并关闭 actor gate。
- critic-only 更新不改变 actor 参数和 actor 版本。
- actor 更新只使用新异回放数据并发布新版本。
- 两个连续退化窗口触发完整状态回滚。
- 安全事件立即回滚。
- `heldout_mask` 不传入 `NoveltyGate`、SAC 更新或回滚评估。

### 10.2 集成测试

- 使用假环境运行完整状态序列，验证控制线程和更新线程没有竞态或死锁。
- 普通路面阶段达到最小全局回放量后，critic 有更新但 actor 版本保持不变。
- 新异状态持续出现后，新异回放池增长，达到门槛后 actor 参数发生变化。
- 强制退化后 actor 参数逐位恢复为稳定快照。
- 现有 base、residual frozen 和 online 三种模式保持兼容。

### 10.3 仿真验收

先使用当前最佳候选 `residual_00050000.pt`，在相同道路、速度和控制超时设置下进行三组对照。单次连续行走需要满足：

- 运行 2000 步且不摔倒、不重置；
- held-out 样本不少于 100；
- 进入新异状态前 actor 更新数为 0；
- 进入新异状态后至少有一次 actor 更新；
- 在线模式后半段的速度误差和倾斜不能同时劣于冻结残差模式；
- 所有回滚均记录明确原因且参数恢复可验证。

单种子通过后再运行至少 3 个具有有效 held-out 覆盖的种子。未覆盖 held-out 区域的种子只计为协议无效，不能计入成功或失败。

## 11. 真机迁移约束

- 新异检测器只能读取状态估计器、IMU、关节和接触信号。
- actor 默认采用确定性输出，残差硬限幅和变化率限制保持启用。
- critic 更新可在伴随计算机持续运行；actor 发布必须经过参数有限值检查和稳定快照保护。
- 通信超时、控制超时、急停或安全监控触发时立即停止 actor 更新并回退到冻结基础 PPO。
- 首次真机实验采用保护绳或吊架，物理急停独立于学习进程。

## 12. 完成定义

只有在以下条件全部满足时，本功能才算完成：

- 新异检测、状态机、双回放、critic/actor 解耦和回滚均有自动化测试。
- 演示脚本不使用 held-out 标签驱动任何学习决策。
- 普通路面 actor 不更新，新异状态下 actor 确实发生梯度更新。
- 连续性能退化与安全异常都能恢复完整稳定状态。
- 单种子及多种子仿真结果按验收标准如实保存和报告。
- 文档、测试和代码提交并推送到项目对应的 GitHub 私有仓库。
