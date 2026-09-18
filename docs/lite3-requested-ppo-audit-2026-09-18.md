# Lite3 请求动作 PPO 与续训参数审计

本轮继续追踪在线训练失败。v5 的 20 回合试训练严格成功 9 次，最近 10 回合成功率 50%，其他四个 development 地形通过 3 个，尚未达到稳定训练标准。训练和冻结评估均沿用严格终点、足端越界及接触协议；本轮没有修改安全阈值。

## 已核实的问题

### 请求动作被安全投影后，不应按投影结果筛选 PPO 样本

worker 从 Gaussian 采样请求动作，并保存该请求及其采样 log probability。安全过滤器将请求转换成实际执行动作。此前 `~projected & ~paused` 又将经过投影的请求从 Actor、Critic、优势归一化和 KL 统计中排除，这属于依赖动作的条件筛选，会引入偏差。

保留请求动作作为策略随机变量、把固定安全过滤器及其历史看作环境转移，就可以使用原请求的 likelihood。不能把执行动作代入原 Gaussian 的 log probability，也不需要对投影计算 Jacobian。

一个确定性积分反例：标准正态请求 `u`，执行 `clip(u, -0.5, 0.5)`，奖励为执行值。在均值 0 处，完整请求分布的策略梯度为约 0.382925；只保留没有被限幅的请求后，条件估计约为 0.080589。

当前 std=0.01、相邻残差变化上限 0.03 时，在相邻独立噪声的理想近似下，12 个关节至少一个超限的概率约 33.9%，与实测约 35% 接近。因此不能把投影比例本身当作异常，更不能通过放宽安全上限隐藏问题。

修复由 `ppo_safety.requested_action_rollout` 统一收集请求和原概率。非暂停的投影行完整参与学习。暂停由动作前安全状态决定，执行与本次请求无关，仍排除其 Actor 梯度；当前 updater 共用样本掩码，因此暂停行也不直接监督 Critic，但保留在 GAE 轨迹中。这一剩余限制没有被包装成“所有真实状态均已参与 Critic 学习”。

### checkpoint 恢复覆盖显式指定的探索幅度和学习率

worker 原先先设置 CLI `action_std`，再恢复 checkpoint；后者覆盖分布参数。Adam 的 `load_state_dict` 同时覆盖 `param_groups.lr`，而 `PPOConfig` 和 `config.json` 仍记录新的请求值。CPU 复现中，配置请求 std=0.005、Actor LR=1e-5、Critic LR=3e-4，恢复后实际仍是 std=0.01、两种 LR=5e-5。

新增 `apply_ppo_runtime_config` 在恢复后、空 rollout 边界应用参数，同时更新 active 和既有 stable 分支的分布及 LR。模型权重、Adam 动量、候选验证状态保持；分布改变时递增策略版本并刷新控制副本，拒绝给已有 rollout 更换分布。冻结评估使用只读 `read_ppo_runtime_config`。运行开始即保存实际生效参数，不再只登记 CLI 请求。

### 学习诊断不足

新增更新前 value/return 均值、原始 advantage 均值和标准差、解释方差、唯一有效样本数及 rollout 总行数。常量 returns 的解释方差写为 JSON null；保留旧 `num_samples` 的跨 epoch 累计语义，避免把重复优化次数当作新环境样本。

## 没有据此下定论的问题

`audit_v6_20260918/stall` 在 x 约 0.5 m 处四足静止，后半段奖励约 -1.69，最大倾斜范数约 0.24，且无安全暂停或非足接触。它不是“站着持续领取正奖励”，也没有触发倾斜阈值关闭全部学习。有限幅探索能否脱离该状态仍需训练验证。

terrain seed 109 的冻结失败伴随 yaw 偏移和横向漂移；body frame 的前进指令校验正确，不等于 world frame 的路线保持正确。现有单帧本体观测不包含全局路线状态。不能仅修样本掩码就宣称已解决路线保持或泛化。

v5 前三个冻结探针失败后，因为尚无已验证恢复点，下一回合重新初始化了 Actor/Critic/Adam。第 20 回合已有通过探针的候选，因此本轮另从该 checkpoint 启动 v5 续训对照，记录此前 20 回合、36,667 个训练交互步的成本。

## 验证与版本

完整 CPU 回归：`1186 passed, 7 skipped`。投影集成测试实际使用 SafetyFilter、生产收集函数和 PPO session；把旧掩码在内存中恢复后，该测试按预期失败。恢复参数测试同时验证控制副本、stable rollback、优化器动量及非法参数不产生部分修改。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests/online_adaptation
```

新学习版本为 `requested_action_ppo_runtime_config_v6`，采样合同为 `requested_gaussian_through_fixed_safety_v1`。奖励仍为 `progress_stall_completion_body_feet_failure_v3`；完成资格仍为 `in_lane_obstacle_interaction_edge_slip_v3`，协议 SHA-256 仍为 `6243db772970452fd70811e7160739e277faa98b05f6c0398cad1e9eafd4f94d`。新实验目录与旧版分开。

仿真验证目录：`artifacts/semester_2026/09/lora_ppo_multiepisode/audit_requested_v6_20260918_2243/`。其中 `source_snapshot.json` 登记基准提交和覆盖文件哈希，`launch_metadata.json` 保存各组完整参数及输入 checkpoint 哈希。两个物理样例已写出 checkpoint 和 summary；视频在进程收尾后另行检查完整性。CPU 或单回合通过均不代表已经稳定学会地形。

持续验收使用既定标准：最近 10 回合至少 80% 严格成功、至少连续 3 次成功、窗口内无跌倒，且冻结探针通过。其他 development 地形冻结复核单独报告，不当作独立测试集结论。

## 首轮 v6 仿真结果

| 样例 | 参数 | 严格结果 | 学习证据 |
| --- | --- | --- | --- |
| `projected_stall` | 旧 episode-0019、sim seed 26、std=0.01、双 LR=5e-5，最多 1200 步 | 未通过，x 最大 0.60574 m，预算结束 | 418 步受安全投影，1200/1200 请求均有效；5 次更新接受，末块 176 步保留 |
| `runtime_override` | 旧 episode-0020、sim seed 7、std=0.005、Actor LR=5e-5、Critic LR=3e-4，最多 2000 步 | 第 1908 步严格完成，x=4.50259 m，无跌倒或安全违规 | 实际参数记录正确，1908/1908 请求有效；8 次更新接受，末块 116 步保留 |

停滞样例的末块预测 value 均值仍约 4.19，而 GAE return 均值约 -17.59，解释方差约 -0.0015，说明修复筛选偏差后 Critic 仍未预测该失败状态。另一组的成功同时伴随探索幅度和 Critic LR 改变，不能归因于单一因素；下一轮将比较两种探索幅度、共同使用 3e-4 Critic LR，并继续使用严格冻结探针。
