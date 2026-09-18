# Lite3 成功终止回报与 Critic 上下文修复

v6 修复了安全投影后的 PPO 样本筛选和续训参数覆盖，两个训练分支仍未达到稳定成功。前 5 回合，std=0.005/0.01 分别严格通过 1/5、2/5，冻结探针均失败；后续继续保留原始失败和更新记录。本轮修改训练目标与值函数输入，完成资格、安全阈值、原始平地 PPO Actor 的 45 维输入保持不变。

## 成功终止目标

原目标在严格成功时立即终止，只获得 5 分；普通行走每步仍有正奖励。gamma=0.99 时，如果延迟一步仍能获得大于 0.05 的普通奖励，则 `r + 0.99*5 > 5`，目标函数反而偏好延迟成功。v6 std010 episode-0003 成功前 100 步实际平均奖励为 1.3884。这个条件比较说明目标冲突，不证明策略已经主动利用它。

当前启用的非终止正奖励上界为 `velocity_weight + yaw_weight + forward_progress_weight = 3`，其他启用项均为非正代价。新成功终止奖励包含吸收成功状态的尾值：`completion_bonus + 3/(1-gamma)`，默认约 305。因此在普通奖励上界内继续等待，不能超过立即成功。只在已有严格成功条件满足时赋值，绕过公共奖励函数的 5 分裁剪；终止保持为 true，bootstrap 为零，不读取 reset 后状态，也不把预测 Critic 值当作成功尾值。

失败仍为 -8。supervisor 明确声明 2000 步任务截止，截止时未成功属于 terminal failure；更早的诊断采样切片仍属于 truncation。截止当步已经严格成功则优先计为成功。是否保留失败更新只决定学习或回退，不改变真实终止类型。

## Critic 输入与旧状态迁移

Actor、Gaussian likelihood、KL 和 reference anchor 仍使用原 45 维输入。Critic 独立追加机体系线速度 XY 和已用时间比例，共 48 维；没有增加世界坐标、地形高度、地形类别或全局 heading。仿真中的线速度来自与在线奖励相同的机器人状态字段；未来真机对应状态估计器，本轮没有验证其真机估计误差。

时间分母优先使用声明的任务截止；例如 max_steps=1200、deadline=2000 时，截断 bootstrap 的时间特征为 0.6，不能误写为 1。位置、航向及停滞历史仍可能产生观测混叠，本轮三项上下文没有解决全部值函数信息缺失。

旧 Critic 首层的 45 列完整保留，新增 3 列初始化为零；active/stable 两分支及 Adam 对应动量一同扩展，其他参数、优化器步数、计数和候选验证状态保持。迁移前后任意新上下文的初始 V 数值相等，新列随后可学习。只接受 pending_rollout=None 的边界；已是 48 维时要求 Critic schema 一致。pending rollout 的两套观测在组合和保存恢复中分别保留，旧 checkpoint 缺少 critic_observations 时仍按旧路径加载。

逐步记录新增真实采样期 old_value、终止原始优势和策略均值漂移。此前只保存块均值，不能据此断言末步优势为负。单独携带的 checkpoint 现在也保存 gamma、任务截止、时间分母和成功终止奖励。

## 验证记录

最终完整 CPU 回归为 **1277 passed、7 skipped**；时间分母和保守模式终止分类也已包含在最终复测中。回归涵盖吸收成功回报的延迟比较、Actor/Critic 分离输入、真实 pending 序列化恢复、旧状态和 Adam 迁移，以及任务截止无 bootstrap。

仿真目录：`artifacts/semester_2026/09/lora_ppo_multiepisode/audit_absorbing_v7_20260919_0031/`，源码快照及输入参数有独立哈希记录。

- `deadline128`：专门把截止设为 128 步的边界回归；当步 stop_reason=task_timeout、奖励 -8、terminal，无 bootstrap，128 条请求完成一次接受更新。这不是自然训练成功案例。
- `completion_seed7`：从 v5 已验证 episode-0020 恢复，std=0.005、Actor LR=5e-5、Critic LR=3e-4；第 1724 步严重非足接触失败，最大 x=3.44843，1724 条请求全部有效，7 次更新接受。名称不代表通过，此运行没有验证自然成功边界。
- `frozen_completion`：原已验证策略在第 1884 步到达 x=4.50388，严格通过，无跌倒或安全违规；零次更新，成功终止奖励约 305。它验证自然完成通路仍可通过，不是新训练能力结论。

以上仿真快照先于时间分母和保守模式日志修正，但其 max_steps 等于声明截止、且启用 retain-terminal-learning，这两项修正不改变对应运行路径。

学习合同：`requested_ppo_absorbing_success_critic_context_v7`；奖励合同：`absorbing_success_explicit_task_deadline_v4`；Critic schema：`policy_plus_body_velocity_xy_elapsed_v1`。仍使用原严格资格协议 `in_lane_obstacle_interaction_edge_slip_v3`，不将奖励数值变化当作资格变化或已学会的证据。
