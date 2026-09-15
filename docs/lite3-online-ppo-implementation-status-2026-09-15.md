# Lite3 在线 PPO 实施状态

本次实现把计划中的低秩 PPO 路线拆成可以独立验收的组件，并保留现有残差 SAC 作为对照。初始策略仍来自平地 PPO；LoRA 的基础权重冻结，在线更新只作用于适配矩阵和新建 Critic。

已实现的接口位于 `rl_training.online_adaptation`：

- `LoRALinear`、`inject_lora`、`mark_only_lora_trainable`、`merge_lora` 和 `unmerge_lora`。适配层采用 `W = W0 + (alpha/rank)BA`，`B` 零初始化，注入后在任意有限观测上的确定性输出与源层一致。原始 `nn.Linear` 的 `weight`/`bias` 检查点键也可以严格加载到注入后的模型。
- `PPORollout` 和 `compute_gae`。每行同时保存 `terminated`、`truncated`、末帧价值、old log-prob 和策略版本；真实终止不 bootstrap，时间截断 bootstrap 但切断优势递推。
- `OnlinePPOUpdater`。实现 clipped PPO、GAE、熵项、可选平地行为锚定、梯度裁剪、有限值/参数范围检查和 KL 拒绝。拒绝会原子恢复 Actor、Critic 和两个优化器，接受后递增策略版本。
- `OnlinePPOSession`。只接受当前策略版本的连续 rollout，达到容量后执行一次更新并清空 on-policy 数据；发布器在接受后切换新版本。未完成 rollout、优化器和计数器可断点保存。
- `RslMLPPolicyAdapter`。把 RSL-RL `MLPModel` 的 TensorDict 观测接口转换为在线 PPO 使用的普通批量 Tensor，并保留 Gaussian distribution、log-prob 和 entropy 接口。

测试覆盖 LoRA 初始化等价、冻结基座、适配器梯度、合并/恢复、原始检查点加载、GAE 终止/超时边界、版本隔离、KL 回滚、会话容量和断点恢复。当前回归结果为 `1102 passed, 7 skipped`。

仿真主循环尚未把现有 RockyRoad SAC worker 改成 PPO worker；下一步应在 Isaac Lab 中使用同一 45 维 PPO 观测、同一安全投影和同一地形/种子协议，接入 `RslMLPPolicyAdapter` 与 `OnlinePPOSession`，再与现有 SAC 进行配对预算比较。未完成该闭环前，不把组件通过声明为机器狗已经学会陌生地形。
