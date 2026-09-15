# Lite3 在线 PPO 实施状态

本次实现把计划中的低秩 PPO 路线拆成可以独立验收的组件，并保留现有残差 SAC 作为对照。初始策略仍来自平地 PPO；LoRA 的基础权重冻结，在线更新只作用于适配矩阵和新建 Critic。

已实现的接口位于 `rl_training.online_adaptation`：

- `LoRALinear`、`inject_lora`、`mark_only_lora_trainable`、`merge_lora` 和 `unmerge_lora`。适配层采用 `W = W0 + (alpha/rank)BA`，`B` 零初始化，注入后在任意有限观测上的确定性输出与源层一致。原始 `nn.Linear` 的 `weight`/`bias` 检查点键也可以严格加载到注入后的模型。
- `PPORollout` 和 `compute_gae`。每行同时保存 `terminated`、`truncated`、末帧价值、old log-prob 和策略版本；真实终止不 bootstrap，时间截断 bootstrap 但切断优势递推。
- `OnlinePPOUpdater`。实现 clipped PPO、GAE、熵项、可选平地行为锚定、梯度裁剪、有限值/参数范围检查和 KL 拒绝。拒绝会原子恢复 Actor、Critic 和两个优化器，接受后递增策略版本。
- `OnlinePPOSession`。只接受当前策略版本的连续 rollout，达到容量后执行一次更新并清空 on-policy 数据；发布器在接受后切换新版本。未完成 rollout、优化器和计数器可断点保存。
- `RslMLPPolicyAdapter`。把 RSL-RL `MLPModel` 的 TensorDict 观测接口转换为在线 PPO 使用的普通批量 Tensor，并保留 Gaussian distribution、log-prob 和 entropy 接口。

测试覆盖 LoRA 初始化等价、冻结基座、适配器梯度、合并/恢复、原始检查点加载、GAE 终止/超时边界、版本隔离、KL 回滚、会话容量和断点恢复。当前回归结果为 `1102 passed, 7 skipped`。

新增 `scripts/reinforcement_learning/online_adaptation/train_lora_ppo.py` 作为独立仿真入口：它在 RSL-RL Actor 上注入 LoRA，使用 pre-reset 末帧处理终止/超时，经过 SafetyFilter 执行动作，并按完整 rollout 更新和发布策略。它还会保存训练配置、逐步记录和在线 PPO session checkpoint。

该入口尚未替代现有 RockyRoad SAC worker，也尚未在本轮启动 Isaac Lab 长程学习。因此当前结果证明的是算法组件、接口和闭环边界正确，不是机器狗已经学会陌生地形；下一步仍需在同一 45 维 PPO 观测、地形/种子和交互预算下运行 LoRA-PPO 与 SAC 配对实验。
