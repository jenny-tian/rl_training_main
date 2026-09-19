# Lite3 在线 PPO 实施状态

本次实现把计划中的低秩 PPO 路线拆成可以独立验收的组件，并保留现有残差 SAC 作为对照。初始策略仍来自平地 PPO；LoRA 的基础权重冻结，在线更新只作用于适配矩阵和新建 Critic。

已实现的接口位于 `rl_training.online_adaptation`：

- `LoRALinear`、`inject_lora`、`mark_only_lora_trainable`、`merge_lora` 和 `unmerge_lora`。适配层采用 `W = W0 + (alpha/rank)BA`，`B` 零初始化，注入后在任意有限观测上的确定性输出与源层一致。原始 `nn.Linear` 的 `weight`/`bias` 检查点键也可以严格加载到注入后的模型。
- `PPORollout` 和 `compute_gae`。每行同时保存 `terminated`、`truncated`、末帧价值、old log-prob 和策略版本；真实终止不 bootstrap，时间截断 bootstrap 但切断优势递推。
- `OnlinePPOUpdater`。实现 clipped PPO、GAE、熵项、可选平地行为锚定、梯度裁剪、有限值/参数范围检查和 KL 拒绝。拒绝会原子恢复 Actor、Critic 和两个优化器，接受后递增策略版本。
- `OnlinePPOSession`。只接受当前策略版本的连续 rollout，达到容量后执行一次更新并清空 on-policy 数据；发布器在接受后切换新版本。会话保留候选更新之前的 stable checkpoint；终止或严重非足接触时恢复 Actor、Critic 和优化器，清空旧 rollout，并用递增的新策略版本重新发布，避免 stale rollout 跨版本训练。结束有限长度实验时，`finalize_for_checkpoint()` 丢弃不足一个完整 rollout 的样本，并在存在未验证候选时恢复 stable policy；最终选择和丢弃数写入独立审计记录。
- `RslMLPPolicyAdapter`。把 RSL-RL `MLPModel` 的 TensorDict 观测接口转换为在线 PPO 使用的普通批量 Tensor，并保留 Gaussian distribution、log-prob 和 entropy 接口。

测试覆盖 LoRA 初始化等价、冻结基座、适配器梯度、合并/恢复、原始检查点加载、GAE 终止/超时边界、版本隔离、KL 回滚、会话容量、stable rollback、稳定策略收尾和断点恢复。当前回归结果为 `1117 passed, 7 skipped`。

新增 `scripts/reinforcement_learning/online_adaptation/train_lora_ppo.py` 作为独立仿真入口：它在 RSL-RL Actor 上注入 LoRA，使用 pre-reset 末帧处理终止/超时，经过 SafetyFilter 执行动作，并按完整 rollout 更新和发布策略。它还会保存训练配置、逐步记录和在线 PPO session checkpoint。

当前入口的保守默认值为 `action_std=0.05`、Actor/Critic 学习率 `5e-5`。`--nonfoot-contact-rollback-force` 默认值为 `20 N`；终止事件和超过该阈值的非足接触会立即回滚，并在 `updates.jsonl` 的独立事件行及 `latest.pt` 的 `rollbacks` 字段中记录原因、恢复版本和丢弃的样本数。

RockyRoad/ContinuousRough、HeldoutBoxes 和 FlatOnline 现在统一使用 `sustained_nonfoot_contact_5n_3steps_v1`：非足接触力超过 `5 N` 且连续保持 3 个仿真步才触发终止。LoRA worker 会把该协议标识同时写入 `config.json` 和 `latest.pt`，使结果可以核查。此前 RockyRoad 配置实际继承了 `1 N / 1 步` 默认值，因此旧的 2000 步 LoRA-PPO 结果不能与标记为 `5 N / 3 步` 的 SAC 结果作严格配对比较。

修正协议后的正式 LoRA-PPO 运行位于 `artifacts/semester_2026/09/paired_seed101_2000/lora_ppo_stable_finalize_v4`。控制轨迹共 2000 步，在第 1932 步因 `37.35 N` 严重接触立即回滚，第 1934 步连续第三帧非法接触触发环境终止。两个事件属于同一次事故；之后环境重置并继续到预算结束。运行执行 7 次完整 rollout 更新，结束时丢弃 66 条部分 rollout；最终 active/stable 均为 version 9，未保存新的未验证候选。

Residual SAC 的正式配对实验新增 `--update-schedule control-step`。该模式按已经提交的 transition 累积更新预算，每个控制步处理完命令后至多等待一个到期更新完成，策略发布时序不再取决于主机负载。seed 101 的两次 1460 步正式运行去掉 `control_compute_ms` 和 `loop_ms` 后逐行相同，Actor 事件流、summary、`residual_before.pt`、`residual_after.pt` 和 `experiment.pt` 也完全相同。两次都在第 1460 步因同一非法接触终止，Actor 更新发生在第 856、1056 步。

当前单 seed 证据中，冻结 PPO、LoRA-PPO 和同步 Residual SAC 的最大重置前前向位移分别为 `4.152 m`、`8.322 m`、`6.719 m`。LoRA 和 SAC 都比冻结 PPO 前进更远，但两者都发生真实非法接触且没有完成路段，因此尚不能声称已经在线学会陌生地形，也不能据此最终选择路线。统一口径结果见 `artifacts/semester_2026/09/paired_seed101_2000/comparison_contact3_sync_v4.md`。
