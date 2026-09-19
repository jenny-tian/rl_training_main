# Lite3 PPO 本体里程计上下文对照

日期：2026-09-19。代码与 CPU 回归已通过，真实仿真与稳定训练尚待验证；不能据此声称学会陌生地形。

## 输入与初始化

`--proprioceptive-context` 为可选开关。冻结源 Actor 和 teacher 仍接收原始 45 维观测；适配策略接收 base45 加三维积分 context，Critic 接收 policy45 + body velocity XY + elapsed fraction + context3，共 51 维。context 为 `sin(estimated_heading)`、`cos(estimated_heading)-1` 和估计横移除以 1.65 后裁剪至 `[-2, 2]`。

估计器对 body-frame 平面速度及 z 角速度做中点积分，在固定出生朝向下以零状态开始。转弯、倾斜和速度偏差会产生积分误差；仿真目前提供机身速度，实机需要相应的速度估计。这里未输入世界位置、世界 yaw 或地形扫描，也不将积分量称为真值。

context head 为无偏置的 45→rank→36 两层线性映射；三个 12 维动作分支分别乘 context 后求和，加到原始 LoRA 均值动作，再做既有动作限幅。第二层零初始化，因此初始均值、Gaussian likelihood 和 entropy 与原策略相同。基础权重保持冻结，训练参数是既有 LoRA 与新增两层权重。

## 迁移与回合边界

旧检查点只在 `pending_rollout=None` 时迁移。active/stable Actor 都保留原权重及 Adam moments，新 context 参数追加到优化器；旧 Critic 的新输入列及对应 moments 补零。保存过的非零 context head 在恢复和冻结 probe 时保留。显式 schema 检查防止相同维度但不同语义的检查点混用。

每个控制动作使用当前 context；下一帧 Critic 使用积分后的 context。环境终止时先读取 pre-reset 传感器、保存终止 context，再清零估计器。逐步记录同时保存当前、下一帧 context 和未裁剪位移快照。训练与冻结 probe 使用相同输入 schema。

固定源平地 anchor 补零 context，因此只约束零 context 下的行为。它不能约束已训练 head 在非零积分状态下的输出；必须通过独立、关闭学习的物理平地行走检验能力保持。

## 验证

在 `env_isaaclab` 环境运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:source/rl_training python -m pytest -q tests/online_adaptation
```

结果：`1420 passed, 7 skipped`。新增策略、里程计和 worker 迁移测试合计 87 项通过。覆盖初始化等价、梯度、active/stable 迁移、Adam 恢复、训练后 head 保留、解析圆弧积分对照、输入异常原子性和终止/重置对齐。`git diff --check` 与改动入口的字节码编译通过。

后续仿真采用已记录的固定起始检查点、相同种子和原严格完成标准；保留 `target_kl=0.01`、源 anchor `max_reference_mse=1e-4` 和既有安全界限。真实结果另行记录。
