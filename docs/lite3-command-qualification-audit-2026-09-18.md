# Lite3 第二轮失败审计：固定指令与资格失败边界

## 已完成的 v4 训练

运行：`artifacts/semester_2026/09/lora_ppo_multiepisode/seed101_boundary_v4_retrain_100ep_20260918_1419`。

100 回合已正常结束，严格成功 14/100，13 次物理跌倒、26 次安全违规；741 次更新中 737 次接受、4 次拒绝。20 次冻结探针通过 8 次，最终选中 episode-0085。开发集另外四种地形通过 3/4（103、107、113 通过，109 越界后严重接触）。尚未达到稳定训练标准；这些结果也不是独立测试集结论。

逐步检查发现两类可修复问题，以及尚未解决的真实障碍停滞。

## 1. 配置的前进指令被特殊采样覆盖

LoRA worker 只关闭了标准 `rel_standing_envs`，但任务使用 `UniformThresholdVelocityCommand`。其独立的 `rel_zero_vel_envs`、`rel_only_lin_y_envs`、`rel_only_ang_z_envs` 默认为各 0.07。在 x=0.3、y=0、yaw=0 的固定范围下，这些分支会生成零指令。再将重采样时间设为 1e9，零指令就持续整个回合。

旧 100 回合中，18 个回合几乎未离开起点，末段速度误差仅约 0.001–0.002：2、6、23、27、30、31、57、64、65、67、72、76、82、84、85、88、90、98。以其中 episode-0098 的仿真种子 104、episode-0097 checkpoint 做旧代码复现，直接观测到实际指令为零。

| 256 步对照 | 实际 x 指令 | 最后 base x | 相对初始 x=-3.5 的净前进 |
| --- | --- | --- | --- |
| 旧 v4 代码 | 0 | -3.48426 | 0.01574 m |
| 修复后 | 0.300000012 | -2.00010 | 1.49990 m |

相同地形 seed 101、仿真 seed 104、输入 checkpoint 和学习超参数。修改采样分支会影响随机数消耗，因此后续自然轨迹不保证逐帧相同。这个对照证明了固定指令修复，不能证明通过率提高。

修复局限于 LoRA 固定前进实验：关闭特殊采样、低速死区和指令渐变，重置后及执行动作前固定实际指令；如果修改了当前指令，则重算策略观测，避免继续使用包含旧指令的观测。物理步结束时再校验实际指令，异常时拒绝将该步送入 PPO。逐步数据记录实际指令和速度，summary 记录校验步数及指令范围。通用运动任务仍保留特殊采样。

## 2. 足端越界后的失败奖励过晚

旧轨迹中 27 个回合有挑战区越界，其首次越界步的奖励仍约为 +0.28 到 +0.78。严格资格检查使用机身和四足，奖励只看机身；资格虽然已永久失败，训练却继续到碰撞或步数耗尽。

例如 episode-0100 第 1841 步足端首次越界：机身 y=1.4926、x=4.1472，奖励 +0.35894；直到第 1938 步严重接触才收到 -8。延迟 97 步，还可能跨 PPO 更新块，使负反馈难以对应越界动作。

修复后奖励横向偏移取机身与四足最大值。严格资格首次出现 `lateral_violation` 或 `edge_slip` 就结束回合，失败当步奖励 -8。在 `--retain-terminal-learning` 模拟训练模式中，该步为无 bootstrap 的学习终止，立即结算末段 rollout。保守模式仍回滚失败候选。冻结评估同样在资格失效时停止。资格门槛与成功定义没有放宽；尚未积累抬脚或稳定步数不视作即时失败。

## 验证与证据限制

仿真产物：`artifacts/semester_2026/09/lora_ppo_multiepisode/audit_v5_20260918/`。

- `old_command` / `new_command`：上表的 256 步对照，退出码均为 0。
- `new_lane`：旧 episode-0099、仿真 seed 106；修复后自然轨迹在第 891 步严重接触，末段 123 步更新接受，奖励 -8，没有数据丢弃。该样例没有自然触发越界，不能当作越界触发证据。
- `new_completion`：旧 episode-0085、仿真 seed 7 的在线学习样例，第 706 步严重接触，末段 194 步更新接受；并未成功到达终点。
- `injected_lane`：专门验证失败通路的故障注入。在真实仿真第 600 步，把诊断快照中一只足的 y 读数设为 2 m，机身 y 仍为 0.074 m。worker 当步停止，`passed=false`、`lateral_violation_steps=1`、奖励 -8、`learning_terminated=true`；最后 88 步更新被接受，丢弃 0 步。此为人工注入的足端测量，不能作为自然越界或行走质量证据。最后一步因安全投影而不直接参与策略损失，终止奖励仍经 GAE 传播到同块更早的有效样本。
- `frozen_completion`：旧 episode-0085、仿真 seed 7，冻结评估在第 1975 步到达 x=4.50081，严格通过；零次更新、零次越界，1975 步实际指令全部校验通过。该样例验证正常完成通路仍可通过，并非新训练结果。

所有六个仿真进程正常退出。完整回归测试 **1154 passed, 7 skipped**；`git diff --check` 和启动脚本 shell 语法检查通过。测试命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests/online_adaptation
```

固定台阶的 `edge_slip` 识别仍不覆盖 RockyRoad 每块岩石边缘；`edge_slip_events=0` 不能证明每个凸起都没有滑落。真正的障碍停滞也仍存在，例如旧 episode-0037 前进到约 x=0.53 后停滞，速度误差约 0.3，不能归因于零指令。

## 后续训练

学习版本为 `fixed_command_qualification_boundary_v5`，奖励版本为 `progress_stall_completion_body_feet_failure_v3`。协议仍为 `in_lane_obstacle_interaction_edge_slip_v3`，SHA-256：`6243db772970452fd70811e7160739e277faa98b05f6c0398cad1e9eafd4f94d`。新的 manifest 阻止直接续写旧实验目录；成对实验默认输出目录升级为 v6。

下一步是独立目录内从冻结平地 PPO 初始化的新 LoRA/critic，先做 20 回合试训练，每 5 回合冻结探针，最后评估其他四个 development 地形种子。仍使用 0.3 m/s、rank 2、std 0.01、rollout 256、Actor/Critic 5e-5、每回合最多 2000 步。稳定标准保持最近 10 回合至少 80% 成功、至少连续 3 次成功、窗口内无跌倒且冻结探针通过。

试训练目录：`artifacts/semester_2026/09/lora_ppo_multiepisode/seed101_fixedcmd_v5_pilot20_20260918_2045`。使用 GPU 0、持久 tmux 会话和提交后的独立代码快照，启动脚本及提交号保存在运行目录。基础 checkpoint SHA-256：`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。

新版在首次越界停止，因此其跌倒数、接触数和回合长度不能与旧版直接等价比较；首先检查实际指令、失败边界和严格通过率，再决定是否扩大训练。
