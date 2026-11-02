# Source 训练环境接线与 CPU 验证

状态：`cpu_integration_passed_simulation_pending_user_approval`，2026-11-02。

本阶段接续 `8c3d0ff` 的 actor-only 初始化和 transition 预算准备。
六类家族及全部 15 个 L1 单元保留。旧平地策略的失败仍是学习基线，
不增加“旧策略先通过所有地形”的准入要求。

用户在本阶段进行中明确要求：**“我同意后才可以进行下一步”。**
因此本阶段完成代码、CPU 验证和提交同步后暂停。下一阶段仿真尚未执行，
并行资格检查、PPO 数值试训和后续训练均不能自动启动。

## 本阶段结果

1. 新增独立训练配置 `source_training_env_cfg.py`。保留既有扫描配置，
   只允许 L1 和显式并行 sampler，三个 run 的 PPO / geometry / sampler
   分别为 `130001/30000/131001`、`130002/33000/131002`、
   `130003/36000/131003`。训练生成器不再消费诊断 seed `10001`。
   状态 ID `40000/43000/46000` 仍是记录标识，reset 未新增随机化，
   不把这些 ID 写成已经消费的随机流。命令固定 `(0.3,0,0)`，关闭独立的
   standing/lateral/yaw 随机命令份额与 critic 观测噪声。
2. 新增向量回合状态机。相对当前 terrain spawn origin 前进至少 9 m，
   并连续安全稳定 50 步才完成。非足接触或退回完成点之前会清零 hold。
   2000 控制步触发 truncation；安全、非法接触、姿态与 tile guard
   等失败仍是 termination，失败仅结束受影响的环境。
3. 新增 `SourceTrainingVecEnv`。PPO 原始 Gaussian request 及其 log-prob
   保留在 RSL storage；复制动作，经现有 Lite3 clipping / SafetyFilter
   后才执行。观测中的 last action 沿用 action manager 的执行动作，
   不用原始 request 代替。跟踪原始、裁剪、请求与执行动作。
4. 安全过滤器增加按 env ID 清理记忆的接口，保留原有全局清理接口。
   reset 只清理对应行的 pause、contact、stable、residual 和 hold。
   接触力先检查原始张量有限性，再转成布尔接触；避免 NaN 比较掩盖故障。
   非有限状态/观测/奖励会硬停。若动作前出现 safety pause，说明 reset
   或上一步安全终止时序有缺口，整次运行在执行该动作前停止；这是接线
   完整性错误，不把它继续当作普通失败回合。
5. 新增 `SourcePPO`，限定本机已审计的 RSL-RL 5.0.1。自动 reset 前捕获
   terminal observations、结束掩码、活动终止项、episode 步数和 hold。
   只有纯 timeout 加入 `gamma * V(actual_terminal_observation)`；禁用
   RSL 默认的 current-state timeout shortcut。done 仍切断跨 reset 的 GAE。
   termination 与 timeout 同时出现时不给 bootstrap。任何其他失败终止项
   同时触发时，都不把 completion 标成成功。
6. reset 分配与 transition 曝光分别记账。最后一步计入旧 terrain cell，
   新回合分配计入 reset 后 cell；不把 reset 配比冒充 transition 配比。

## 验证证据与限制

最终针对性 CPU 回归 **191 passed**，详见 `focused.junit.xml`。
覆盖新接线、原有 actor 初始化/预算、安全桥、并行分配、异步 reset、
tile bounds，以及原有 SafetyFilter 和 Lite3 action conventions。
最终代码语法检查和 `git diff --check` 通过，模型子模块保持干净。

新用例在真实安装的 RSL-RL 中证明：

- 投影后 storage 中的 action/log-prob 与原始 Gaussian request 一致；
- 人为区分 current value=7、terminal observation=2、post-reset observation=1000，
  以 `V(obs)=3*obs` 验证 timeout reward 为 `1+0.99*6`；
- 同时 termination/timeout 不 bootstrap，受影响行 reset，其他行计数保留；
- 完成 hold 连续性、非有限接触力拒绝、缺少终止捕获拒绝；
- 仍用诊断 seed 或未注册训练结束项的配置，在第一步前拒绝；
- 一次 **合成 CPU 数据** PPO 更新得到有限 loss，优化器产生新状态。

最初一轮有一项测试因 RSL storage 的 log-prob 带额外 singleton 维度而失败；
测试改为比较同形值后通过，未修改 RSL 或用宽松近似掩盖数值差异。

本阶段 **Isaac Sim 控制步为 0、Source 训练 transition 为 0**。
CPU 替身运行了实际 adapter/callback 代码，但没有验证 Isaac configclass
构造、PhysX、真实传感器刷新、USD 碰撞隔离、120 环境显存或吞吐。
回归中的单次合成更新不属于 Source 训练，也没有训练产物被选为候选策略。

尚未形成可执行的正式训练 lock：专用预算训练循环、逐更新有限梯度/权重检查、
六类验证门与 checkpoint 选择规则、G3/G4 和最终输入哈希锁定仍需后续阶段。
通用 `train.py` 不会自动获得这些约束，不应拿它直接启动 Source 训练。
中断后的 sampler/预算恢复仍未实现，禁止自动 resume。

## 下一阶段提案：仅 2 环境仿真接线检查，待同意

本阶段已经准备脚本 `scripts/tools/source_training_integration_smoke.py`，
仅语法检查，**未运行**。机器可读范围见 `next_stage.json`。

- 2 个环境、1 行 shared bank、全部 15 个 L1 cell 保留；运行 1 的训练 seed。
- 采用 pinned actor 均值加 std=1 的独立 Gaussian 噪声，检查 stochastic
  request → clipping → SafetyFilter → execution 接线；不创建 PPO/optimizer。
- 最多 24 次全局控制步，即 48 个诊断 transitions。最后一步只对 env 0
  人工注入 2000-step timeout，核验 pre-reset snapshot 与 env 1 计数不变。
  人工 timeout 不计作自然完整回合证据。
- 拟用 GPU0，共享使用前重新检查资源，至少 12288 MiB 可用；
  GPU 不满足条件时停止，不自动换卡或终止已有进程。
- 外部 wall timeout 240 s，TERM 后 15 s 清理宽限；保留日志、结果和
  运行前后资源快照；失败不重试、不放大规模、不转入训练。
- 此检查只说明短时接线，不判定地形成功率、不视为 G3/G4 通过，
  不授权 8/120 环境检查或 91-update 数值试训。

同意后的拟执行命令（本阶段未执行）：

```bash
PYTHONPATH=source/rl_training \
timeout --signal=TERM --kill-after=15s 240s \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/source_training_integration_smoke.py \
  --headless --device cuda:0 \
  --checkpoint /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt \
  --output /tmp/lite3_source_training_integration_smoke_20261102/result.json
```

该阶段结束后，无论成功或失败，都先归档并详细汇报，再等待下一次同意。

## 复现本阶段 CPU 检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
tests/source_scan/test_training_environment.py \
tests/source_scan/test_bridge.py tests/source_scan/test_source_training.py \
tests/source_scan/test_parallel_terrain_allocation.py \
tests/source_scan/test_source_parallel_reset_lifecycle.py \
tests/source_scan/test_tile_bounds.py tests/online_adaptation/test_safety.py \
tests/online_adaptation/test_lite3_joint_conventions.py
```

继承的 actor 审计和训练预算见
[2026-10-30 准备记录](../lite3-source-training-preparation-20261030/README.md)。
