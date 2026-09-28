# G0 执行进展：冻结基线差异定位与轨迹记录

日期：2026-09-28。执行依据：[v2.2](lite3_execution_plan_v2_2_2026-09-28.md)。

后续更新：五地形完整配对及共同奖励/终止协议已完成验证，最终结果与剩余G0.4工作见 [G0.3报告](lite3-g0-task-objective-2026-09-28.md)。下文保留本阶段诊断过程和当时的待办。

## 已验证结论

同一 `seed=7` 曾不代表同一执行器状态。两个 runner 在最后一次物理 reset 前分别初始化 LoRA/PPO 或 SAC 网络，消耗不同数量的 Torch 随机数；`DelayedPDActuator.reset()` 随后使用该随机流采样 0–1 个物理步的延迟。仅比对 root pose、关节位置和配置无法发现它。

`actuator_audit/` 的 seed 101 实测：

| 执行入口 | Hip 延迟 | Knee 延迟 |
|---|---:|---:|
| 旧 LoRA reset | 1 | 0 |
| 旧 SAC reset | 1 | 1 |
| 统一 seed 后两入口 | 1 | 1 |

延迟单位是 actuator 的物理更新步，不是完整控制步。`runner_contract.seeded_initial_reset()` 现在在最后一次初始 reset 前调用环境的公共 `seed()` 接口，使该采样不依赖模型初始化顺序。没有把延迟范围改成常量，也没有调地形凑历史成败。

另一个动作契约差异来自安全投影：LoRA 的零增量路径对可行目标也做“归一化动作→物理目标→归一化动作”的往返，产生最大 `2.086162567138672e-7` 的偏移；SAC `force_zero_residual` 分支将它归零。已改为仅对实际越过关节边界的目标执行反变换，可行请求保持原浮点值，并保留真实的小幅边界修正。

单独修复舍入后，轨迹仍从第一物理步分歧，故不能将全部差异归因于浮点舍入。记录执行器延迟并统一 reset 后，101/109 两入口各 128 步的观测、root state、关节位置/速度、基础动作、请求动作、执行动作及位置均逐项完全一致（`max_abs=0`）。同一批 float32 观测在真实 RSL 模型、LoRA 零增量与两条安全路径上离线重放，CUDA 最大差异也为 0。

解析环境配置原先还相差五个指令字段：四个 special-mode 比例和 planar deadband。它们现由公共固定前向指令函数统一；短复测的完整环境配置 diff 为空。先前代码中看起来不同的 terrain-out-of-bounds 设置，在本次 RockyRoad 解析配置中实际相同，不作为根因。

## 代码与记录产物

- `runner_contract.py`：公共固定速度指令、必要时刷新观测、独立试验初始 reset、执行器延迟记录。
- `safety.py`：消除可行动作的无意义反变换，记录过滤器前一残差、暂停及计数状态；幅值、速率、姿态和关节上限保护仍执行。
- `StepTrajectory`：默认 float16，可显式选 float32；保存布尔字段原类型，异常退出保留部分 chunk 并标记 interrupted。
- SAC 地形主 worker 接入逐步记录：观测、当前帧/历史、基础原始输出、裁剪后基础动作、均值、请求/执行动作与残差、过滤前安全输入/记忆、奖励、位置、耗时、终止、bootstrap、reset 前后观测。均值来自实际采样调用，不为记录再次采样。
- LoRA 同步记录原始基础动作、物理状态和过滤器记忆。
- `run_runner_parity.py`：仅接受既有调试种子，记录预算、命令、代码差异、源文件哈希和轨迹自检。
- `analyze_runner_parity.py`：读取完整 float32 chunk，报告最早分歧、配置差异和同观测离线重放。

本次未接入独立源数据预训练入口 `train_residual_sac.py`；实际地形执行走已接入的 `play_online_adaptation.py`。潜变量 z 的逐步记录、所有 reward 分项及完整在线 session 恢复还没有完成。

## 实验账本与状态

根目录：`artifacts/g0_runner_parity_20260928/`，属于开发诊断，不进入正式适应成功率。

| 目录 | 内容 | 预算/实际步数 | 状态 |
|---|---|---:|---|
| `pre_fix/` | 首次沙箱尝试 | 0 | CUDA 不可访问，未生成环境轨迹 |
| `pre_fix_gpu/` | 101/109 × 两入口 | 512 / 512 | 完成，初始动作舍入及首步物理差异 |
| `safety_fix/` | 仅修复动作舍入 | 512 / 512 | 完成，同观测动作相同，物理仍分歧 |
| `actuator_audit/` | 101 × 两入口，记录实际延迟 | 32 / 32 | 完成，Knee 延迟不同 |
| `reset_fix/` | 统一 reset 和命令 | 512 / 512 | 完成，128 步逐步一致 |
| `five_seed/` | 101/103/107/109/113 × 两入口 | 最多 20,000 | 完整冻结复核已启动，结果以 manifest/summary 为准 |

短诊断累计 1,568 个环境 transition。完整复核即使耗尽预算，本组累计也不超过 21,568，仍在 v2.2 G0 的 60,000 步上限内。旧 G0-base-pair 的成本在原登记项单列，不能作为正式 session 的免费样本。

base SHA：`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。
短复测的 `runner_parity_report.json` 保存数值证据。五地形运行启动时的代码 commit 为 `7158527` 加工作区差异；`source.diff`、`runner_sha256` 与源文件哈希保留具体实现。完整复核开始后只调整了 supervisor 的结果读取/来源登记及分析工具，不改两个 worker 的行为。

运行例子（在项目根目录并激活 `env_isaaclab`）：

```bash
export PYTHONPATH=.:source/rl_training:/home/tjx/IsaacLab/source
python scripts/reinforcement_learning/online_adaptation/run_runner_parity.py \
  --output artifacts/new_parity_run \
  --base-checkpoint logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt \
  --steps 128 --seeds 101 109
```

## 尚未通过的 G0 门槛

动作/初始动力学配对已获得短程证据，**不等于整个 G0 已通过**，也不证明算法学到了新行为。

1. **奖励仍不同。**LoRA 横向惩罚取机身和四足偏移的最大值，SAC 只取机身；LoRA 成功采用 305 的显式吸收目标奖励，SAC 仍是裁剪后的普通完成奖励。完全相同的短轨迹也已有奖励差异。下一步需冻结共同奖励及单位，将显式目标值与观测 MC 区分。
2. **任务结束仍不同。**LoRA 已区分成功/失败/deadline 与 collector cutoff；SAC 当前 trace 记录实际旧语义，不能把记录功能误称为已经修复 finite-horizon bootstrap。必须给 SAC 成功、2000 步 deadline 和采集预算截断明确不同处理。
3. **输入权限未对齐。**base/LoRA actor 是 45 维单帧；SAC 是 49 维帧、25 帧历史和 base action，其中多了仿真接触量。仅写 proprio 不等于相同信息权限。见 [当前输入清单](lite3_g0_observation_contract_20260928.json)。
4. **预算与恢复尚未统一。**已有 checkpoint 的网络/replay/优化器/RNG保存不等于完整 session 的无损续跑；物理 reset 的 history/safety 重置与 session 成本继续累计需要分开定义。统一账本、候选探针成本及封存守卫未完成。
5. **基线重复性范围有限。**同一仿真种子和同一软件/GPU上的配对，不代替三个独立仿真种子或跨设备重复分布。真实训练中还需将环境随机流和探索随机流分离，或使用显式 reset 种子日程。

下一步仍按 v2.2 做 G0.3/G0.4，再进入 LoRA 与 SAC 各自的小规模学习信号诊断。高度扫描 teacher、正式封存测试和长训练均未启动。

## 验证

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:source/rl_training python -m pytest tests/online_adaptation -q`

结果：**1516 passed, 7 skipped**。新增覆盖 float32 微小动作差异、无重采样轨迹快照、零残差精确回归、真实小幅关节修正、独立初始 reset 和指令观测刷新。pytest 自动加载的外部 typeguard 插件与当前 typing_extensions 不兼容，故按项目测试环境禁用第三方插件自动加载；没有安装或修改依赖。
