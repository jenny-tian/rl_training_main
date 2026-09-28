# G0.3：LoRA/SAC 共用任务奖励与终止

日期：2026-09-28。依据 [v2.2](lite3_execution_plan_v2_2_2026-09-28.md)，接续 [runner 配对修复](lite3-g0-runner-parity-2026-09-28.md)。

## 已完成的原协议基线复核

五个调试地形的两个入口均完成，逐步观测、请求/执行动作、root/joint 状态及位置的最大差异均为 0。两入口各自使用同一个 simulation seed=7：

| 地形 | 两入口步数 | 冻结结果 | max_x |
|---|---:|---|---:|
| 101 | 2000 | 未完成，预算到期 | 0.652272 |
| 103 | 1721 | 成功 | 4.502398 |
| 107 | 1169 | 跌倒失败 | 1.625622 |
| 109 | 1628 | 成功 | 4.506969 |
| 113 | 1505 | 成功 | 4.506341 |

该批实际消耗 16,046 个控制 transition，连同前面的短诊断共 17,614 步。证据：`artifacts/g0_runner_parity_20260928/five_seed/{manifest,runner_parity_report}.json`。它解决了同种子跨入口配对问题，不替代多仿真种子重复实验或正式失败池筛选。

## 新的共用任务版本

显式选择 `--task-objective finite-horizon-v1`，旧命令保留 legacy 路径。共同定义保存在 [协议 JSON](lite3_task_objective_v1.json)，由 `task_objective.py` 生成哈希，两个 worker 的 config、summary、trajectory 与学习 checkpoint 均保存对应协议。

- 普通奖励：保留速度跟踪、前进、姿态、滑移、冲击、功率、残差/变化率、非足接触、停滞及横偏代价；横偏统一为机身和四足偏移的最大值。
- 第一版不启用依赖学习状态的过阶/抬脚额外奖励，不新增投影奖励项。
- 合格成功：显式终点奖励 **305**；失败或任务 deadline：**-8**。
- 305 保持原 LoRA 任务的目标值尺度，但固定为任务规定的终点值；不把完成后没有执行的物理步写入轨迹或宣称为实测 MC。
- 默认 deadline=2000，`--task-deadline-steps` 可为诊断声明较短任务；改变 deadline 会改变协议哈希。`max_steps` 不得超过任务期限。

| 结束原因 | learning terminated | learning truncated | bootstrap |
|---|---:|---:|---:|
| 成功、物理失败、安全失败、资格失败、task deadline | 1 | 0 | 0 |
| 提前用尽 collector budget、物理时间截断 | 0 | 1 | 1 |
| 普通 transition | 0 | 0 | 1 |

同一步成功与 deadline 重合时成功有效；物理/安全/资格失败优先于成功。物理 terminated/truncated 仍独立保存，不把逻辑任务终止伪装成发生过物理 reset。截断目标使用已捕获的 reset 前末帧。

SAC 把这些学习终止标志真正传给 `finish_step`/n-step/replay，而不只是添加日志。采集 cutoff 会 flush 未满 n 的短尾，保留其 bootstrap；任务 terminal 则把末尾前缀的 bootstrap 置零。LoRA 的 GAE 使用同一任务结束结果，末帧进入 rollout 后再结束本次试验。

轨迹新增 `ordinary_reward` 与 `terminal_reward_override`，两者相加重构实际 reward。完整逐项 reward 分解仍未接入，不能把这两个字段称为全部奖励分项。

## 旧数据与新 session 的边界

旧 reward/deadline 的 critic、replay 和 novelty baseline 不能混入新学习过程。LoRA 学习恢复检查 task objective；SAC 将 objective 放入已有训练契约，恢复时严格比较。纯冻结评估可保留旧权重，但必须记录本次评估的目标协议。

新 SAC session 允许从零均值 residual 和全新 critic/replay 开始，在目标域重新校准 novelty；不要求加载旧平地 novelty baseline。没有加载平地 replay 时不执行旧的平地 critic warmup。**这不等于 G1 的 4000 步目标域预热/critic-only 更新已经实现**：本轮只验证短诊断及持久化，actor/Q 更新被样本量门槛禁用。

## 验证与实验输出

输出根目录：`artifacts/g0_task_objective_20260928/`。全部提前登记于 [实验登记表](lite3_experiment_registry.md)。

- `cutoff/`：101/103 两入口各128步，deadline2000；逐步普通/终点/总奖励、learning endings、bootstrap 完全一致，末步bootstrap=1。
- `short_deadline/`：101两入口各64步，deadline64；逐步奖励与endings一致，末步reward=-8、bootstrap=0。
- `deadline/`、`success/`、`failure/`：完整任务终点复核，结果以各目录 `runner_parity_report.json` 为准。
- `replay_smoke/`：64步、n=3的新SAC session；核对真实checkpoint的短尾和终点观测，不将零更新的smoke报告为学习效果。

完整终点实测结果：

| 场景 | 两入口步数 | 末帧reward | 末帧bootstrap | 全程reward/endings配对 |
|---|---:|---:|---:|---|
| 101，任务deadline2000 | 2000 | -8 | 0 | 通过，reward max_abs=0 |
| 103，成功 | 1721 | 305 | 0 | 通过，reward max_abs=0 |
| 107，物理跌倒 | 1169 | -8 | 0 | 通过，reward max_abs=0 |

SAC新session的真实checkpoint有64条replay，末尾三个n-step奖励为 `[-4.970087, -6.547157, -8]`，bootstrap均为0，next_frame与轨迹末帧在存储精度下完全一致；actor/critic更新均为0。校验保存在 `replay_smoke/replay_validation.json`。初次启动在CLI检查阶段被旧baseline要求阻止，消耗0步；修正新协议校准入口后才完成这64步。

本组共消耗10,484步（配对640步、完整终点9780步、真实replay诊断64步），与前组累计28,098步，未超G0登记的60,000步上限。此前原G0-base-pair成本仍按原登记单列。

组件测试覆盖冲突终点优先级、collector cutoff、有限任务deadline、共同横偏奖励、协议变更拒绝resume、n-step短尾进入真实replay并保留末帧。最终完整测试：**1530 passed, 7 skipped**；命令为 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:source/rl_training python -m pytest tests/online_adaptation -q`（env_isaaclab）。所有本轮GPU诊断已结束，无训练后台任务。

## 后续门槛

G0 尚需统一 session 步数账本、候选探针成本、物理新回合与学习恢复的区别，以及封存守卫。输入清单已存在，但SAC历史/接触权限与LoRA单帧权限还需明确比较设计。只有这些工作和协议锁定完成后，才开始 G1 的目标域预热和 LoRA/SAC 小规模学习比较；本轮没有启动长训练。
