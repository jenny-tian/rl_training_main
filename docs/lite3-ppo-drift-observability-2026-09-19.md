# Lite3 PPO：横向漂移、观测信息与近期训练证据

日期：2026-09-19。本文记录三个 v9 source-anchor 分支在开始读取时的固定窗口，覆盖窗口中的全部回合，不按成功结果挑选回合或检查点。后续训练继续产生的数据不计入这些统计。

## 数据范围与计算口径

工作树：`/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation`。

下表中的运行目录均位于 `artifacts/semester_2026/09/lora_ppo_multiepisode/`：

| 简称 | 运行目录 | 完整训练回合窗口 | 最近已完成 probe |
|---|---|---|---|
| anchor1 | `seed101_source_v9_anchor1_100ep_20260919_0441` | `episodes/episode-0016` 至 `episode-0025` | `probes/episode-0025/seed-101` |
| anchor100 | `seed101_source_v9_anchor100_100ep_20260919_0441` | `episodes/episode-0019` 至 `episode-0028` | `probes/episode-0025/seed-101` |
| std100 | `seed101_source_v9_std100_anchor100_100ep_20260919_0532` | `episodes/episode-0003` 至 `episode-0012` | `probes/episode-0010/seed-101` |

每个回合读取 `summary.json`、`updates.jsonl`、`latest.pt`。逐步证据来自 `latest.pt["rows"]`；终止更新记录来自 `latest.pt["terminal_learning_updates"]`。检查点仅以 `map_location="cpu"` 读取，没有运行 GPU 仿真。

更新次数只统计 `updates.jsonl` 中带 `rejection_reason` 字段的 PPO 更新记录，排除 boundary/finalize 事件，避免重复计数。`approx_kl` 使用实现记录的近似量，不称为解析 KL。末步优势使用保存的 `terminal_raw_advantage`，即真正终止、无 bootstrap 时的 `reward - old_value`，不是归一化后的优势。

奖励分项和运动学分析取每个相关回合的最后 100 个非终止步，即 `rows[-101:-1]`。probe 属于冻结评估，`old_value` 和 `terminal_raw_advantage` 为 `null`，不为其补造优势值。

三个分支使用相同的固定源平地 anchor：

- 路径：`artifacts/semester_2026/09/lora_ppo_multiepisode/source_flat_anchor_v1/source_anchor.pt`。
- 观测形状：4096 × 45；SHA-256：`539ee7eba9301a2f0cbff6aaf109f72b37a7b6e298eefe04e299cdfc6f1de2cb`。
- 类型：`reconstructed_clean_flat_policy45_v1`；teacher 在同一组重建观测上重新计算均值动作，没有将历史保存动作直接当作 teacher 目标。

## 回合结果与实际更新

| 分支 | 成功 | 横向越界 | 跌倒 terminated | 严重非足接触 | 超时 | PPO 接受/尝试 | 接受率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor1 | 1 | 6 | 1 | 1 | 1 | 71 / 72 | 98.61% |
| anchor100 | 2 | 6 | 0 | 2 | 0 | 63 / 64 | 98.44% |
| std100 | 2 | 5 | 1 | 1 | 1 | 68 / 69 | 98.55% |
| 合计 | 5 | 17 | 2 | 4 | 2 | 202 / 205 | 98.54% |

每个分支都只有一次 `anchor_mse` 拒绝，没有 `target_kl` 拒绝。全部候选的最大 `approx_kl` 也低于 0.01，因此并非被 anchor 拒绝原因的优先级掩盖了 KL 超限。

| 分支 | approx_kl 均值 | approx_kl 最大值 | candidate_anchor_mse 均值 | candidate_anchor_mse 最大值 | 更新块 EV 中位数 |
|---|---:|---:|---:|---:|---:|
| anchor1 | 0.003139 | 0.009140 | 7.3125e-5 | 1.0016e-4 | −0.0198 |
| anchor100 | 0.003014 | 0.006974 | 7.6560e-5 | 1.0850e-4 | +0.0630 |
| std100 | 0.003932 | 0.008619 | 5.1691e-5 | 1.2599e-4 | −0.0714 |

这 30 个训练回合的 PPO 行有效率均为 100%，全部末步都有效，全部终止边界更新都被接受。训练中的安全投影比例约为 99.7%–100%，但当前 requested-action PPO 保留这些经过固定安全映射的请求动作，不再因为投影而丢弃它们。高投影率仍会改变实际探索形态；更新接受本身不等于运动表现改善。

`candidate_anchor_mse` 约束的是固定平地样本分布。逐步 `policy_reference_mse` 测量当前粗糙地形观测上的均值动作差，两者不是同一数据分布。例如 anchor1 episode-0025 末段当前地形 MSE 约为 3.05e-4，同时该回合 8 次更新全部接受，不能据此认定 1e-4 的平地 anchor 检查失效。

## 成功和失败的终止信用分配

五个成功终止均得到 reward 305，保存的原始末步优势全部为正：

| 样本路径（上述运行目录内） | old_value | terminal_raw_advantage | 终止更新接受 |
|---|---:|---:|---|
| anchor1 `episodes/episode-0022/latest.pt` | 93.6762 | +211.3238 | 是 |
| anchor100 `episodes/episode-0025/latest.pt` | 144.6420 | +160.3580 | 是 |
| anchor100 `episodes/episode-0026/latest.pt` | 147.1126 | +157.8874 | 是 |
| std100 `episodes/episode-0003/latest.pt` | 107.1397 | +197.8603 | 是 |
| std100 `episodes/episode-0012/latest.pt` | 134.2677 | +170.7323 | 是 |

其余 25 个失败终止均为 reward −8，末步原始优势全部为负，范围为 −125.3095 至 −32.1085。例子：

- anchor1 `episodes/episode-0025/latest.pt`：超时，old_value 46.8106，末步 A = −54.8106，边界更新接受。
- anchor100 `episodes/episode-0028/latest.pt`：横向越界，old_value 115.9516，末步 A = −123.9516，边界更新接受。
- std100 `episodes/episode-0011/latest.pt`：跌倒，old_value 76.8488，末步 A = −84.8488，边界更新接受。

因此，这个窗口不支持“成功终止仍被赋予负优势”或“失败数据没有进入学习”的解释。上述数值是每个末步的保存值，不以整块 value 均值代替末步 value。

## 最近冻结 probe

| 分支及 probe 路径 | 步数 | 结果 | max_x | 关键诊断 |
|---|---:|---|---:|---|
| anchor1 `probes/episode-0025/seed-101` | 2000 | task_timeout | 0.6360 m | 1039 个 stalled 步；末段 reward −1.4598，速度误差约 0.300 m/s，倾斜量约 0.223 |
| anchor100 `probes/episode-0025/seed-101` | 1846 | lateral_violation | 4.0925 m | 终止时 body_y = 1.5005 m，最大足端横向偏差 = 1.6710 m，超过原 1.65 m 边界 |
| std100 `probes/episode-0010/seed-101` | 1007 | terminated | 0.7458 m | 终止倾斜量 = 0.6584；末步 body 速度 = (−0.6892, +0.4961) m/s |

三个最近 probe 全部未通过。anchor100 episode-0025 的在线训练成功，并不代表保存后的冻结策略已经通过：对应 probe 的种子固定为 7，且使用均值策略、关闭在线更新，与训练轨迹条件不同。std100 的最近 probe 检查的是 episode-0010；不能将它的失败当作尚未被该 probe 检查的 episode-0012 的结论。

anchor1 probe 的 x 在 step 1280 已约为 0.62229 m，到 step 2000 仍约为 0.62228 m。末段世界 X 前进奖励接近零，stall 项为 −2，实际总奖励约为 −1.4598。这个停滞案例不是原地站立获得正奖励的套利。

## 横漂实例与奖励的坐标系差异

样本：`seed101_source_v9_anchor1_100ep_20260919_0441/episodes/episode-0017/latest.pt`。

对 steps 1719–1818 的 100 个非终止步，世界平均速度由 step 1718 到 1818 的位置差除以 2 秒计算：

| 量 | 数值 |
|---|---|
| 身体平面速度均值 | (0.28459, −0.00298) m/s |
| 世界位移对应的平均速度 | (0.25871, +0.12011) m/s |
| 两个速度方向之差的平面航向代理 | +0.44512 rad，约 +25.5° |
| step 1819 最大足端横向偏差 | 1.65102 m，越界 |

代理计算式为 `wrap(atan2(world_vy, world_vx) - atan2(body_vy, body_vx))`。它依赖平面运动近似和窗口平均，受转弯、倾斜、速度变化及测量时刻影响；它不是保存的陀螺积分结果，也不是原始 yaw 真值，不能当作训练时可直接获得的真值输入。17 个横向越界回合的相同末段代理落在约 +0.290 至 +0.788 rad，显示一致的方向偏移模式。

该例身体侧向速度几乎为零，但世界侧向速度已经达到 0.120 m/s。当前奖励在身体坐标系追踪 `(0.3, 0)` 速度，并奖励 body yaw-rate 接近 0；完成任务却要求沿世界 X 前进且保持在走廊内。停止继续转动不等于已经朝向赛道方向。

以 anchor100 的六个越界回合各末 100 个非终止步汇总，可从保存字段准确重算的项为：

| 奖励项 | 均值 |
|---|---:|
| 身体速度跟踪 `2 * exp(-4 * velocity_error²)` | +1.921986 |
| 世界 X 前进 `0.5 * forward_progress` | +0.203895 |
| 横向偏差 `-(lateral_deviation / 1.65)²` | −0.801277 |
| 倾斜 `-2 * tilt_rms²` | −0.011769 |
| 足端滑移 `-0.05 * foot_slip` | −0.001078 |
| stall、非足接触 | 0 |
| 未单独保存项及如有的裁剪贡献合计 | −0.756920 |
| 实际总 reward | +0.554838 |

实际 `base_ang_vel_z`、foot impact、joint power 等没有保存在这些 rows 中，所以不能进一步声称重算了全部奖励分项。余项包括 yaw-rate、垂向速度、冲击、功率及残差成本。

奖励公式还说明一个局部目标差异：若机身朝向偏移 0.3 rad，但身体速度仍为 `(0.3, 0)`、yaw-rate 为 0，则身体速度和 yaw-rate 项仍可取最大值；世界侧向速度约为 0.08866 m/s，世界 X 奖励相对朝向正确时只下降约 0.01340。横向偏差惩罚和最终失败会反对继续侧漂，但纠偏需要识别累计方向和位置，不能仅由当前 yaw-rate 是否为零决定。这属于奖励项之间的局部取舍，不是已经证实的奖励符号实现错误。

## 观测缺口与下一轮 context 对照

v9 actor 使用原始 45 维瞬时本体观测：角速度、投影重力、命令、关节位置/速度及上一动作。它没有直接得到累计 heading 或相对横移。上一动作和闭环身体状态可能携带间接历史信息，但不能将其等同于已提供了可靠的里程计状态。v9 critic 的 48 维输入额外包含身体平面速度和时间比例，仍没有这两个累计量。

在重力方向、关节状态、角速度及上一动作相同的情况下，世界 yaw 或整体横向位置可以不同，所需纠偏动作也不同。上述身体侧向速度接近零却持续世界横移的实例，与这个观测缺口一致。该证据足以支持一次加入累计 context 的对照，但不证明全部 17 次越界只有这一原因。

拟增加的 `sin(heading)`、`cos(heading)-1` 和相对横移估计可以补充累计方向与位置的信息。由 body 速度和角速度积分得到的是平面近似里程计；倾斜时 body z 角速度并非精确 yaw-rate，速度偏差和积分误差会累积。需要保留估计快照进行诊断，并在真实重置后清零；不把估计值称为世界真值。

context 不保证解决全部停滞或跌倒。anchor1 probe 在 x≈0.62 m 的长期停滞、std100 probe 在 x≈0.75 m 的失稳，仍涉及接触与步态通过能力。近期 EV 中位数接近零也说明价值学习仍有局限，但不能仅凭该统计把全部问题归因于观测维度。

下一轮可延续 `action_std=0.03`、`actor_lr=1.5e-4`、`critic_lr=3e-4`、`reference_anchor_coef=100`，把 context 作为主要改变：

- 当前接受率已为 98.54%，窗口内没有 KL 拒绝，没有证据要求再提高学习率或放宽更新约束。
- std=0.1 同时使用 actor_lr=5e-4，近期成功数仍为 2/10，最近冻结 probe 跌倒；样本窗口和种子不同，不能宣称这构成受控优劣结论，也不足以支持继续加大探索。
- 保持原 `target_kl=0.01`、源平地 `max_reference_mse=1e-4`、原严格完成与安全门槛、固定 probe 流程。用预先确定的相同起始检查点和种子比较 context 的影响，不以事后挑出的成功轨迹作为训练成功证明。

本次审计没有修改奖励、阈值或训练进程。
