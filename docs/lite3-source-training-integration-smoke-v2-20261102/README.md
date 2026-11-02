# 2 环境训练接线复验：限定范围内通过

状态：`passed_within_two_environment_24_step_scope`，2026-11-02。
用户同意后，在新目录完成唯一一次运行。父进程验收与独立离线复核均通过，
最终报告、逐步原始证据和清理记录完整。没有重试、扩大规模或 Source PPO 更新。

## 本次运行和验证结果

运行提交 `1c0d1a6dfab96529ba33262cca24e5c3eb323b6d`，启动前工作树干净、
远端 SHA 一致；manifest 锁定 208 个生产 Python 文件，checkpoint SHA 为
`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。
使用运行 1 的 geometry 30000、sampler 131001、PPO 根 seed 130001、
噪声 seed 132001、std=1；reset 状态标识 40000 仍不代表随机初态已启用。

| 检查 | 实际结果 |
| --- | --- |
| 场景 | 2 环境、1 行 shared bank、15 个登记 L1 cell，设备 cuda:0 |
| 控制交互 | 每环境 24 步，每步 0.02 s，即各 0.48 s；共 48 诊断 transitions |
| 原始记录 | 24 条连续、同 attempt ID 的 JSONL，完整保留四层动作和结束步观测 |
| 父进程 | accepted=true、退出码 0、reasons 为空、无资源监控错误 |
| 最终结果 | status=passed、steps_written=24、SHA 与 journal 一致 |
| 原始动作裁剪 | 13/48 个环境步（27.0833%）发生 canonical clipping |
| 后续过滤器投影 / pause | 0 / 0 个环境步 |
| 最终 done / timeout | `[1,0]` / `[true,false]` |
| 最终 episode counters | `[0,24]`，仅 env 0 reset |
| 终止前观测 | policy 45 维、critic 393 维；所有保存值有限 |
| Last-action 语义 | terminal policy 最后 12 维与最后执行动作最大误差 0 |
| PPO / Source 训练 | 0 更新、无 optimizer/训练器构造 |

第 24 步只对 env 0 人工设置 `episode_length_buf=1999`，步进后捕获 terminal
steps=`[2000]`。唯一活动项是 `source_training_timeout`；terminated=`[false,false]`，
truncated=`[true,false]`，completed=`[false]`、hold=`[0]`。
本轮没有触发安全、非法接触、姿态或 tile-bound 终止。
人工 timeout 是接线探针，不是自然完成 2000 步或前进 9 m 的证据。

## 地形台账的独立重建

初始分配列为 `[3,1]`，对应 slope_downhill 和 multi_scale_rough；最后一步仍在
旧列 `[3,1]` 上执行，env 0 reset 后列变为 `[1,1]`。

| Cell label | Episode assignments | Transitions |
| --- | ---: | ---: |
| slope_downhill | 1 | 24 |
| multi_scale_rough | 2 | 24 |
| 其他 13 个 cell | 0 | 0 |

独立按每步 before/after 列和 done 重算，与最终 exposure 完全一致。
标签表示 episode 分配；0.48 s 短探针不证明机器人接触了这些 cell 的完整目标区域，
也不提供六类技能、成功率或摩擦覆盖结论。其余 cell 保留在配置中，没有删除家族。

## 持久化、资源和清理

本次 `result.json` 在环境/应用关闭之前完整写入，应用关闭后仍可读取；
`result.steps.jsonl` 与 `result.cleanup.json` 均保留。
环境 close 已返回；应用 close 状态为 `closing`，Kit 关闭后进程退出。
不把这一状态写成 Python 层应用 close 已返回。父进程确认进程组为空，
独立后续进程组检查也无残留，persistence_errors 为空。

启动前 GPU0 可用 24107 MiB，超过 12288 MiB 门槛；5 秒间隔采样中卡级
总占用峰值 2848 MiB、最小可用 21277 MiB。运行前后 GPU0 均为 18 MiB 占用，
未终止或修改其他用户任务。这是稀疏卡级采样，不是本进程精确峰值或训练吞吐。

监督器计时 20.257029 s，关闭前报告计时 17.172716 s，未触发 240 s 超时。
日志中的地形生成 0.320697 s、场景创建 1.166834 s、simulation start 1.174421 s
只是本次启动时间。持久化 I/O 和启动时间都不能用于 G4 稳态吞吐判定。
Kit 的 IOMMU/P2P 与足部 visual USD 引用警告保留在原始日志中；
本轮未取得完整物理隔离或视觉资产资格。

## 归档和当前边界

`raw/` 为本次 12 个运行文件及 Kit/Isaac Lab 完整日志的原样副本。
`review.py` 只读复核本次数据；`review.json` 汇总验收、动作、终止和计数结果。
`provenance.json` 记录运行输入与副本哈希，`SHA256SUMS` 校验本目录文件。
历史首轮的 `incomplete_runtime_artifacts` 原样保留，本轮不追改历史结论。

本次仅证明修复后的 2 环境短时训练动作/结束/写盘链路工作，
并读回 live critic 宽度 393。没有执行 PPO，所以没有验证 live critic 参与更新、
真实 timeout bootstrap 或长回合学习；尚未授予 G3/G4、8/120 环境或训练资格。
CPU 阶段的 220 项测试仍绑定其修复提交，本阶段不重复运行无改动的回归。

## 下一阶段提案：8 环境运动/reset 检查准备，仅 CPU，待同意

下一步建议实现并 CPU 验证 8 环境训练接线与运动隔离诊断，复用既有 shared-bank
G2 接触/扫描工具，补齐训练 wrapper、物理子步接触、扫描归属、全刚体边界、
异步结束/reset 与恢复首步的证据链。旧策略普通跌倒/边界失败作为各环境失败回合，
不要求旧平地 actor 先通过所有地形，也不删除六类家族；无效状态、跨环境接触、
错误扫描归属和非 reset 环境被修改属于完整性故障，必须停止。

本提案仅包含：检查既有 8 环境 G2 证据可复用范围，编写独立 8 环境诊断与验收器，
CPU 检查异步 reset/时序/落盘，冻结具体运行预算、seed、显存和超时规格。
保持当前 2 环境验收器限定范围，不直接放宽为 8 环境通过。
准备完成后提交、汇报，再单独申请执行一次 8 环境仿真。
**本次下一阶段同意若获得，也不包含 GPU/Isaac/PPO 执行。**

`next_stage.json` 为未批准提案。当前阶段已完成，停在用户确认点。
