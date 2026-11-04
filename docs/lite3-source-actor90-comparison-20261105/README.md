# 90 更新模型配对诊断：切换阶段停滞，已停止

本次已执行一次 GPU0 作业，验收失败，未获得有效的新旧模型对比。
原模型的 2000 个控制步已完成，进入新模型前持续 391.15 秒没有新控制步，
随后发送 SIGTERM 停止。未重试，compare 槽位已消费。

| 项目 | 实际结果 |
|---|---|
| 冻结源码 | `2617bdfe6675f809f3374d1d49b1f20864223359`，启动时已提交、推送并核对远程 |
| 运行环境 | `/home/tjx/anaconda3/envs/env_isaaclab/bin/python` |
| 计划条件 | 120 环境、15 个 L1 条件、一个共享 bank row、固定命令 `(0.3, 0, 0)` |
| 原模型 | 已记录 2000 个控制步，逐步有限性检查均为 true |
| 新模型 | 0 个已记录控制步，未达到第二次 initial-ready |
| 已记录诊断交互 | 240000 条，由原始控制步日志独立计数 |
| 新增训练 / PPO 更新 | 0 / 0 |
| 子进程执行时长 | 527.91 秒，包含等待切换的停滞时间 |
| 停止与清理 | SIGTERM 后进程组为空；未使用 SIGKILL，GPU0 回到 18 MiB、0% |
| 最终验收 | accepted=false；缺少 `result.cleanup.json`，总监督程序返回 1 |

## 已交付的证据

- [状态摘要](deliverables/status.json)、[控制步 CSV](deliverables/execution_steps.csv)。
- [真实控制步原件](raw/runtime/result.steps.jsonl)、[运行日志](raw/runtime/runtime.log)。
- [停止现场与原因](raw/runtime/agent_stop.json)、[执行状态](raw/runtime/execution.json)、[失败验收](raw/runtime/acceptance.json)。
- [全部原件与哈希索引](raw/runtime_artifacts.json)、[源码和运行清单](raw/runtime/run_manifest.json)。

这里的 CSV 仅整理已执行的原模型控制步，不包含模型效果配对数据。没有生成配对
CSV/PNG，不能推断新模型是否改善。两个 actor 的实际初态与几何配对检查也没有完成。

## 阻塞点与数据缺口

最后一条控制步为 old/2000，之后没有新 actor 的 initial-ready 或控制步。
具体阻塞点尚未确认：系统拒绝了只读调试器附加，见 [调试尝试](raw/debugger_attempt.log)。
疑点位于原 actor 结束后的结果整理、环境关闭和下一环境创建之间，不能仅凭无日志
把原因认定为某一个 IsaacLab 调用。

现有运行器仅在 initial-ready 和整个作业结束时保存汇总；首个 actor 的结果追加
到内存后立即关闭环境，没有先写盘。因此 `result.json` 仍为启动时快照，显示 0 步
且 actors 为空；它不是最终控制步计数。2000 步事实来自独立的逐步日志。首回合
前进距离、横漂、姿态、速度误差等指标未能保存，不能从 done 标记补造这些数据。

SIGTERM 后 Isaac Sim 子进程返回 0，但缺少关闭收据且工作没有完成，父监督程序
正确保留 accepted=false。`agent_stop.json` 记录外部停止原因；没有修改原始执行
记录，把外部停止伪装成达到 1800 秒总超时。

## 已完成的准备与原计划边界

按 [既有对比提案](../lite3-source-actor90-comparison-proposal-20261104.md) 及
[用户继续执行记录](raw/authorization.json)，修正了仅吞吐失败候选的合同入口，
核对 90 次更新、259200 条训练交互、有限损失、actor/critic 变化、候选模型哈希和
CPU 推理，并补充配对 CSV/PNG 导出。最终 [10 项 CPU 针对性检查](raw/focused_checks.log)
全部通过，耗时 1.71 秒；未运行整组回归或额外仿真测试。

训练台账前后字节一致，累计仍为 259584/262144 条；既有数值试训仍因
431.2859 条/秒低于 600 而失败，原失败收据和完整 G3 未通过的事实保持不变。
两份模型哈希未改变。已消费的其他作业槽位未改变，本次只增加 compare。

本阶段按异常停止规则结束。后续应先修复 actor 切换的场景生命周期，并在每个
actor 结束、关闭环境之前保存指标，同时分别监控关闭与下一场景启动的期限。
新的 GPU 对比属于后续阶段，本次没有执行。
