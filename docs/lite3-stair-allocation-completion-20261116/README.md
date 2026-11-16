# 整体能力补测完成

2026-11-16。已完成用户批准的 5600 个新增冻结评测回合，包括其他地形的 960 回合。无新增训练、无模型替换、无部署。

B 保持了较高的带反馈上阶通过率，并改善了随机化、无反馈上下阶及部分非台阶场景。它仍有粗糙地面横向漂移、名义粗糙下坡推进不足、中速右转跟踪变差及少量起步姿态失败；不能概括为所有能力都提高。A 的名义上阶明显较差，不作为首选候选。建议保留 B 供下一阶段核对，旧模型继续作为基线，不为追求上阶 100% 自动追加训练。

- [完整阶段报告](RESULTS.md)：八个非台阶子型、上下阶、转向、起步、预算和结论边界。
- [其他地形对比图](results/retention_comparison.png)、[台阶对比图](results/stairs_overview.png)、[转向对比图](results/turn_comparison.png)：正式评测数据图，不是仿真截图。
- [批准范围](PLAN.md)、[实际执行计划](execution_plan.json)、[完成记录](completion.json)、[分析汇总](results/summary.json)、[验证记录](verification.json)。
- [逐地形完成统计](results/retention_terrain_summary.csv)、[逐行指标](results/retention_groups.csv)、[每回合记录](results/retention_episodes.csv)。
- [下一阶段建议，待用户批准](NEXT_STAGE_PROPOSAL.md)。

本阶段单独保存，上一阶段的停止记录和历史证据保持原样。
