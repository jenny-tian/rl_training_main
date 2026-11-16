# MuJoCo 中旧模型与 B 的冻结对比

2026-11-16。4 段接口检查和 36 个正式回合完成，0 次训练更新，未替换部署模型。平地、5 cm 上下阶均为两模型各 3/3 完成。粗糙地面的原始评分受到足部接触识别问题影响；按接触位置复核后，两模型各 3/3 满足完成条件。B 的中速右转依然偏慢，左弧线推进也偏慢。

[打开本地视频页面](index.html) · [关键截图](overview.png) · [完整结果及评分修正说明](RESULTS.md) · [部署接口审计](INTERFACE_AUDIT.md)

全部 18 个配对起点都可查看。每组左侧为 zero10100，右侧为 B10200。54 个视频（36 个单模型、18 个并排）和截图仅保留本地，未加入 Git；网页中的媒体链接需要在这台机器上打开。

优先查看起点 0：

- [5 cm 上阶](media/comparison_stairs_up_5cm_0/comparison.mp4)、[5 cm 下阶](media/comparison_stairs_down_5cm_0/comparison.mp4)
- [粗糙直行](media/comparison_rough_0/comparison.mp4)：视频保留原始终止标记，请结合报告中的接触识别复核解读。
- [中速右转](media/comparison_turn_right_0/comparison.mp4)、[左弧线](media/comparison_arc_left_0/comparison.mp4)

采用部署机器人资产，独立适配为训练策略周期 20 ms、髋/膝力矩上限 24/36 Nm。原部署入口为 12 ms、全部 30 Nm，本结果不代表原入口无需适配即可直接使用。每类只有 3 个固定起点，没有物理参数随机化；结果不与 Isaac Lab 合成成功率，也不作为实机可靠性结论。

[执行计划](execution_plan.json) · [逐回合指标](episodes.csv) · [统计汇总](summary.json) · [接触复核](contact_audit.json) · [最终核验](verification.json) · [下一阶段建议，待批准](NEXT_STAGE_PROPOSAL.md)
