# 旧模型与 B 的多地形视频对比

2026-11-16。已完成 15 个场景、30 个单模型视频及 15 个并排对比。左侧为 zero10100，右侧为 B10200。所有回放的初态和原评测轨迹均精确一致，模型冻结，无训练、无部署、无模型替换。

[打开完整视频页面](index.html) · [关键画面总览](overview.png) · [详细阶段报告](RESULTS.md)

视频和截图只保留在本地，页面中的媒体链接需在这台机器上打开。Git 继续保存代码、文字报告、指标和验证记录。按用户要求，媒体已从当前版本的 Git 跟踪中移除并加入忽略规则；此前提交 51c73ee 曾上传媒体，历史提交未改写。

优先查看：

| 场景 | 对比视频 | 观察重点 |
|---|---|---|
| 随机化粗糙地面 | [播放](comparisons/rough_randomized/comparison.mp4) | B 在机身直立时横向偏离；旧模型完成 |
| 粗糙下坡 | [播放](comparisons/slope_rough_down/comparison.mp4) | 旧模型完成，B 推进不足超时 |
| 随机化起步 | [0.4 倍慢放](comparisons/startup_randomized/comparison.mp4) | B 在 0.54 s 失稳；旧模型稍后横向越界 |
| 5 cm 上阶 | [播放](comparisons/stairs_up_5cm/comparison.mp4) | 两者完成，选定起点 B 用时更短 |
| 5 cm 下阶 | [播放](comparisons/stairs_down_5cm/comparison.mp4) | 两者完成 |
| 右侧 90° 航向目标 | [播放](comparisons/heading_right/comparison.mp4) | B 最终残差约 36.5°，旧模型约 21.4° |
| 边走边左转 | [播放](comparisons/arc_left/comparison.mp4) | B 转向时前进较慢 |
| 中速右转 | [播放](comparisons/turn_right/comparison.mp4) | 此起点早期均速接近；总体结论仍看完整样本 |

其他平地、粗糙地面、上下坡和摩擦分区视频见完整页面。画面来自 Isaac Lab 中记录状态的渲染；渲染本身不推进物理。终止后保持最后姿态，已标注 ENDED，不代表结束后还在稳定站立。视频只展示固定起点，不能重新计算或替代上一阶段的总体成功率。

- [计划](PLAN.md)、[原执行计划](execution_plan.json)、[修正记录](execution_plan_v2.json)
- [分析汇总](summary.json)、[逐片段指标](selected_metrics.csv)、[复现证据](evidence/)、[完成记录](completion.json)
- [最终核验](verification.json)、[全部场景截图](all_scenes_contact_sheet.png)、[交付文件哈希](artifact_manifest.json)
- [下一阶段建议，待批准](NEXT_STAGE_PROPOSAL.md)
