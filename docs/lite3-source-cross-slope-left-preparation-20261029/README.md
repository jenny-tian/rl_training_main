# 5° 左向横坡 L1 单次基线准备

2026-10-29，用户在明确的横坡方案后表示“同意”。本阶段准备并执行一次冻结平地 actor 的左向横坡检查，完成或失败后停止，不再请求同范围授权。

固定 `slope_L1_cross_slope_left`，bank row0/column4，geometry/state seeds **10001/20080**，difficulty **0.30318197031305205**。以轨道前进方向为准，+Y 为左侧，主坡段左高右低、横向倾角 5°；机器人沿轨道纵向行走。12×4 m，入口平段 3 m，过渡 0.5 m，出口平段 2 m。命令 `(0.3,0,0)`，heading controller 关闭；前进 9 m 并连续稳定保持 50 步才算完成。

GPU0 允许已有计算进程，显存余量至少 12288 MiB；单环境、一个回合、最多 2000 控制步，启动／总时限 120/240 s。原策略权重、奖励、物理参数、安全阈值及 link/COM 诊断不变。重点报告横漂方向与幅度、偏航、实际速度和指令的差距、足部滑移；安全通过与控制精度分开判断。

237 项输入重新冻结，生产代码只切换准备文件路径。当前横坡清单拒绝上坡、下坡、右向横坡及未指定工况。**143 项针对性 CPU 检查通过**，无失败或跳过；与原 bank 几何等价记录逐值一致，相关 8 项几何源码哈希未改变，复用既有精确网格等价证据。该复核不新增 PhysX 运行证据。

见 [preparation.json](preparation.json)、[run_spec.json](run_spec.json)、[input_changes.json](input_changes.json)和[验证结果](post_review.json)。原 checkpoint SHA-256 为 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。待用目录 `/tmp/lite3_source_policy_cross_slope_left_link_reference_20261029` 在准备时不存在；复制最新基线账本，尚未增加横坡回合。

单次通过不代表整类成功率或新步态学习，失败也不自动修改基线策略。GPU 资源按整卡报告，不当作本任务独占数据。本次不授权重试、右向横坡或其他工况、PPO、G3/G4 或训练。
