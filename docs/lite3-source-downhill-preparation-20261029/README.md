# 5° 下坡 L1 单次基线准备

2026-10-29，用户在基线测试说明后同意继续；承接[已登记下坡提案](../lite3-source-uphill-link-reference-recheck-20261029/next_stage.json)，准备并执行一次冻结平地 actor 的下坡检查，无需重复确认同一范围。

固定 `slope_L1_downhill`，bank row0/column3，geometry/state seeds **10001/20070**，difficulty **0.3636354823034158**。12×4 m、5° 下坡、2 m 出口平段；命令 `(0.3,0,0)`，要求前进 9 m 加连续稳定保持 50 步。GPU0 允许已有计算进程，显存余量下限 12288 MiB；最多 2000 控制步、120/240 s 时限，结束或失败即停止，不重试、扫其他工况或训练。

237 项输入重新冻结，生产代码只变更运行器的准备文件路径。checkpoint、物理配置、命令、安全阈值、link/COM 诊断与资源准入逻辑和上一轮一致。下坡工况与原 bank 几何等价记录逐值一致，相关 8 项几何源文件哈希未变化，复用既有精确网格等价证据。此检查不等于新增 PhysX 验证。

针对性 CPU 测试 **143 项通过**，无失败或跳过；当前清单只接受下坡，拒绝上坡、横坡或未指定工况。见 [post_review.json](post_review.json)、[focused.log](focused.log)、[input_changes.json](input_changes.json)与[运行规格](run_spec.json)。其他代码沿用上一阶段 657 项相关回归证据，本阶段没有因单行清单绑定重复整套回归。

原平地策略 checkpoint SHA-256：`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。待用目录 `/tmp/lite3_source_policy_downhill_link_reference_20261029` 在准备时不存在；[基线账本](baseline_progress.json)复制上一阶段，下坡尚无回合，运行后另行记账。共享 GPU 的整卡占用和耗时不作为本任务独占资源数据。当前授权不覆盖 PPO、策略／奖励修改、G3/G4 或下一工况。
