# 已批准的有限证据训练交付阶段

本阶段已停止后续 GPU 作业，实际交付首个 90 更新的 actor 和训练曲线。CPU 有限
范围复核与 G4 通过；数值训练全部完成，最终验收因吞吐未达标而拒绝，对比没有启动。

| 项目 | 实际结果 |
|---|---|
| 冻结源码 | `67d92d7443491c8a2b05e453f7a7ad34eeb58759` |
| CPU 有限证据 | 1884 控制步、7536 物理子步、77 批 reset、15 个条件和全部 120 环境地面正对照通过，1184.43 秒 |
| G4 | 288 控制步、34560 条只读交互；897.0691 条/秒，reset p95=0.648526 秒；6 批计划 reset 覆盖全部 120 环境，测量期另有 3 批自然 reset；验收通过 |
| 数值训练 | 90 次真实 PPO 更新、259200 条交互、1800 次反向/Adam 步；全部损失有限，actor/critic 改变，模型可加载 |
| 训练验收 | accepted=false，唯一理由为端到端吞吐 431.2859 条/秒低于 600；测量时间 600.9934 秒，GPU 子进程约 624.12 秒，正常清理 |
| 配对对比 | 未启动，compare 槽位未消费 |
| 计费 | 新预留和实际训练均为 259200 条；保留历史 384 条，累计 259584/262144；本轮预算收据状态 failed，不退款、不重试 |

可直接使用的交付文件：

- [actor_update_90.pt](deliverables/actor_update_90.pt)，760457 bytes，SHA256 为
  `14d11c8b57392c3ac0ac1e08b26de1cc1254ee059d1f653172433df9a05bdf42`。
- [训练曲线 PNG](deliverables/training_curves.png) 和 [CSV](deliverables/training_curves.csv)。
- [交付状态](deliverables/delivery_summary.json)、[数值验收](raw/runtime/numerics/acceptance.json)、
  [完整训练报告](raw/runtime/numerics/training/run_report.json) 和
  [逐次更新收据](raw/runtime/numerics/training/update_receipts.jsonl)。

actor 为推理权重，不含可恢复的 critic/optimizer；独立 CPU 加载和零输入推理
通过。训练曲线来自 90 次实际更新，覆盖全部 15 个条件；训练 reward 的变化不能
替代同条件新旧模型对比，当前没有效果提升结论。只读对比待批准方案见
[下一阶段提案](../lite3-source-actor90-comparison-proposal-20261104.md)。

全部本阶段原件及哈希索引见 `raw/runtime_artifacts.json`。`python review.py` 核对
归档、收据、模型、曲线、预算和未启动对比的事实，不重跑物理仿真或授予吞吐资格。
原 G3 的超时/未完成及有限范围的区别均保持，没有修改任何失败原件。

训练第一次调用停在 GitHub 源码核验的 TLS 错误，尚未创建清单、占用槽位、预留
预算或启动子进程。确认这些事实后使用单条命令的直连配置，真实核对远程分支成功，
才执行唯一一次 GPU 训练。问题记录与连接恢复记录一并归档；没有重复 GPU 训练。

以下为本阶段的批准范围与实现说明。

用户已回复同意执行 [修订方案](../lite3-source-training-delivery-revision-proposal-20261104.md)。
批准绑定该方案在 `5107448972f87a37d85f276a9b0d1078c55f3376` 的内容，原整包批准、
已消费槽位和两次失败 G3 的记录保持原样。补充批准原件见 `raw/revision_approval.json`。

本阶段先进行一次 CPU 独立复核，覆盖已保存的 1884 个完整控制步及全部对应物理
子步，上限 1800 秒；原始事件、动作、扫描、接触和 reset 均须重建核对，不能跳过
错误帧。完整控制区间之外的中断尾部单独计数，不获得资格。成功时仅生成
`lite3_limited_real_motion_evidence_v1` 收据，accepted/G3_qualified/training_ready
仍为 false，scope_qualified 才表示上述限定范围独立复核完成。

资格衔接显式引用补充批准和有限范围收据。G4、数值试训和配对对比继续使用原未
消费槽位，各只运行一次。G4 在第 88、128、168、208、248、288 步之前分别对
20 个不重复环境执行原有 reset，刷新场景和观测后执行该控制步，所以每批均在
测量期经历物理恢复。计划 reset 与自然 reset 分别保存样本；所有相关耗时计入
吞吐，reset p95 按实际同步测得的 _reset_idx 耗时统计，另报刷新就绪耗时。

数值训练仍要求实际 90 更新、259200 条新交互、1800 次反向/Adam 步、有限损失、
actor/critic 改变、可加载 actor 和原吞吐上限。训练预算只在真实训练入口一次性
预留，失败或部分执行不退还。正式 G3 未通过、正式选模未授权的事实会在数值
收据中保留。配对对比只产生固定条件诊断，不宣称独立地形成功率。

修改仅涉及资格、复核和测量：现有物理、资产、环境、奖励、动作、原生接触
解码器及扫描保护输入逐项保持不变；额外 AST 检查锁定 G4 场景配置/扫描检查和
数值监督执行主体。CPU 合成验证不能取得真实收据。所有运行期间保持已提交、
已推送源码冻结。实际运行原件保存在原阶段持久目录的 `run/prefix_review`、
`run/g4`、`run/numerics`、`run/compare`；阶段收尾补充结果与哈希索引。

实现前置验证：216 passed、0 failed，247.61 秒；最终资格标记调整后的相关检查
24 passed、0 failed，0.48 秒。日志见 `raw/revision_scope.junit.xml` 和
`raw/revision_final_scope.junit.xml`。这些是 CPU 验证，不授予真实运行资格；没有
重复整组千项回归，没有重跑 G1/G2/G3，也未预留新 PPO 训练预算。
