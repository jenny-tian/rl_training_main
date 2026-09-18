# Lite3 候选训练连续性

v7 已在真实训练中验证成功奖励进入 PPO：std030 的第 3、5 回合严格完成，末步奖励 305，原始优势分别约 +219.061、+201.688，末块更新均接受。但第 5 回合冻结探针失败，supervisor 随即回退到先前验证模型，丢弃了该段候选 Actor、Critic 和 Adam 的后续进展。两组此前反复发生这一情况，100 回合上限并不等于同一候选获得 100 回合连续训练。

新增显式 `--probe-failure-policy continue_candidate`，用于当前仿真开发：探针失败仍记录失败，继续训练未验证候选；只有冻结探针严格通过才更新 best/recovery。原默认 rollback 保留，所有终止、安全阈值、完成资格和稳定窗口标准保持原样。NaN、策略版本错配等更新保护仍由原 updater 执行。

`--initial-training-checkpoint` 只启动训练分支，不会标为已验证。原 `--initial-checkpoint` 单独保存验证备份。运行清单分别记录文件及哈希，改变候选保留策略必须使用新配置清单；日志分开记录 probe_failures、probe_rollbacks 和 probe_retained_candidates，latest_probe 明确记录 promote、rollback 或 continue_candidate，不把继续训练写成验证成功。

该选择允许仿真中的临时退化继续提供学习数据，不意味着未验证模型适合真机部署。当前仍以最近 10 回合至少 80% 严格成功、至少连续 3 次成功、窗口内无跌倒、冻结探针通过作为稳定标准，并另做开发地形冻结复核。

联网核对了 [SLowRL 原文](https://arxiv.org/html/2603.17092)：其 LoRA 学习率为 1e-2，同时适配 Actor/Critic 并使用训练期恢复策略；实验是已有运动技能的仿真到仿真/真机迁移，不能据此宣称本项目已能从平地获得新地形能力，也不能直接复制其学习率和结果。本项目实际参数继续由各 run_manifest.json 登记。

supervisor 实验版本升级为 v8，worker 保持 `requested_ppo_absorbing_success_critic_context_v7`，奖励与观测合同没有在此次修改中改变。
