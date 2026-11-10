# Codex 同步说明：Lite3 两阶段研究（v2）

日期：2026-11-10；状态更新：2026-11-11
对象：Codex
取代：[Codex 任务书 v1](lite3-ea-lpl-codex-brief-v1-20261105.md)。v1 的 P2 及以后各阶段已作废，P0、P1 的结果继续有效。

## 1. 当前进度

| 阶段 | 状态 | 位置 |
|---|---|---|
| EA-LPL P0 | 完成（2026-11-06） | [P0 报告](lite3-ea-lpl-p0-20261106/README.md) |
| EA-LPL P1 | 完成（2026-11-09，61b9aa0） | [P1 报告](lite3-ea-lpl-p1-20261109/README.md) |
| 旧 P2（单一地形在线学习） | 用户否决，不执行 | — |
| 研究方案 v2 | 用户已同意（2026-11-09，2b8a314） | [研究方案 v2](lite3-ea-lpl-research-plan-v2-20261109.md) |
| 阶段一实施计划 | 既定训练和复评结束，两个种子延长后 Q1–Q4 均不合格，停在 S1-D；未选出 θ1 | [最终报告](lite3-two-stage-stage1-20261109/README.md) |
| 后续工作 | 下台阶与平地退化诊断提案待用户批准；阶段二未启动 | [诊断提案](lite3-two-stage-stage1-20261109/next-step-proposal.md) |

进度日志：[lite3-ea-lpl-progress-log.md](lite3-ea-lpl-progress-log.md)，最新记录在最上面。

## 2. 研究目标（用户 2026-11-09 的要求）

先学会六种初始地形，然后在这六种地形的步态基础上，遇到新的地形或者更难的地形时，经过失败尝试后学会通过。

据此分成两个阶段：

1. **阶段一：基础步态。** 从 model_1998 继续 PPO 训练（加载 actor、critic、优化器和迭代计数），在 10 行 × 20 列的课程地形上训练，难度最高到 L2，不训练 L3。得到的策略 θ1 要在六种 Source 地形族（平地、多尺度粗糙、坡面、台阶、坡面加粗糙、摩擦变化）的 L1 和 L2 上通过资格评估，并保留 model_1998 的平地全向行走能力。
2. **阶段二：在线学习。** θ1 固定，在 L3、boxes、rocky road 上用小型残差适配器从失败中在线学习。主要结果是每种地形分别比较 Frozen 和 Updated（同一个初始适配器）；扩展实验是多种地形连续累积，并检查已掌握地形是否退化。阶段二的计划在阶段一结束后再写。

用户的选择记录：方案 B（Source 为六族 L1+L2，在线目标为 L3+boxes+rocky road）、方案 C（逐地形会话为主结果，连续累积为扩展）、阶段一方案 A（从 model_1998 全量继续 PPO）。

## 3. 阶段一计划要点

完整内容以[阶段一计划](superpowers/plans/2026-11-09-stage1-source-curriculum-training.md)为准，共 9 个任务：

1. 不依赖 Isaac 的课程核心：行参数表、列布局、结局判定、安全公式、各地形族统计。
2. Isaac Lab 接口：地形块速度指令、只对地形块生效的边界和完成判定、全向区截断、安全终止、课程升降级。
3. 地形函数、环境配置、任务注册 `SourceCurriculum-Deeprobotics-Lite3-v0`、PPO runner。
4. GPU 环境检查脚本。
5. 试运行 30 次迭代，然后种子 42、43 各训练 1500 次迭代；GPU1 可用显存不足时，两个种子在 GPU0 上依次训练。
6. 评估器支持指定 actor，生成阶段一评估配置。
7. 平地保持能力检查。
8. Q1–Q4 判定、种子选择、训练曲线。
9. 资格评估 24 次、平地检查 3 次、L3 预览 8 次，写阶段一报告。

预登记判定规则（研究方案 v2 第 5.6 节）：

- Q1：每个地形族合并完成率在 L1 不低于 90%，在 L2 不低于 85%。
- Q2：30 个条件每个不低于 24/32。
- Q3：每个等级的安全终止不超过 24/480。
- Q4：平地速度跟踪误差不超过 model_1998 的 1.10 倍，跌倒率不超过 model_1998 加 1 个百分点。
- 两个种子都不通过时，延长训练 1000 次迭代一次；仍不通过则停在决策点 S1-D，交给用户决定。

## 4. 分工与工作方式

- 执行者：Claude 完成任务 1–3，Codex 接续并完成任务 4–9 的既定训练、两轮资格评估与报告；因没有合格 θ1，按规则未执行 L3 预览，停在 S1-D。保留已有实现与执行记录。
- 工作树：`/home/tjx/rl_training_main/.worktrees/lite3-ea-lpl-20261106`，分支 `codex/lite3-ea-lpl-20261106`，远程 `jenny-tian/rl_training_main`（私有）。旧工作树 `lite3-source-parallel-qualification-20261027` 不再推进。
- 证据目录：`/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/ea-lpl-evidence/`，原始数据不进 git。
- 汇报要求（用户 2026-11-10）：每完成一个阶段，说清楚具体完成了什么，以及下一个阶段准备做什么；内容不要太多，也不要太少，大约一屏。阶段结束后停下，等用户同意再进入下一阶段。
- 阶段内部的改进选择由执行者自己决定，并在报告里说明。

## 5. 必须遵守的规则

- 测试命令必须带 `-p no:typeguard`：
  `PYTHONPATH=$PWD/source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -p no:typeguard <路径> -q`
- 必须设置 `PYTHONPATH=<工作树>/source/rl_training`，否则会导入 `/home/tjx/rl_training` 里的同名包。
- 只用 GPU0、GPU1，并且空闲显存不少于 12288 MiB；GPU2、GPU3 不是我们的。
- 提交时只 `git add` 明确的路径，禁止 `-A`；`.superpowers/` 和 `logs/` 不提交。
- 不用不带标签的 `git stash`；需要暂存时用 WIP 提交。
- GPU 评估要求工作树干净时，不要修改 `source/` 和 `scripts/`。
- 每次有效改动都提交并推送，推送后用 `git ls-remote origin codex/lite3-ea-lpl-20261106` 核对远程与本地一致。
- 遇到安全或数据完整性问题、阈值或安全上限或预算的修改、决策点 S1-D、破坏性操作时停下，交给用户决定。
