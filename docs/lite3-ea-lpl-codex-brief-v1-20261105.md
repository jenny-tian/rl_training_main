# Codex 执行任务书：Lite3 EA-LPL P0 到 P2（v1）

> **2026-11-10 已作废：** P2 及以后各阶段不再执行。当前进度、目标和规则见 [Codex 同步说明 v2](lite3-two-stage-codex-brief-v2-20261110.md)。

日期：2026-11-05
对象：Codex
总方案：[EA-LPL 研究方案 v1](lite3-ea-lpl-research-plan-v1-20261105.md)，判定规则以它的第 5 节为准。
阶段计划：

1. [P0 航向闭环基础策略与目标条件](superpowers/plans/2026-11-05-ea-lpl-p0-heading-baseline.md)
2. [P1 动作基、改进空间与执行响应诊断](superpowers/plans/2026-11-05-ea-lpl-p1-action-headroom.md)
3. [P2 C1-a 并行 session 在线学习](superpowers/plans/2026-11-05-ea-lpl-p2-c1a-sessions.md)

## 1. 要做什么

把研究从 Source PPO 小步微调转到计划书的 EA-LPL 路线。基础策略固定为 `model_1998.pt` 加航向闭环，在它失败的地形上测试二维残差适配器的在线学习。P0 建立基线并选目标地形，P1 选动作基并确认改进空间，P2 做 C1-a 的 Frozen 对 Updated 比较。

Source PPO 小步微调暂停。update1 阶段提出的离线奖励与观测设计方案不执行。已有代码、checkpoint 和归档保持不动，可以复用。

## 2. 授权与自主范围

用户一次批准 P0、P1、P2 三个阶段。阶段之间按研究方案第 5 节的规则自动衔接：

- P0 选出目标条件后直接开始 P1。
- P1 有确认的改进空间后直接开始 P2。
- 每个阶段结束写报告、提交、推送，然后继续下一阶段，不等待确认。

只有遇到第 7 节的停止条件或用户决策点时才停下。

## 3. 加速规则（必须遵守）

过去的流程把大量时间花在每次运行前后的手续上。本任务改用下面的做法：

| 以前的做法 | 现在的做法 |
|---|---|
| 每次运行前单独开一个准备阶段，冻结 237 项输入哈希，写 authorization.json 和提案 SHA 链 | 每次运行只写一个 manifest：git 提交、spec 文件、checkpoint SHA256、命令行、GPU、起止时间、退出码 |
| 每次运行前重新校验历史归档的 SHA256SUMS | 不再校验历史归档，只核对本次用到的 checkpoint 哈希 |
| 每个小改动前后跑 600 多项全仓库回归 | 每个任务只跑本任务新增或修改模块的测试，加一次冒烟运行；阶段结束时跑一次 `tests/ea_lpl` 和 `tests/source_scan` |
| 原始 npz、逐步日志提交进 Git | 原始数据放证据目录（第 6 节），Git 只提交汇总 JSON、CSV、图和 SHA256SUMS |
| 实验结果不好就停下等待批准 | 实验结果不好也是结果，照常记录，按规则进入下一步 |
| 一次只用 GPU0 | GPU0、GPU1 都可用，空闲显存 ≥ 12288 MiB 就能启动，允许与他人共享；互不依赖的运行在两张卡上同时跑 |
| 基础设施失败后停下等待 | 崩溃、显存不足、启动超时、GPU 被占等基础设施失败自动重试 1 次并记录原因；同类失败连续 2 次才停下汇报 |
| 每一步写一份长 README | 每个阶段一份 README，不超过 2 页，用第 8 节模板；中途只在进度日志里加一行 |

另外：

- 预登记的门槛和规则不能在看到数据后修改。觉得不合理就在报告里写建议，仍按原规则执行。
- 每个任务有预计用时。超过 2 倍时在进度日志写一行原因，然后继续或请求决策。
- 被某个决定卡住时，把问题写进进度日志，先做不依赖它的任务。

## 4. 工作区与 Git

1. 在当前分支 `codex/lite3-source-parallel-qualification-20261027` 上找到添加本任务书的提交（提交说明以 `docs: add EA-LPL research plan` 开头），从这个提交建新分支和工作区：

```bash
cd /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
git worktree add ../lite3-ea-lpl-20261106 -b codex/lite3-ea-lpl-20261106 <该提交的SHA>
cd ../lite3-ea-lpl-20261106
git push -u origin codex/lite3-ea-lpl-20261106
git ls-remote --heads origin codex/lite3-ea-lpl-20261106
```

2. 之后所有 EA-LPL 改动都在这个工作区提交，每个任务结束提交并推送一次，推送后用 `git ls-remote` 核对远端 SHA。远端是私有仓库 `jenny-tian/rl_training_main`。
3. `remote.origin.fetch` 只映射了 `feat/lite3-online-adaptation`，`origin/<分支>` 跟踪引用会过期，核对远端一律用 `git ls-remote`。
4. 进度日志：新建 `docs/lite3-ea-lpl-progress-log.md`，最新记录写在最上面，每条不超过 5 行。旧方案日志 `docs/lite3-source-parallel-runtime-qualification-plan-v0.1-draft-2026-10-27.md` 顶部已经有一条指向本任务书的记录，不需要再更新它。

## 5. 运行环境与命令

```bash
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
# CPU 测试。env_isaaclab 里可编辑安装的 rl_training 指向参考项目 /home/tjx/rl_training，
# 必须把本工作区放在 PYTHONPATH 最前面；typeguard 插件会让 pytest 启动失败，必须关闭插件自动加载。
PYTHONPATH=source/rl_training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/ea_lpl -q
# GPU 运行统一通过 scripts/tools/ea_lpl_run.py 的监督进程启动，它负责选卡、超时和清理。
```

- 基础策略：`/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt`，SHA256 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`，用 `rl_training.source_scan.frozen_actor.load_frozen_source_actor` 加载。
- 环境：120 个并行环境、共享地形库 1 行、50 Hz、每回合最多 2000 控制步，完成条件为前进 9 m 后连续稳定 50 步。这些都不改。
- 安全过滤器上限不改。

## 6. 证据目录

原始数据放在 Git 公共目录下：

```bash
EVID=$(cd "$(git rev-parse --git-common-dir)" && pwd)/ea-lpl-evidence
# 例如 $EVID/p0/p0_e1p_l1/
```

每次运行目录包含 manifest.json、spec.json、runtime.log、result.json、npz、事件文件和 execution.json。阶段结束时为证据目录生成 SHA256SUMS，复制到对应的 docs 阶段目录一并提交。

## 7. 停止条件与用户决策点

立即停止该运行并汇报：

- 非有限状态、奖励或动作；跨环境接触；数据文件损坏；安全过滤器异常（例如暂停状态下仍执行了动作）。
- 需要修改安全过滤器上限、动作缩放、完成条件、判定门槛或预算上限。
- 任何访问封存地形（boxes、pyramid_stairs、pyramid_stairs_inv）或封存命名空间 9 的操作。

用户决策点（停下写报告，等待用户选择）：

- D-0：P0 选不出目标条件。
- D-1：P1 所有动作基都没有确认的改进空间。
- D-2：P2 未通过主判定。

预算上限（超过即停，次数含重试）：P0 不超过 8 次仿真运行、2M 条交互；P1 不超过 30 次、8M 条；P2 每个目标等级训练不超过 15M 条、评估不超过 8M 条。

## 8. 阶段报告模板

阶段目录：`docs/lite3-ea-lpl-p<编号>-<日期>/README.md`，不超过 2 页。

```markdown
# P<编号> <名称>（日期）

结论：通过 / 未通过 / 进入决策点 D-x。关键数字一行。下一步：自动进入 P<n+1> / 等待用户。

| 运行 | git 提交 | GPU | 环境数 | 回合数 | 交互数 | 用时 | 退出码 |
|---|---|---|---|---|---|---|---|

主要结果表（各阶段计划给出固定列）。

| 预登记规则 | 实际值 | 结果 |
|---|---|---|

偏离与异常：没有就写无。
文件：summary.json、图、证据目录路径、SHA256SUMS。
```

进度日志每个阶段结束加一条，格式：日期、阶段、结论、下一步，不超过 5 行。

## 9. 不要做的事

- 不改 `model_1998.pt`，不训练基础策略，不继续 Source PPO。
- 不修改既有归档目录里的任何文件。
- 不为同一个运行写多份报告；不在汇报里堆放可以从 summary.json 读到的数字。
- 不把 GPU 运行放进 pytest。
