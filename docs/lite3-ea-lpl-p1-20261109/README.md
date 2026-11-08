# Lite3 EA-LPL P1：动作余量筛选、动作基选择与 EA 前提判定

日期 2026-11-09，分支 `codex/lite3-ea-lpl-20261106`。证据目录 `<git-common-dir>/ea-lpl-evidence/p1`，校验和见 `SHA256SUMS`。

## 结论

- 选定动作基：**roll**（得分 sag 0、roll 0.9375、hipx 0）。
- P2 目标：L1 第 5 列右横坡（`slope_cross_slope_right`），预言码 `roll:-2,+2`，确认增益 15/16。
- 第 4 列左横坡的 roll 码完成数明显提高（最高 12/16），但安全终止 3 次以上，超过零码 0 次 + 1 的上限，未通过确认。
- EA 判定：C1-a 不通过（p_proj 0.017%，p_guard 0.003%，阈值 5%）；C1-b 通过（roll 基宏偏差比例 81.7%，预测误差中位数 0.216）。
- 按阶段审批规则，P1 结束后停止，等待用户批准再进入 P2。

## 主要结果

| 条件 | 动作基 | 最佳筛选码 | 筛选 完成/8（零码 完成/8） | 确认 完成/16（零码 完成/16） | 安全终止 | 是否确认 |
|---|---|---|---|---|---|---|
| L1 c4 左横坡 | roll | roll:+2,-1 | 7 (0) | 10 (0) | 4 | 否 |
| L1 c4 左横坡 | roll | roll:+1,-1 | 6 (0) | 12 (0) | 3 | 否 |
| L1 c4 左横坡 | roll | roll:+1,-2 | 3 (0) | 6 (0) | 9 | 否 |
| L1 c5 右横坡 | roll | roll:-2,+2 | 8 (0) | 15 (0) | 0 | 是 |
| L1 c5 右横坡 | roll | roll:-1,+2 | 4 (0) | 5 (0) | 4 | 否 |
| L1 c5 右横坡 | roll | roll:-2,+1 | 2 (0) | 0 (0) | 7 | 否 |
| L1 c8 粗糙上坡 | 全部 | 无 | 全部 0 (0) | 零码 0 | 0 | 无候选 |
| 任一条件 | sag / hipx | 无 | 全部 0 (0) | 未入围 | - | - |

零码在确认运行中三个条件均为 0/16、安全终止 0 次。完整筛选表见 `screen.csv`（3 条件 × 73 码 × 8 起点），确认表见 `confirm.csv`，5×5 完成数热图见 `headroom_heatmaps.png`。

## EA 诊断（`ea_diagnostics.json`）

- C1-a（固定码执行时的投影和保护触发比例，按样本加权，1890672 个样本）：p_proj = 1.70e-4，p_guard = 3.33e-5。不通过。
- C1-b（随机切换码的开环回放，种子 0）：

| 动作基 | 决策数 | 单步偏差比例 | 宏步偏差比例 | 预测误差中位数 |
|---|---|---|---|---|
| sag | 73527 | 0.988 | 0.817 | 0.217 |
| roll | 56411 | 0.988 | 0.817 | 0.216 |
| hipx | 58450 | 0.980 | 0.027 | 0.208 |

- 回放保真：第一个文件和全部 10 个有投影或保护活动的文件（00、02、03、04、06、07、08、09、13、14）均做了回放，最大误差 0.0。

## 运行与预算

- P1 共 18 次运行（1 次零码复核 + 15 次网格 + 2 次确认），全部 passed，交互步数 2328052（预算 ≤30 次、≤8M）。仿真墙钟合计约 2087 s；确认运行在 GPU0/GPU1 并行，各约 130 s。筛选分析（主要是回放保真检查）CPU 耗时约 74 分钟。
- 零码复核 `p1_zero_check_l1` 与 P0 基线 `p0/p0_e1p_l1_retry1` 完全一致（`p0/p0_e1p_l1` 是 P0 首次失败的运行，不作基线）。
- 网格规格 11-14 第一次启动时被干净工作区检查拒绝（仿真尚未开始，无数据损失），之后重新运行。
- 确认规格使用命名空间 3、起点槽位 0-15，与筛选（命名空间 2）不重叠。

## 复现

```bash
PY=/home/tjx/anaconda3/envs/env_isaaclab/bin/python
$PY scripts/tools/ea_lpl_p1_analysis.py screen --evidence $EVID1 --inputs docs/lite3-ea-lpl-p0-20261106/p1_inputs.json --out docs/lite3-ea-lpl-p1-20261109
$PY scripts/tools/ea_lpl_run.py --spec docs/lite3-ea-lpl-p1-20261109/specs/p1_confirm_l1_00.json --output $EVID1/p1_confirm_l1_00 --gpu 0
$PY scripts/tools/ea_lpl_run.py --spec docs/lite3-ea-lpl-p1-20261109/specs/p1_confirm_l1_01.json --output $EVID1/p1_confirm_l1_01 --gpu 1
$PY scripts/tools/ea_lpl_p1_analysis.py final --evidence $EVID1 --out docs/lite3-ea-lpl-p1-20261109
```
