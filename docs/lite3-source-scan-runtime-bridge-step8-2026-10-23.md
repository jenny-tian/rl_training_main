# Lite3 Source 扫描运行器与 SafetyFilter 桥接：第 8 步

日期：2026-10-23（Asia/Shanghai）。本步实现冻结 PPO 推理与 SafetyFilter 桥接，补上扫描运行器的终止前采集，并执行受限 Flat L1 仿真 smoke。此结果不构成完整回合资格或 Source 能力结论。

## 实现

- `source/rl_training/rl_training/source_scan/bridge.py`：记录冻结策略原始动作、Lite3 动作边界裁剪、请求动作、过滤器执行候选动作、过滤器投影前状态和推理/过滤延迟。SafetyFilter 报告暂停时，运行器记录候选动作后立即停止，不把该动作传入环境。
- `scripts/reinforcement_learning/online_adaptation/run_source_qualification_scan.py`：完整扫描入口仍先要求 v0.2 manifest 通过执行预检；修正 RSL wrapper 的 seed-then-reset 顺序，并在 Isaac Lab 自动 reset 前复制终止状态和触发的 termination term。完整扫描结果将缺少的指标显式标为 unavailable，且不输出资格声明。
- `scripts/tools/source_terrain_policy_safety_smoke.py`：限定到登记的未授权准备稿、Flat L1 单元、固定 checkpoint 和不超过 200 步；输出文件不可覆盖，不训练或保存策略。
- `tests/source_scan/test_bridge.py`、`test_runner_smoke.py`：覆盖动作裁剪与 trace、非有限和错误维度动作、暂停停止、自动 reset 前终止态采集、固定命令分布和 wrapper seed/reset 顺序。

运行时发现，速度命令实现另有 standing、纯横移、纯偏航和零速度抽样分支。原运行器只限定速度范围，仍可能在单环境下抽到非前进命令。本步把扫描分布固定为 100% 纯前向，并通过仿真读回确认平均命令为 0.300 m/s。

## 受限仿真 smoke

运行仅使用 `model_1998.pt` 和 `flat_L1_forward_flat`，最多推进 200 个控制步。完整动作与状态 trace 已归档在 `docs/lite3-source-scan-preparation-v0.2-2026-10-22/policy_safety_smoke_20261023_flat_l1.json`，SHA-256：`3231447f3cdec2544b1623b15595db0bc351565b72127de8b2debc4004c4f650`。

- 执行 200 步后按 `smoke_step_limit` 结束；不是完整 episode，未达到地形终点。
- checkpoint SHA-256 与准备稿一致；actor 状态前后哈希一致；`policy_updated=false`、`sealed_accessed=false`。
- 200 组 12 维动作全部有限；请求与执行候选最大差为 0；过滤器投影率、暂停数和非有限状态数均为 0。
- 平均命令速度为 0.300 m/s，平均机身前向速度约 0.294 m/s，记录前进约 1.17 m；平均速度误差约 0.039 m/s。
- Isaac Sim 报告 Lite3 足端 visual sublayer 的 `Lite3_physics.usd` 引用无法解析，但 articulation 与场景照常建立并完成步进。此警告与此前 Source Terrain 接入记录一致。

## 验证边界与后续

静态编译、`git diff --check` 和 Source terrain / Source scan / SafetyFilter 相关测试通过（207 passed；其中桥接与运行器新增回归包含在内）。离线测试覆盖了自动 reset 回调和暂停分支；实际 200 步仿真没有发生 episode 终止，因此没有在该次仿真中触发终止回调。

本步未运行 225 回合扫描、PPO 训练、模型更新、Development 选择或 Sealed 评估。准备稿仍是 `draft_not_for_run`，完整扫描入口仍会拒绝它。

资格化前仍需补齐并复核共同滑移比例的阈值定义，以及 Rough、Stairs、Friction Variation 的专用指标（局部实测峰谷差、逐阶进度/碰撞与足间隙、接触摩擦及分区滑移）；还需用独立运行验证真实 episode 终止回调及完整报告合同。完成这些工作后再重新审计扫描准备稿。当前结果只能证明冻结推理、安全过滤和受限仿真接线可运行，不能据此选择训练难度或开始完整扫描。
