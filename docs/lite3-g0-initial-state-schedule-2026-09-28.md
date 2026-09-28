# G0.5 初态分组、付费探针与平地评估

已实现并验证阶段独立初态、主预算内探针、固定次数最终评估、真正平面上的统一稳态指标，以及连续短跌倒停止规则。本轮是执行协议诊断，不能作为适应收益或 G1 warmup 完成的证据。

## 执行约定

- 两入口新增 `--initial-state {fixed,perturbed-v1}`；旧独立命令默认 fixed，统一 session 使用 x/y 各 ±0.02 m、yaw ±2°，其余 root/joint reset 保持既有口径。轨迹记录分布和实际初始 root pose/velocity。
- 初态种子由 learner seed、phase 和该 phase 的尝试编号派生。增加适应/探针次数不会改变最终评估起点；相同 learner seed 的两算法和 base 共用最终起点。中断的尝试也消耗其起点槽位，不按成败替换。
- manifest 升级为 `lite3_session_v2`，锁定次数、回合长度、探针调度和停止阈值；旧 v1 账本可读取，但旧 source-bound session 不能用新源码续训。
- 默认 B=40000，其中 selection 与 flat_selection 共用最多8000。满长默认调度：7000开始地形探针2000步、9000开始平地探针1000步；17000/19000重复；38000开始最后地形探针2000步。提前结束的未用额度返回学习采集，不增加探针次数。学习采集在下一探针边界截断。
- 探针关闭学习，不推进学习 checkpoint。当前候选规则仍是最后一次完整适应回合的 checkpoint；尚未实现按探针得分晋升/rollback。
- 最终 evaluation 最多10次、每次2000步；retention 最多5次、每次1000步。提前结束不会追加起点填满预算。两阶段锁定同一候选，禁止返回适应/选择。
- 连续5个不足100步的学习回合跌倒后停止适应；探针不重置该连续计数。重启账本后仍生效，summary 明确标记适应提前停止和失败原因，最终无反馈测量仍可进行。
- SAC 平地入口现在使用无限平面，并与 LoRA 共用5秒排除期、body-frame XY速度MAE、进度与数据资格指标。短于排除期的数据标记 unavailable，不能作保持通过结论。

运行入口：`scripts/reinforcement_learning/online_adaptation/run_adaptation_session.py`。同一 session 依次以 `--phase adaptation`、`--phase evaluation`、`--phase retention` 启动；配置需一致。开发小预算可显式指定 `--probe-milestones`、`--evaluation-trials` 和 `--retention-trial-steps`，全部进入不可变 manifest。

## 实际验证

产物：`artifacts/g0_initial_state_schedule_20260928/`，包括 `runner.py`、`validate.py`、`validation.json`、各 trial 的日志、配置、轨迹与checkpoint链。

| Session | 学习采集 | 地形探针 | 平地探针 | 最终地形 | 最终平地 | 总计 |
|---|---:|---:|---:|---:|---:|---:|
| LoRA |96|128|32|128|64|448|
| SAC |96|128|32|128|64|448|
| base / LoRA入口 |0|0|0|320|320|640|
| base / SAC入口 |0|0|0|320|320|640|

合计2176 transition，全部 confirmed=charged，uncertain=0。每条适应路线9个独立worker，base路线各2个；最后 retention 重复 resume 新增0步。探针/最终评估更新与 replay 写入均0，候选链没有被探针改写。

两入口 base 在 rough_4cm seed101 与真实平面上分别320步：初始13维root状态、观测、动作、物理轨迹、reward与endings逐步完全一致（max_abs=0；不比较墙钟时间）。平面稳态70个样本，两入口 XY MAE 均为 **0.02276685099641327 m/s**，comparison_eligible=true。短平地探针与短retention均正确标为无稳态数据。

最终完整回归 **1564 passed, 7 skipped**，`git diff --check` 通过。GPU运行时源码已存档为 `validated_smoke_source.tar.gz`，其指纹与实验manifest一致；该summary输出补充不改变物理/调度路径。

本组与前述G0账本、奖励、配对组累计31090步（原 G0-base-pair 另记）。没有启动长训练或新sealed seed评估。

## 协议检查与后续门槛

[protocol_v2_2.json](protocol_v2_2.json)记录已实现的开发执行约定、验证证据与未锁定项，明确**不是最终sealed协议**。本次扰动只完成 seed101 和少量起点的物理一致性诊断；建立失败池前仍需完成全扰动范围的出生几何审核，不得把本smoke当作全域证明。

G1的4000步目标域warmup、两路线学习信号修复尚未实现。候选晋升/rollback规则、相对base的平地保持阈值及最终算法配置/封存成员集合仍须按计划后续锁定。上述缺口不会由工程测试通过自动消除。
