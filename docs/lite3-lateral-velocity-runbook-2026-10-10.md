# 世界横向回移速度消融：运行与审计

工作目录为本项目 `feat/lite3-policy-learning` 工作树。Python 使用 `/home/tjx/anaconda3/envs/env_isaaclab/bin/python`，`PYTHONPATH=.:source/rl_training`。物理运行使用 `--device cuda:1 --headless` 并取消 DISPLAY。输出目录不可覆盖；重新运行物理实验前必须另登记预算，原失败或中止账本保留。

## 固定方案

在无尾段保持的回线航向配方上，只新增 `-8 × (v_y - clip(-0.5 y, ±0.15))²`，由奖励管理器乘控制周期 0.02s。y 是相对跑道中心线的世界横偏，v_y 是世界坐标横向速度；目标用于训练奖励，不作为推理时控制器或额外观测。训练仍采用48D Actor、51D Critic、12D关节动作，从6a监督Actor初始化，fresh critic/Adam、learner2281、64×32×256、lr3e-6、std0.05；四组自均值保持100、无尾段保持、在线普通/前12恢复步保持100、机身/足端横偏−8、航向−1/1m/20°均不变。

校准只读上一轮全部 TRAIN 轨迹：固定2秒回移时间尺度、限幅为前进命令一半；系数由恢复age≥12的速度误差代价与10%的 `8y²` 机身横偏代理匹配，取最接近的2次幂并限制在[0.5,8]，得到−8。这个代理不等于完整机身/足端最大横偏奖励。未读取pilot/留出轨迹来定参数。旧策略不断变化，统计有存活选择影响。

IsaacLab `ArticulationData.root_state_w` 的位姿属于根连杆坐标原点，线/角速度属于根连杆质心，均以世界坐标表达。因此本轮精确定义是“根连杆原点的路线横偏”和“根连杆质心的世界横向速度”，不是全机器人质心速度，也不能把速度积分严格等同于原点位移。校准、训练奖励及独立重算使用同一约定。该约定来自 `/home/tjx/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py` 的 `root_state_w` 实现。

## 不可变输入

- [预登记计划](lite3_lateral_velocity_plan_20261010.json)
- [TRAIN校准](lite3_lateral_velocity_calibration_20261010.json)
- [工程协议](lite3_lateral_velocity_smoke_protocol_20261010.json)
- [训练协议](lite3_lateral_velocity_train_protocol_20261010.json)
- [冻结pilot协议](lite3_lateral_velocity_pilot_protocol_20261010.json)

工程预算25600、训练524288、pilot上限256000，只有原六门槛全部通过才登记原7来源seed2081候选验证上限256000；总上限1061888。选择唯一最终256轮模型，不挑中间checkpoint、不换种子、不放宽70cm。此前恢复/FlowDAgger账本charged33289984、confirmed33289920、历史uncertain64，旧完整PPO11900884另列。

## 复算顺序

1. `calibrate_lite3_lateral_velocity.py` 用 no_tail_heading TRAIN轨迹、训练/航向审计生成固定校准结果。
2. `pytest` 对 route/lateral_velocity/mean_anchor/recovery_anchor/vector_step_ledger 五个测试文件执行；使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，环境中无关typeguard插件存在依赖冲突。首次插件加载失败未运行任何测试，后续40项通过。
3. `smoke_lite3_recovery_anchor.py` 执行工程协议；用 `audit_lite3_lateral_velocity.py`、`audit_lite3_route_heading.py`、`audit_lite3_recovery_anchor.py` 分别重算速度、航向和动作保持。与旧同seed工程运行全部400步配对，公共字段除总奖励外逐元素相同。
4. `train_lite3_policy.py` 执行训练协议。完成后 `audit_lite3_recovery_teacher.py --training` 复算数据隔离、完成/失败、动作限幅与实际历史。再运行前述三项奖励审计，动作/航向审计增加 `--training-audit`。
5. `audit_lite3_mean_anchor.py` 使用本轮计划/训练协议，传入训练、动作保持、航向及 `--velocity-audit`。核对主回放抽样、冻结目标、零留出更新、唯一最终模型、首32步公共字段一致，奖励差只由新项解释。变更仅限3个运行源文件，其余173个与对照相同。
6. `evaluate_lite3_recovery_teacher.py` 执行固定pilot协议；随后 `audit_lite3_recovery_teacher.py` 重算全部轨迹，并抽样CPU回放动作。
7. `audit_lite3_prefix_imitation_pilot.py` 使用原 `lite3_prefix_imitation_plan_20261008.json` 判定既定门槛，复用 `artifacts/prefix_imitation_20261008/learner2247/pilot-evaluation` 的原PPO2063基线及其审计；要求初态/延迟哈希一致。本轮计划的pilot标准/种子与原计划逐项相同。原基线的hash与监督初始化6a不同，不能误将6a当成历史基线。
8. `summarize_lite3_mean_anchor.py` 使用本轮计划/更新审计/pilot结果、no_tail_heading对照及 `--engineering-audit` 汇总成本和图。全部配对恢复横偏用 `diagnose_lite3_recovery_lateral.py`，冻结世界速度用 `diagnose_lite3_frozen_return_velocity.py`；诊断不生成训练标签。

所有分析程序位于 `docs/analysis`，物理入口位于 `scripts/reinforcement_learning`。本轮产物统一前缀 `docs/lite3_lateral_velocity_`、物理目录 `artifacts/lateral_velocity_20261010/`。归档JSON依赖哈希固定，不修改旧报告。旧self_target更新审计及no_tail汇总已复算，结果完全一致。

## 本轮实际的分进程续跑

原 `pilot-evaluation` 只完成普通条件，在第二场景创建时停滞。原report/ledger未改写为完成；中断后依据[续跑登记](lite3_lateral_velocity_pilot_continuation_addendum_20261010.json)，用[恢复单条件协议](lite3_lateral_velocity_recovery_continuation_protocol_20261010.json)调用同一评测入口，输出 `artifacts/lateral_velocity_20261010/recovery-continuation`。

`assemble_lite3_lateral_velocity_pilot.py` 读取续跑登记、原始运行和续跑运行，验证全部公共协议字段、模型/源码哈希、完整条件manifest及不重复计费，输出 `artifacts/lateral_velocity_20261010/pilot-combined`。这个目录以符号链接引用两段完整原始轨迹，并明确标记两进程来源。步骤6之后的全部轨迹、门槛、汇总和诊断使用此组合目录；对应[组合审计](lite3_lateral_velocity_pilot_assembly_audit_20261010.json)。原普通和续跑恢复各87232控制，总174464，无普通重跑和额外控制预算。详见[完整结果及中断解释](lite3-lateral-velocity-result-2026-10-10.md)。
