# Lite3 实验登记表

依据：[执行方案 v2.2](lite3_execution_plan_v2_2_2026-09-28.md)及 `lite3-plan-audit-2026-09-28.md`；历史运行保留其原协议。
所有新运行先登记，再启动。调试结果不进入正式成功率统计。

## G0-20260928-base-pair

- 假设：当前 LoRA-PPO runner 的冻结 base 在 RockyRoad rough_4cm 上复现历史结果；若不一致，先定位协议/环境差异。
- 地形种子：101、103、107、109、113；simulation seed=7；纯本体感知，冻结 base，关闭探索与更新。
- checkpoint：`logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt`。
- SHA-256：`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。
- 每种子最多2000控制步，速度0.3 m/s，deadline=2000，严格赛道完成判定，物理终止即停止。地形属于调试集。
- 预期：101失败，103成功，107失败，109成功，113成功。不以调参强行匹配结果。
- 输出：`artifacts/g0_base_pair_20260928/`，每种子独立 runtime.log/config/summary/trajectory。
- 验收：summary和逐步记录步数一致；post-action/pre-reset的max_x与summary一致；冻结运行updates=0；保留失败和差异。
- 预算：最多10,000实际环境步；只做基线复核，不晋升策略。
- 状态：已完成，`complete_with_baseline_mismatch`；证据为输出根目录 `manifest.json`。
- 执行代码：`1113b4e`；1509 passed, 7 skipped。
- 首个结果：101在1785步完成、max_x=4.502410411834717、updates=0、无跌倒/安全违规，与历史预期不一致。轨迹/summary步数及max_x核对通过，轨迹671535字节。
- 完整结果：101/103/113 完成，107 跌倒失败，109 侧向越界失败；101 和 109 与历史成败不同。五次均 updates=0，逐步记录 max_x 与 summary 一致。
- 决定：记录基线差异，不调参凑历史结果；该预登记运行已结束，G1 效果判定等待 runner 差异审计。

## EXEC-20260928-v2.2

- 依据：[v2.2 详细执行方案](lite3_execution_plan_v2_2_2026-09-28.md)。
- 状态：G0 已开始执行；详见下方 runner-parity 登记与 [进展报告](lite3-g0-runner-parity-2026-09-28.md)。G1/G2 正式适应实验尚未启动。
- 当前硬门槛：先完成 runner parity、SAC 逐步轨迹、观察/终止/预算契约；G0 未通过前不得把适应结果写成学习收益。
- 路线：A LoRA-PPO、B Residual SAC、C 结构化低维修正，独立晋级；不预设主算法。
- 主 session 预算：40,000 个实际控制 transition；最终冻结评估 10 个共同扰动起点；平地保持 5 个共同起点。所有预热、候选筛选、探针和更新样本均计入。
- G3 候选：G0 通过且路线具备开发晋级证据后，最多 8 个失败池地形 × 3 次独立 session × 3 臂；具体地形和代码协议哈希锁定后再启动。
- 立即动作：P0/P1/P2 工程包；禁止直接重启高度扫描 teacher、重训 diffusion base 或启动封存实验。

## G0-20260928-runner-parity

- 假设：旧结果差异来自两个执行入口的配置、reset 随机状态或动作链；先记录最早分歧再修复。
- base SHA 同 G0-base-pair；terrain=rough_4cm，调试 101/109，simulation seed=7，速度 0.3，冻结且无探索；相同残差上限/rate=0.03。
- 首轮每入口每地形 128 步，共最多 512 步；float32 轨迹。输出 `artifacts/g0_runner_parity_20260928/pre_fix/`。
- 修复后先同样短复测（最多 512 步），再五个调试地形各两入口 2000 步（最多 20,000 步）。诊断累计仍须低于 v2.2 的 60,000 步上限。
- runner：`scripts/reinforcement_learning/online_adaptation/run_runner_parity.py`；manifest 记录代码 commit、未提交差异哈希、全部命令、退出码、实际步数及检查结果。
- 验收：trace/summary 步数、max_x、reward_sum 一致；updates=0；同一观测离线 base/action max_abs≤1e-6；物理漂移单独记录。旧训练语义在短诊断期间不作为 G0 已统一的证据。
- 状态：短诊断和五地形完整冻结复核均完成；103/109/113成功，101未完成，107跌倒，两个入口完整轨迹与结果一致。五地形实际16,046步，本组累计17,614步。
- 短诊断追加：`actuator_audit/`，101 两入口各 16 步（最多 32 步），只补记录实际执行器延迟以检查第一步物理分歧；不改变学习/动作设置。
- 实际目录：沙箱尝试 `pre_fix/` 为 0 步；CUDA 首轮 `pre_fix_gpu/` 512 步；`safety_fix/` 512 步；`actuator_audit/` 32 步；`reset_fix/` 512 步。短诊断累计 1568 步。
- 已定位：模型初始化消耗随机数不同，导致最后一次 reset 的 Knee actuator delay 为 LoRA=0、SAC=1；另有零残差安全投影舍入差异。固定初始 reset 种子、统一指令和可行动作投影后，101/109 的同观测动作及 128 步物理轨迹最大差异均为 0。
- 五地形输出 `five_seed/`，最多 20,000 步。SAC 通过状态以 `episode_summary.json.passed` 为准；后处理将其写回总 manifest，并生成 `runner_parity_report.json`。
- 当前代码验证：1516 passed, 7 skipped。完整 G0 仍等待共同奖励、任务终止、session 账本/恢复和封存守卫。

## G0-20260928-common-objective

- 假设：共用 `finite-horizon-v1` 后，同观测轨迹的逐步奖励、成功/失败/deadline 与 collector cutoff 标志、bootstrap 口径在 LoRA/SAC 一致。
- base SHA、rough_4cm、速度0.3、simulation seed=7 与上一登记相同；原协议五地形复核已完成，两入口全程物理/动作差异为0，103/109/113成功，101超时未完成，107跌倒。
- 新任务协议：固定普通奖励、机身/足端最大横偏、成功305、失败-8；任务成功/失败/deadline 不 bootstrap；采样 cutoff bootstrap。目标奖励是显式终点值。任务协议 SHA 单独记录，不冒用地形协议 SHA。
- 首轮：`artifacts/g0_task_objective_20260928/cutoff/`，101/103 × 两入口 × 128步，deadline2000，最多512步；`short_deadline/`，101 × 两入口 × 64步，deadline64，最多128步。
- 终点复核：`deadline/` seed101、`success/` seed103、`failure/` seed107，每种子两个入口最多2000步，总上限12,000步。配对入口在相同GPU运行，不同地形可分GPU；全部无学习。
- 验收：trace重建summary；逐步reward与endings一致；物理终止保留pre-reset末帧；成功、失败、deadline末帧bootstrap=0，128步cutoff bootstrap=1；更新次数为0。单元测试同时验证3-step短尾在截止处flush进入真实replay。
- 状态：全部诊断完成。cutoff、64步deadline及2000步deadline/成功/跌倒的reward与learning endings逐步一致，reward最大差异0；本组实际10,484步。详见 [G0.3报告](lite3-g0-task-objective-2026-09-28.md)。完整G0仍需预算/恢复/封存工作。
- 补充实际checkpoint诊断：`replay_smoke/`，SAC新建在线session，101，simulation seed7，64步deadline，n=3，replay capacity/min_replay=256，禁止在样本不足时更新；最多64步。核对保存的replay有64条，末尾三个n-step前缀保留terminal observation、bootstrap=0且不跨reset。该运行只验证保存与目标口径，不评估学习收益。
- 最终验证：1530 passed, 7 skipped；replay实际64条、actor/critic更新0次、最后三个前缀bootstrap均0，保存末帧校验通过。本轮诊断均结束。

## G0-20260928-session-ledger

- 假设：统一日志账本与checkpoint链能在独立worker进程之间保留预算/学习器状态；独立物理reset不重置学习RNG，最终评估不写replay或更新策略。
- base SHA同前；新公共任务协议deadline=128；terrain=rough_4cm seed101，learner seed7；每回合reset seed由固定的learner/phase/trial编号派生。
- 两个独立session，LoRA与SAC各适应预算256步（两个128步worker），最终冻结评估16步；每次最多一个worker/卡，最多两卡。单session三次独立supervisor调用验证续接；重启已耗尽的评估阶段应消费0新步。
- SAC smoke：replay capacity=256，min_replay=128，n=3，critic UTD=1，actor更新预算=0，novelty calibration=100000；用于验证Q/replay/optimizer续接，不作算法收益判断。LoRA使用既有默认优化器。
- 输出：`artifacts/g0_session_ledger_20260928/{lora,sac}/`，含不可变manifest、逐步journal、每trial独立trajectory/checkpoint/summary。
- 验收：每session最终charged=confirmed=272、uncertain=0；第二worker输入为第一worker的已提交checkpoint；replay/优化器延续；evaluation更新与replay写入均0；重复resume不重跑已完成ID；单测覆盖崩溃pending保守计费、torn write、上限、锁定sealed协议及评估后禁止回流。
- 状态：已登记，准备运行。新封存seed和最终论文协议不在本smoke中生成。
- 第一轮结果：两路线各256适应+16冻结评估，重复resume新增0步；LoRA Adam步数4→8，SAC replay128→256。SAC默认alpha=1导致全部Q更新被现有mean_td_error_limit拒绝，不能据此声称其Adam恢复已验证。
- 追加 `sac_lowalpha/` 独立session，唯一超参变化initial_alpha=0.02，仍不放宽TD保护阈值；额外上限272步，用于产生非空Adam状态并检查恢复。恢复审查同时补充保存/恢复未消费的critic更新额度；actor额度属于上一物理适应回合，在新回合清零。保留原失败配置，不把这次工程诊断当作G1超参选择结论。
- 最终状态：三个session各272步，全部结束；本轮816步、uncertain=0。低温度SAC Adam步数2→131，replay128→256，critic剩余额度126→125；冻结评估更新/写replay均0，重复resume新增0步。验证 `validation.json`；完整测试1550 passed, 7 skipped。详见 [G0.4报告](lite3-g0-session-ledger-2026-09-28.md)。


## G0-20260928-initial-state-schedule

- 已预登记：测试 perturbed-v1（x/y ±0.02 m，yaw ±2°）在两个入口的初态/冻结轨迹一致性；验证阶段独立种子、付费探针、固定评估次数、真实平面稳态指标和冻结零写入。
- base SHA 同前；仅 development rough_4cm seed101，learner seed7，deadline2000。最多两GPU、每卡一个worker；不训练正式长任务，不生成sealed种子。
- `artifacts/g0_initial_state_schedule_20260928/{lora,sac}/`：各B=256，probe标记128/256，terrain trial64，flat trial32，selection cap160；最终评估2×64，retention2×32。单路线最多448步，两路线896步；重复resume不得新增试验。
- `base_{lora,sac}/`：直接冻结base，各1×320步地形评估+1×320步真实平面，独立初态序列；总上限1280步。平面超过5秒，必须提供稳态速度指标。
- 本组总上限2176个实际控制transition。所有启动失败/无效初态与消耗保留，不按任务成败丢弃。初态范围的全几何域验收仍须单列，不以两起点smoke替代。

- 完成：实际2176步，四session全部结束；uncertain=0。base两入口地形/平面各320步，物理/观测/动作/reward/endings max_abs=0。平面稳态XY MAE两者均0.02276685099641327 m/s。学习session各448步，探针计入B、冻结零更新/零replay写入、重复retention resume新增0步。详见 [G0.5报告](lite3-g0-initial-state-schedule-2026-09-28.md)。


## G0-20260929-spawn-geometry

- 预登记出生几何审核：实际 RockyRoad Lite3 USD 碰撞体，nominal reset + x/y ±0.02 m、yaw ±2° 的8角点。读取 PhysX body pose，不用旧USD世界姿态；控制transition预算0，不调用env.step。
- 核对各碰撞体对地间隙，用水平包围圆证明固定关节/高度下所有yaw及整个XY扰动范围均在平坦出生平台内。未知形状/缺失碰撞体直接报审核失败，所有案例保留，不按跑步成败筛选。
- 范围：development RockyRoad 默认配置、GPU0单worker；不生成封存种子。输出 `artifacts/g0_spawn_geometry_20260929/physical/`；USD只读组成检查另记0步。

- 首次物理审核只写入中心案例后进程结束，报告保持running、不认定通过，控制步0。追加 `physical_retry/`，同一9案例/0步方案，先补充异常持久化定位退出原因。

- retry定位：角点yaw的numpy.float64让Isaac reset产生Double姿态，与Float缓存冲突。修正为Python float，并使异常退出明确非零；追加 `physical_validated/` 同样9案例/0控制步，前两次诊断各只完成中心案例，不删除。

- 完成：`physical_validated/audit.json` 的9案例全部通过；实际yaw/XY偏移与设定误差<1e-6，关节姿态一致。22碰撞体的连续范围证明给出最小平台余量1.1036949118 m、最小对地间隙0.0265100528 m。总控制步0，失败诊断保留。完整测试1575 passed, 7 skipped；managed perturbed-v1现绑定模型/reset几何证书。详见 [G0.6报告](lite3-g0-spawn-geometry-2026-09-29.md)。

## G1-20260929-target-warmup-contract

- 预登记工程验证：两路线独立从共同 base 开始，development rough_4cm seed101，learner seed7，perturbed-v1，速度0.3，deadline2000，最多GPU0/1各一worker。
- 每路线 adaptation B=4512，含4000步冻结mean base目标域采样及最多512步后续学习；不做selection探针。先完成一个物理trial并检查checkpoint，再续跑剩余额度。evaluation最多1×64步，retention最多1×320步；总上限9792个实际控制transition。所有失败/不确定步均保留，不按结果补起点。
- LoRA沿用rollout256、epochs2、原lr/rank；warmup只拟合critic，lambda=1 block return在截断处bootstrap，明确不是完整回合MC诊断；完成或离开warmup时清空buffer后才允许actor使用新样本。
- SAC沿用n=1及现行网络、critic/actor每transition额度0.5/0.005，warmup不累计更新额度；收集完成后恰好1000次critic-only更新，再允许actor/alpha更新。actor联动的critic更新一并扣critic额度，仍保留novelty与安全门控。replay_capacity=5000、min_replay_size=256。initial_alpha=0.01沿用G0已验证的低温度工程配置（alpha=1曾触发TD守卫），此处不作超参数优越性结论。
- 输出：`artifacts/g1_target_warmup_20260929/{lora,sac}/`。检查warmup累计计数、actor/alpha/optimizer不变、base动作等价、PPO旧样本丢弃、SAC预热拒绝时禁止actor、resume不重复预热、冻结阶段零写入。短评估只测执行契约，不做路线晋级或平地保持效果判定。
- 保留完整步轨迹、SAC逐minibatch soft-target诊断和PPO块内value误差。完整回合4/2训练验证划分、两地形价值诊断及冻结效果验收仍是G1后续工作；不把训练内误差改善作为通过证据。

- 启动前检查发现两处入口问题：无探针时selection=0被旧manifest校验拒绝；SAC新预热参数误加在独立train parser。两次均在物理worker前退出、0控制步，日志保留为`{lora,sac}_first.log`。修正为允许零selection预算且仍校验探针总额，并把参数接到实际play parser；新增真实supervisor→worker参数回归后原预算重试。

- 运行中SAC多次20 ms控制超时，保留全部回合与reset序列；LoRA首trial完成后暂缓其续跑，让SAC单独执行，不放宽安全阈值。SAC在trial-00010累计4000物理采样步后完成1000次critic-only更新；该trial随后完成至206步，但完整预热记录把未计算actor_loss/alpha_loss的NaN占位直接放入严格JSON，summary落盘失败。ledger保留该trial为aborted，总计4087步、uncertain=0；checkpoint/float32完整trace/1000条更新诊断均保留，不重写原始状态。
- 检查该checkpoint：target_warmup=4000/1000；actor和alpha Adam状态仍为空，alpha=0.01；critic Adam=1043（1000次预热＋随后87物理步中的43次常规critic更新）。本组不再追加SAC长采样。
- 追加登记仅验证日志修复与恢复：独立`sac_logging_fix/`，新source-bound session，从原base新初始化；warmup=256、B=512、min_replay=128、critic-only=1000、其余沿用原SAC配置；eval1×64、retention1×320，上限896步。缩短warmup只用于工程回归，不是G1新候选或效果结论。新总上限10688步；LoRA原B及评估额度不变。旧SAC最后checkpoint仅作诊断证据，不跨源码静默续接或晋升。

- 完成核验：本轮9748控制步（LoRA4896、SAC原诊断4087、SAC修复回归765），uncertain=0；两个主验证各4000个warmup动作均与base差异0。LoRA预热128次critic minibatch/0 actor，随后512步才发生2次actor更新；SAC1000次预热完成，原末trial仍为aborted。修复回归checkpoint续接不重复1000次预热；冻结阶段零更新/零replay写入，重复阶段调用新增0步。
- 保留限制：SAC控制超时在单worker阶段也出现；修复回归冻结评估2步、平地251步提前结束，平地比较不合格，不能宣称保持能力或G1路线通过。原预热NaN只是未计算loss的日志占位；实际critic诊断有限，修复后严格JSON不含NaN。完整回归1592 passed, 7 skipped。详见[G1.1报告](lite3-g1-target-warmup-2026-09-29.md)及[验证JSON](lite3_g1_warmup_validation_20260929.json)。
