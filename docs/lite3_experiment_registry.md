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

## G1-20260929-control-latency

- 预登记控制时延诊断：先记录GC回调重叠时间、当前线程CPU时间、base/adapter+安全/device同步三个宿主阶段；原20 ms控制计算阈值不变。不停止其他项目进程，不根据成功与否替换起点。
- baseline：独立managed SAC，直接冻结base，development rough_4cm101、learner7、perturbed-v1，最多2个预定地形起点×2000步；不更新、不写replay，不续接旧source-bound session。输出`artifacts/g1_control_latency_20260929/baseline/`；首轮最多4000实际控制步。此次只定位运行时尖峰，不用于路线晋升。
- 若发现可修复的确定性开销，再登记相同控制阈值下的修复复核预算。所有失败、未使用额度及计算诊断保留。

- baseline结果：两个预定trial分别1步、705步控制超时，共706步。回退前24.14/26.61 ms，线程CPU约24.14/26.60 ms；GC重叠均0。回退后旧`control_compute_ms`为39.09/41.28 ms，不能把回退计算误计为触发阈值的原始耗时。
- 0物理步离线重放进一步定位：慢调用的8次Tensor.item累计约16 ms，两次固定动作边界torch.tensor约4 ms。1000次CUDA重放中190次>20 ms；同数据CPU单线程无>20 ms。优化缓存固定边界、5个安全原因一次回传、terrain-valid标量每步只读一次；保留全部安全条件。加入共享动作/稳定性文件的manifest源码哈希，异常checkpoint补齐已有target_warmup状态。
- 追加预登记修复仿真：`optimized/`同一base/地形/learner，2个共同预定地形起点各最多2000步，2个平地起点各最多1000步，总上限6000步；直接冻结base、无学习/无replay写入、20 ms阈值不变。结束后重复phase调用应新增0步。与baseline数据一起保留，不把控制窗口完成视为策略适应收益。

- 同进程交错512步对照：CPU/CUDA各512步，新旧action/residual/安全原因/历史/replay待写字段逐位一致。CUDA交错对照的两个实现均未超过20 ms（当时系统负载不同），因此独立重放190→1的超时变化只作运行时诊断，不宣称统计因果；独立重放唯一优化后超时是未预热的首调用。真实worker保留原启动预热和20 ms阈值。
- 追加预登记学习/恢复计时smoke：`learning/`，B=2000，trial_steps=1000，warmup=256，critic-only=1000，replay=5000/min=128、alpha=.01、原0.5/0.005更新额度；deadline2000，仍是101/learner7，依次adaptation→evaluation(1×320)→retention(1×320)并重复冻结phase验证不新增步数。总上限2640；小warmup仅工程验证，不是G1候选。与前述两组预算合计上限12640控制步。

- 学习计时smoke已暴露并保留一个额外缺陷：2000步适应确有3次actor更新，但随后冻结地形/平地的action_residual_gain均为0，实际执行base。这两段数据只验证时延与零更新，不是学习后策略评估。
- 修正managed frozen candidate执行增益：使用配置已绑定的adapting_residual_gain（当前0.5）；online等待态仍用原residual_gain=0，base模式仍为0。常规bare worker保持显式residual_gain语义，不跨旧source-bound session修改配置。
- 追加`frozen_gain/`独立回归：相同开发地形/learner，B=1000、trial_steps=1000、warmup=256/critic-only1000，沿用learning_options；冻结地形1×320、平地1×320，重复phase验证不追加试验。总上限1640，累计本组预登记上限14280。要求实际actor更新后冻结阶段action_residual_gain=0.5、mean/request/executed residual非零、零更新/零replay写入，并保留安全阈值。该短回归仍不作G1任务表现晋级结论。

- 本组完成：baseline706＋优化冻结5416＋学习/恢复2640＋冻结gain修复1640＝10402控制步，uncertain=0。修复后9696步零控制超时，最大10.280 ms，阈值仍20 ms。两个1000步base平地窗口有效；学习/恢复确有3次actor更新且不重复1000次预热。冻结gain修复后两个320步窗口gain=.5，执行残差非零、actor张量不变、零更新/零replay写入，重复phase新增0步。
- 原学习组的零gain冻结结果已明确降格为base执行证据，未重写原始记录。全量测试1602 passed, 7 skipped，源码版本重建/账本/动作等价验证通过。报告：[G1.2](lite3-g1-control-latency-2026-09-29.md)，[验证JSON](lite3_g1_latency_validation_20260929.json)。下一步进入完整回合留出价值诊断，尚不宣布G1路线晋级。

## G1-20260929-complete-value

- 预登记完整回合价值诊断：[固定协议](lite3_g1_value_protocol_20260929.json)。development rough_4cm 101/107，分别固定6个perturbed-v1起点，learner seed7，前4回合拟合、后2回合验证；冻结确定性base、无动作噪声、无actor/critic学习采集。terrain101/107由先前G0失败/混合诊断选定，不以本轮结果挑选。
- 每回合任务deadline2000，collector上限同为2000；两地形总上限24000实际控制transition。按trial编号预定划分，不补抽失败；完整task终止（包括失败/deadline）纳入MC，collector/physical截断拒绝作为完整MC并保留成本。最多GPU0/1各一个worker，不生成sealed种子。
- 使用LoRA冻结入口已有48维critic观测（45维policy＋body XY velocity＋绝对elapsed/2000），轨迹float32；网络仍48-128-128-1 ELU。与在线入口共用工厂，未改网络容量或actor信息权限。
- 每地形只拟合一次：100 epochs、batch128、Adam lr5e-5、value loss=.25*MSE、grad cap1、shuffle seed11；原始输入/目标不归一化。起始critic来自该地形trial00000的未训练checkpoint。只保存固定最终epoch；训练曲线不包含验证集。
- 常数基线为训练样本均值；时间基线为[1,t,t²,t³]训练集最小二乘。最终才评估2个留出回合；报告pooled和按回合等权指标、target variance、RMSE/NRMSE、bias、EV/R²，以及pre-action进度三分段和非足接触阶段误差。低方差EV/NRMSE标null。
- 输出`artifacts/g1_complete_value_20260929/terrain-{101,107}/`，诊断`analysis/`。此为reward-only完整MC，不是SAC soft-Q真值，离线critic不导入在线actor或候选session。发现数据/数值契约问题先修复；不根据这2个留出回合增加训练轮数或调整超参。

- 完成：101为10285步、107为8930步，共19215/24000实际控制步；12完整回合，无替换，uncertain=0；重复evaluation新增0步。每地形只拟合一次100 epochs（5400/5300 minibatch更新，CPU约10.2/9.1秒）。101/107留出EV为−0.392/0.342，RMSE为63.581/37.341；107时间基线RMSE24.583更低。两验证集均无正非足接触标签，对应分层标为无样本。
- 增强数据入口的冻结critic优化器、有限观测、二值接触标签检查，显式bool掩码防止整数标签误作行索引；重新审核原12回合通过，没有重拟合或增加物理成本。全量测试1611 passed, 7 skipped。报告：[G1.3](lite3-g1-complete-value-2026-09-29.md)，[验证JSON](lite3_g1_value_validation_20260929.json)。G1学习信号尚未合格；速度历史仅为后续待注册假设，本组未运行第二候选或新确认实验。

## G1-20260929-value-history-confirmation

- 预登记单一历史特征候选：[固定协议](lite3_g1_value_history_protocol_20260929.json)。仅新增两个排除当前时刻的 body XY velocity 积分特征：`sum(v[j], j<t)*.02/12`，每回合清零；12 m来自0.3 m/s×2000×.02秒，不从数据拟合。它是旋转机体系下的行程代理，不是世界XY里程；禁止添加simulator位置/朝向真值，actor输入和在线学习实现不变。
- 使用原101/107各四个训练回合，保持100 epochs、batch128、Adam lr5e-5、loss=.25*MSE、grad cap1、shuffle seed11；48→50维，隐藏层128/128 ELU不变，初始第一层追加零列，其他初始权重相同。只拟合一次；先保存并哈希固定模型，再启动新确认采集。旧验证轨迹不用于拟合/调参。
- 新确认使用development rough_4cm101/107、learner seed19，各2个预定perturbed-v1起点（协议锁定reset seeds），冻结确定性base，无噪声/学习，完整deadline2000；新增总预算≤8000控制步，GPU0/1各最多一个worker。输出`artifacts/g1_value_history_20260929/`；所有失败、截断和成本保留，不补起点，不生成sealed种子，不跨源码续接旧session。
- 原保存48维critic、原训练均值常数、原训练三次时间基线均锁定为比较对象。候选只有在两地形pooled EV均>0，且相对全部三个比较对象的pooled RMSE及回合等权RMSE都降低，才记为“该次诊断支持”；否则不支持，不依据确认结果再调参数。任何不完整/aborted回合使确认不充分，禁止将其重标为MC。该判定仍不等于在线G1路线晋级。

- 确认采集前已完成两次固定拟合：101/107各5400/5300 minibatch更新，CPU约10.51/8.97秒（与回归测试同时运行）；初始输出max_abs均0，训练EV为0.821683/0.818298。未据训练曲线改预算。确认目录尚不存在时保存[模型锁定记录](lite3_g1_value_history_fit_lock_20260929.json)，fit manifest SHA256=`9a45cf46ebd3f89bd3048e4df8bcf4f47eaf82e6f0f4ece7281be5642fb08a56`；确认必须校验该哈希。全量回归1623 passed, 7 skipped。

- 采集前协议/源码/模型锁定已提交推送`b2f0048`。确认完成：101两个task_timeout各2000步；107两个physical_failure为1037/1119步；共6156/8000新增控制步，uncertain=0，无替换。重复evaluation均新增0试验/0步。连同G1.3累计价值诊断25371步；未启动在线学习、平地回合或额外拟合。
- 固定模型比较：101原48维/历史50维RMSE为80.386/79.709，107为38.789/39.248，107时间基线27.535。101单地形满足方向条件，107不满足，整体预登记判定为“不支持”，不部署、不调参补救。新确认仅4回合，不作统计显著性或路线晋级结论。报告：[G1.4](lite3-g1-value-history-2026-09-29.md)，[验证JSON](lite3_g1_value_history_validation_20260929.json)。后续优先独立SAC soft-Q诊断。

## G1-20260929-sac-fixed-soft-diagnostic

- 预登记零新增物理步/零更新离线审核：[协议](lite3_g1_sac_soft_protocol_20260929.json)。读取旧SAC预热session的trial00009（预热3881物理步、0 critic更新）及trial00010（1000预热＋43常规critic更新）检查点。后者账本仍为aborted，不恢复或晋升，不将其改称恰好1000步更新后的状态。
- 使用before replay全部3871有效行，要求after完全保留相同前缀；actor/alpha逐位相同且优化器为空。冻结策略和温度，统一使用注册适应gain=.5（after保存的瞬时等待gain=0，不能混用为不同目标策略）。每行16个随机动作，CPU batch128、seed23，使用同一随机数比较前后target critic。
- 目标为`r + mask*discount*(min(targetQ)-alpha*log_prob)`，n=1/gamma=.99；熵在normalized tanh action坐标，实际请求为gain×action_limits×tanh动作。报告soft Bellman残差、双Q分歧、目标方差、动作采样波动、有效熵贡献及随机请求相对base零残差的Q变化；不与reward-only MC作同尺度真值比较。
- 元数据检查已发现这两个replay没有terminated/truncated行、before请求残差全0。诊断将明确报告覆盖缺口；不能把训练内自举一致性解释为真实soft-Q校准或新策略收益。预算仅两固定快照×16次全样本target采样，无新拟合、仿真、参数搜索或sealed数据。

- 首轮完成：target critic目标RMSE由1.603419降到0.120417，当前minQ均值由−0.005857升到8.597847。但这只衡量对慢更新目标网络的拟合。追加[固定当前Q算子检查](lite3_g1_sac_soft_self_protocol_20260929.json)：同一快照、同一3871行、相同16次随机动作及所有参数，仅用冻结的当前critic替换bootstrap target critic。再计两快照×16次全样本前向；仍0新增控制步/0更新，禁止据此挑选模型或调整训练轮数。原报告不改写。

- 两轮诊断完成：after当前Q算子RMSE仍1.514500、偏置−1.496550；不能将target网络拟合误差0.120417当作Q收敛。两快照/算子共247744次随机动作前向，0新增物理步/0更新。旧11个原始末尾均有task terminal，replay均未保留；前10个可读summary是safety_failure，属于时延修复前数据，不以此直接放宽安全mask。原trial00010保持aborted。
- 新诊断工具及最终回归1629 passed, 7 skipped；测试夹具共享边界张量问题已修正，未改在线目标/网络/安全语义。报告：[G1.5](lite3-g1-sac-soft-diagnostic-2026-09-29.md)，[验证JSON](lite3_g1_sac_soft_validation_20260929.json)。下一步先做时延修复后的SAC完整任务采样与覆盖核验，尚不晋级。

## G1-20260929-sac-boundary-collection

- 预登记时延修复后的SAC完整任务采样：[协议](lite3_g1_sac_boundary_protocol_20260929.json)。development rough_4cm101/107、learner seed31、perturbed-v1，每地形最多2个预定起点×2000步，deadline2000，合计≤8000真实控制步；最多GPU0/1各一worker，不补起点，不生成sealed种子。
- 使用冻结确定性base的warmup路径以验证实际replay准入。每地形managed B=4001、warmup=4001，而固定两回合最多4000步，确保本组不会启动critic预热或actor/alpha学习。replay5000/min256、n=1、alpha=.01，其余默认。多出的1步只用于防止采集末尾触发学习，不执行；未用适应预算不追加试验。
- 逐步日志新增critic/actor实际replay写入数和安全准入mask，便于与轨迹及checkpoint逐行核对。两次attempt后通过账本API锁定evaluation，0控制步，不执行evaluation/retention；重复锁定不新增成本。采集窗口结束不代表预热或完整适应session合格。
- 核验自然任务终止/截断与安全mask、base动作等价、零优化器状态、跨checkpoint累积replay、终止bootstrap。保留无法学习的安全失败样本在原始trace中，不直接放宽准入。输出`artifacts/g1_sac_boundary_20260929/`。

- 完成：101为lateral_violation/1833及task_timeout/2000，107为physical_failure/1404及physical_failure/1264，共6501/8000步，uncertain=0，无补抽。全部6501行进入replay，4个终止行均保留、bootstrap=0；15字段按存储dtype逐位对齐，零优化器状态/零更新，base动作等价。控制超时0，最大7.614 ms。
- 两session已通过账本锁定evaluation，重复带阶段检查的锁定新增0步；未执行评估/保持回合，未用适应预算168/1333步不追加。本组未完成预热、不作策略或Q合格结论。新增日志和对账工具，全量回归1630 passed, 7 skipped。报告：[G1.6](lite3-g1-sac-boundaries-2026-09-29.md)，[验证JSON](lite3_g1_sac_boundary_validation_20260929.json)。自然终止覆盖已验证；非零请求覆盖仍待后续登记。

## G1-20260929-sac-success-guard

- 预登记零物理步数值复现：真实SAC网络（49维frame、25历史、12动作）、初始化seed7，128行合成batch，127个ordinary reward1.5和1个合法success reward305/mask0；其余mask1/gamma.99。固定alpha=.01、gain=.5、max loss800/mean TD10/grad norm3000，actor冻结，每种设置仅尝试一次critic更新，模型丢弃。
- 比较原MSE与已有Huber(delta=1)单一损失改动；使用完全相同初始权重和seed23随机动作。检查是否因单个合法成功奖励触发loss守卫，同时验证actor/alpha不变。该检查不是策略收益证据，不修改奖励、终止bootstrap、TD/梯度守卫或增加训练预算。
- 若MSE确实拒绝而Huber通过，则将Huber作为后续新source-bound SAC小预算候选登记，保持其他学习配置不变；不自动扩展为LN/UTD联合搜索。

- 复现完成：MSE loss1458.2755触发critic_loss_limit；相同目标和初始化的Huber1 loss6.8721通过，mean TD均3.9484，actor/alpha不变。两次合成更新attempt、一次接受（模型丢弃），0物理步；首次辅助脚本因无用helper导入失败在更新前退出，0步/0更新。原结果存`artifacts/g1_sac_success_guard_20260929/report.json`。Huber降低大TD样本的梯度权重，不把该数值通过当作价值或策略性能改进。

## G1-20260929-sac-huber-coverage

- [预登记协议](lite3_g1_sac_huber_protocol_20260929.json)：SAC单一Huber1候选，development101/107、learner43、perturbed-v1，各独立B5000（4000冻结base采样＋最多1000后续学习），恰好1000 critic-only预热；保留n=1、现有网络、alpha=.01、更新额度.5/.005及原安全/TD/梯度阈值。
- 每地形固定1×2000任务冻结评估、1×1000平地保持；另起fresh base session执行相同phase/reset seeds的1×2000＋1×1000对照。每地形上限11000，总≤22000控制步；最多两GPU、每卡一worker。原始诊断数据不作为免费适应样本，不做LN/UTD/n-step联合改动。
- managed入口允许已有critic-huber-delta参数并纳入source/config哈希；预热逐更新日志补记当时residual_gain。无探针，候选固定最后提交checkpoint，冻结phase不能回流适应，重复phase调用不得新增试验。失败/aborted及成本全部保留，不按表现替换评估起点；预热失败不晋升其aborted checkpoint，本组不追加救援候选。
- 观察真实非零请求、探索、接受更新、冻结动作增益、零更新/零replay写入和完整任务/平地指标。本组只有单适应seed及单评估起点，工程/覆盖诊断不等于G1路线晋级或显著性结论。

- 本组物理阶段已完成，随后新日志暴露预热目标增益依赖瞬时控制状态：101全部1000次为.5，107全部1000次为0。两个session虽共享adapting_residual_gain=.5，却拟合了不同增益的目标策略；该数据保留为修复前证据，不作为一致目标策略下的Huber跨地形效果确认。
- 先完成8次冻结phase重复调用，均新增0步/0试验，再修复预热增益：critic-only更新显式接收注册适应增益，避免读控制等待态；不修改control gain，不改变常规actor更新行为。该修复仅做真实agent集成/单元回归，新增物理步0，不在本组追加救援候选或跨源码续接旧session。

- 本组实际19129/22000控制步、uncertain=0，四session均完成。每地形5000适应步、1000critic-only预热＋500常规critic更新；非零随机请求入库101为361、107为578行。actor更新0/2次；101冻结等于base，二者completion1483步；107候选/base均physical_failure（1106/1057步，前进5.08205/5.12575 m）。四平地窗口均1000步且可比较；单起点结果不作晋级或显著性结论。
- 一次控制超时保留：101适应trial00001第951步，回退前34.319 ms、线程CPU5.025 ms、GC0；回退后总值44.072 ms。未放宽20 ms阈值，未替换该失败。冻结actor张量不变、零更新/写入和constant replay size验证通过；初版对账错误要求冻结完整checkpoint，修正读取契约后通过，失败记录保留，新增物理步0。
- 最终回归1636 passed, 7 skipped。报告：[G1.7](lite3-g1-sac-huber-2026-09-29.md)，[验证JSON](lite3_g1_sac_huber_validation_20260929.json)。预热gain修复尚无新物理验证，本轮不追加救援候选；下一步登记新版本复核，再评估有效actor更新调度。

## G1-20260930-sac-warmup-gain-physical

- 预登记修复后工程复核：[协议](lite3_g1_sac_warmup_gain_protocol_20260930.json)。新source-bound SAC session，development101/107、learner43、perturbed-v1；每地形B512、warmup256、trial256、task deadline2000，先执行一个trial检查checkpoint，再恢复剩余预算。随后各1×64冻结检查并重复phase调用，不执行retention。总上限1152真实控制步，不补抽失败起点。
- 使用Huber1、alpha=.01、replay5000/min256、1000 critic-only更新；novelty calibration1024使本组适应控制保持等待gain0。目标预热gain仍注册适应值.5。缩短warmup/延长calibration只为隔离工程条件，不用作正式4000步预热或actor收益候选。
- 预热日志同时记录目标gain和当时control gain；验证1000次更新均目标.5/控制0、actor/alpha不变，跨checkpoint不重复预热，冻结零更新/零replay写入，重复phase新增0步。最多GPU0/1各一worker，原20ms安全阈值不变，所有失败和实际成本保留。

- 同轮只读核对G1.7已有有效actor样本和实际更新数：使用原manifest更新额度及逐步写入记录，计算每物理回合能获得的更新次数上界，并核验结束时剩余额度。不拟合新模型、不改调度、不使用冻结评估结果选参数；新增物理步/优化器更新均0。

- 完成：实际1152步、uncertain=0，两地形首256步各完成1000次预热更新，全部target gain=.5/control gain=0；检查点恢复各新增256物理步及128次普通critic更新，warmup计数不增加。共2256次critic更新、0 actor/alpha更新。各64步冻结检查零更新/写入、actor张量不变，重复phase新增0步。控制超时0，最大7.991 ms，原20ms阈值不变。
- 原actor额度核对：101有效样本分为65/140，分回合额度floor均0；107有效样本579，上界2；实际0/2与上界一致。没有改变调度或提高学习率。报告：[G1.8](lite3-g1-warmup-gain-2026-09-30.md)，[验证JSON](lite3_g1_sac_warmup_gain_validation_20260930.json)。全量1636 passed, 7 skipped，短工程窗口不等于正式4000步预热或G1路线通过。

## G1-20260930-sac-actor-schedule

- 重启恢复检查确认HEAD=e3dc2f6、工作区干净，无未完成仿真进程；不重跑G1.8。预登记[actor额度单因素对照](lite3_g1_sac_schedule_protocol_20260930.json)：development101/107、learner53，reference=.005与actor025=.25；critic额度保持.5。Huber1、预热目标gain=.5、初始alpha=.01、n=1、网络/lr/安全条件相同。暂不同时改critic UTD=1；同样有效行数下名义actor/critic额度比从1:100变为1:2。
- 每臂/地形从base独立采4000步并做1000 critic-only预热，B5000；随后固定1×2000冻结任务、1×1000真实平地。四session总上限32000控制步，预热数据不共享或免费导入。101先reference，107先actor025，固定顺序不由结果决定；每地形固定GPU，最多两worker。
- Huber候选作为保留的第一配置，actor额度修订作为第二配置；本组不追加第三配置或救援。取最后提交checkpoint，不挑最好epoch/起点；所有失败和不确定成本保留，aborted checkpoint不晋升，不替换评估起点。输出`artifacts/g1_sac_actor_schedule_20260930/`。
- 描述接受更新数、非零请求覆盖、任务完成/前进/跌倒和平地有效性/MAE，并核验配对reset初态、冻结零更新/写入及重复phase不增加步数。一个适应seed、一个评估起点不构成G1晋级或显著性证据。

- 完成：实际30509/32000步（适应20000、任务6509、平地4000），uncertain=0，四session全部结束，无替换。两臂预热root/action/reward/terminal字段逐位一致，配置仅actor额度不同，冻结初态匹配；每session1000预热＋500常规critic更新，全部预热gain=.5。
- reference/actor025接受actor更新：101为3/176，107为2/182。reference任务均completion（1850/1755步）；actor025在101 lateral_violation/1333步、107 physical_failure/1571步，进度6.87035/5.89782 m，均低于reference约8m。不采用.25、不追加第三候选，不按结果改选checkpoint。平地窗口均有效，但107高额度X误差增大，不能仅用平均XY误差降低宣称整体改善。
- 控制超时0，最大11.718 ms；冻结actor张量不变、零更新/写入；8次冻结重复调用及2次完成驱动resume新增0步/0试验。全量1639 passed, 7 skipped。报告：[G1.9](lite3-g1-actor-schedule-2026-09-30.md)，[验证JSON](lite3_g1_sac_schedule_validation_20260930.json)。下一步先复盘既有Q/回退，不认定路线通过。

## G1-20261002-schedule-q-readonly

- 预登记[只读协议](lite3_g1_schedule_q_protocol_20261002.json)：锁定G1.9四个最终适应checkpoint及日志SHA256，不挑中间版本。新增物理步/优化器更新均0，CPU单线程；每行16次动作采样、seed23、batch128、gain .5。
- 验证每地形两臂前4000行所有TransitionBatch字段完全一致；首次满库cursor0允许，超过首次填充或字段不同则停止。全库及共同前缀、各自后续、非零请求、终止/成功/失败分层；分别计算lagged target和online Q算子残差、动作敏感性。解析回退事件、累计更新及实际恢复版本，保留原始逐行结果和成本。
- 固定每个checkpoint自己的actor/alpha，不要求不同臂相同；只诊断训练分布自洽性，不称真实Q校准，不新增第三候选，不凭本次指标选择训练参数。

- 完成：CPU单线程50.493990秒墙钟/50.483344秒进程CPU，新增物理步/优化器更新0，前4000行所有字段配对一致、输入哈希不变。发现成功终止Q约6–10而目标305，107高额度目标/当前Q动作差分异号；这些是训练回放诊断，不能确证冻结失败因果。性能回退1/2次（各地形reference/actor025）；fall_reset单独保留不计权重回退。报告：[G1.10](lite3-g1-schedule-q-2026-10-02.md)。诊断测试6 passed，不采用.25、不新增训练、不晋级。

## G1-20261002-q-report-refinement

- 预登记[已有行结果再汇总](lite3_g1_q_refinement_protocol_20261002.json)：锁定G1.10 report及8份npz哈希，零物理步、零优化器更新、零网络前向。修复事件字段把fall_reset混入rollbacks的歧义；全库/共同前缀/各自后续分别给出终止样本平方误差占比及非终止RMSE。保留旧报告不覆盖，不把事后分层称为独立校准，不挑新候选。
