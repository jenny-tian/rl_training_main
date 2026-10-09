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

- 完成：仅保存行再汇总，墙钟0.042507秒/CPU0.042495秒，输入哈希未变。101两臂和107 reference的终止行贡献>99.5%全库lagged平方误差；共同前缀非终止lagged RMSE约.20–.26、current约1.33–1.39。修正control_events命名并分类fall reset，旧结果保留；12 tests passed。详见[补充报告](lite3-g1-q-refinement-2026-10-02.md)，不新增训练、不作独立校准结论。

## G1-20261002-learning-replication

- 用户明确要求持续改进直至学习出可稳定通过崎岖地形的步态。预登记[独立学习复现](lite3_learning_replication_protocol_20261002.json)：保留现有Huber1/.005候选，101/107×learner61/67，各独立B5000含4000预热、1000critic-only更新，最终checkpoint固定5×2000任务复测、2×1000平地；各配fresh base同起点复测。总上限116000步，GPU0/1各一worker，所有失败及成本保留。
- 不追加G1.9第三配置，不导入旧replay或挑旧权重。先检验真实学习增益与复现性，非加长训练掩盖Q问题。工程晋级沿用主计划；稳定开发要求每地形≥9/10且每session≥4/5、无评估安全违规、学习增益及平地保持同时成立。平地阈值及步态记录规则见协议；最终目标还需要步态证据和新开发地形确认，不能以本轮单独宣称普遍稳定。
- 101先candidate、107先base；各seed61再67。无选择探针、不替换失败起点、不晋升aborted checkpoint。后续如需新配置，须本轮复盘后单独登记，旧候选负结果保留。

- 中期事实（非最终晋级判断）：seed61四session完成并通过逐回合审计，实际49161控制步、uncertain0、timeout0；101候选/base完成1/5与2/5，107为1/5与0/5。相同初态、冻结actor不变/零更新/零回放写入验证通过，两地形平地保持通过。第二seed67仍按注册运行。摘要：[seed61审计](lite3_learning_replication_seed61_audit_20261002.json)，完整原始审计在artifacts/g1_learning_replication_20261002/audit-seed61/。不替换失败起点，不宣称新步态或稳定成功。

- 完成：双seed四学习＋四base session实际95403/116000步，uncertain0，无替换。101学习/base累计1/10与2/10；107为1/10与0/10，未通过≥20百分点双seed正向学习门槛或≥9/10稳定门槛。全部平地保持比较通过，冻结零更新/写入、actor不变和配对起点检查通过。详见[完整报告](lite3-learning-replication-2026-10-02.md)及[验证](lite3_learning_replication_validation_20261002.json)。不追加该配置长训练，转入独立登记C0工程验证。

## C0-20261002-nominal-kinematics

- 独立搜索分支预登记[USD名义运动学审核](lite3_c0_kinematic_protocol_20261002.json)：读取实际部署资产的关节轴、父子锚点与局部旋转，零基础动作姿态，既有gain.5和限幅，代码零点及前/后列±1。零仿真/零优化器更新，保存资产哈希与全部位移结果，不据本次结果调基；只能证明名义关节目标运动学，不能声称摆动相位或闭环抬脚/地形成功。

- 完成：实际USD关节图齐全，既有gain/限幅下正代码对应前或后足端名义上移1.740584 mm，零代码精确零；资产哈希未变，零物理步/优化器更新。该幅度不直接支持4cm地形能力，未据此调基。详见[工程报告](lite3-episode-search-engineering-2026-10-02.md)和[结果](lite3_c0_kinematic_validation_20261002.json)；相关22 tests passed。

## C0-20261002-fixed-code-physical

- 预登记[冻结代码工程验证](lite3_c0_physical_protocol_20261002.json)：必须等待当前双seed SAC复现完整审计文件完成后才能启动。101/107、seed71，base/zero/前后腿±1共6臂，各64步任务窗口＋64步平面窗口，总上限1536步；GPU0/1每卡一worker。新source-bound冻结session，无导入learner/replay、零优化器更新，维持既有限幅、gain和安全过滤。
- 核验零代码物理等价、非零请求公式、冻结零更新/写入及重复调用不新增试验；保存全部失败和成本，不替换起点。64步窗口不作完整任务、平地保持、headroom或学习效果结论；外层归类工程成本，不能把child ledger的evaluation名称冒充独立学习评测。

- 完成：24短窗口全部完成，1536/1536步，uncertain0，无跌倒/安全违规/控制超时；base/zero轨迹比较字段逐位相同，非零请求公式误差≤7.45058e-9，冻结actor不变、零更新/写入。24次phase重复调用所有session_summary完全不变，新增步/试验0。见[报告](lite3-c0-physical-check-2026-10-02.md)及[验证](lite3_c0_physical_validation_20261002.json)。仅工程通过，下一步另登记完整回合headroom。

## C0-20261002-grid-headroom

- 预登记[完整回合headroom](lite3_c0_headroom_protocol_20261002.json)：已通过1536步工程验证；101/107、固定二维{-1,0,1}网格9候选各1×2000，seed83相同起点；按完成/相对进度/较少安全干预/固定序号排序，保存全部结果及哈希后锁定前2（零代码可入选）。随后seed89新起点上base、rank1、rank2各3×2000。每地形≤36000，总≤72000新控制步；GPU0/1各一worker，无回合内优化器更新。
- 所有阶段为付费开发筛选成本，child evaluation名称不等于最终评测；本组不运行retention，child正数占位预算不允许实际使用。失败及不确定成本保留，不替换起点、不挑新候选；不完整/collector cutoff不入选，少于2个有效结果不进入验证。锁定code不因验证结果改变。
- headroom规则见协议：三新起点对照不得增加跌倒，且额外完成/其他起点进度不降，或进度中位数≥base1.1倍。通过只支持另登记一次独立确认，不宣称稳定步态或路线晋级；不增加幅值/维度、不使用teacher或额外感知。

- 完成：52346/72000步，uncertain0；两地形前两名在seed89各3新起点验证。101 rank1/2均2/3 vs base0/3，跌倒1→1，中位数进度比1.83884/1.84088；通过进度分支但各存在单起点退步。107 rank1为2/3 vs0/3、跌倒3→0、进度比1.53845，rank2零代码不算改进。原锁定候选未换，完整审计通过。见[报告](lite3-c0-headroom-2026-10-02.md)。下一步对原网格排名最高且通过的非零候选另登记独立确认及完整平地保持，不能宣称稳定步态。

## C0-20261002-independent-confirmation

- 预登记[独立确认和平地保持](lite3_c0_confirmation_protocol_20261002.json)：根据完整headroom审计，固定各地形原网格排名最高且通过的非零候选，101[-1,-1]、107[1,0]。新seed97，候选与base各3×2000任务＋2×1000平面；101候选先、107base先，总上限32000新控制步，GPU0/1各一worker。
- 原计划任务确认每地形12000步，明确另加每地形4000步配对完整平地保持，禁止以短窗口代替保持验收。按新三起点单独检验原headroom方向及已锁定平地阈值，不合并旧验证数据挽救失败。代码不更新、不替换起点、不换候选，失败及不确定成本全保留；通过也不等于9/10稳定或新步态证明。

- 完成：25415/32000步、uncertain0；seed97候选两地形均1/3、base均0/3。101中位数进度比1.37909、任务和平地保持通过；107比1.00392且一个非成功起点进度低于base，任务确认失败（保持通过）。未合并旧数据挽救失败。见[报告](lite3-c0-confirmation-2026-10-02.md)及[验证](lite3_c0_confirmation_validation_20261002.json)。不调整固定代码迎合本批结果；下一组另登记从零开始的回合级学习，107失败保留，稳定目标未完成。

## C0-20261002-episode-learning

- 预登记[从零回合级参数学习](lite3_c0_episode_learning_protocol_20261002.json)：101已独立确认有限headroom，107固定代码确认失败保留但允许探索性学习对照。不导入旧code/replay/checkpoint。两维初始mean0/std.3，每批4候选（首批含0），5批，完整回合前2精英更新均值/标准差、std下限.1；同批起点一致，批seed=learner+100000*(批次+1)。每session最多40000训练步。
- learner137先行，149需首seed完整审计满足每地形≥4/5、平地/安全通过及至少一地形≥20百分点收益后才运行。最终均值固定后，每地形/seed候选与base各5×2000任务、2×1000平面。每地形/seed≤64000；首seed≤128000，两seed总≤256000步。最多GPU0/1每卡一worker。
- 所有回合计费、失败保留，基础设施/数值失败停止受影响session不晋升最终code、不重试替换。child frozen evaluation在搜索期全部归类训练成本；不冒充SAC梯度学习。最后均值不挑最好epoch；最终9/10稳定及独立学习/新地形/步态证据仍需验证，不凭一个seed完成目标。

- 首seed137完成：101025/128000步（训练59885、任务33140、平地8000），uncertain0；两地形各5次分布更新均由原始回合重算一致。最终101候选/base1/5对1/5，107为0/5对0/5，保持均通过。未达每地形≥4/5及学习增益门槛，按预登记不启动149。详见[报告](lite3-c0-episode-learning-2026-10-02.md)及[验证](lite3_c0_episode_learning_validation_20261002.json)。新神经策略方法独立准备，旧失败和预算保留。

## Policy-20261002-environment-smoke

- 预登记[新神经策略环境工程实测](lite3_policy_environment_smoke_protocol_20261002.json)：8环境、64正常向量步＋1明确人工截止步，总520控制转移，单GPU，零优化器更新。混合2平地列＋各3列rough101/107，固定difficulty1/amplitude.04；不宣称与旧随机difficulty mesh逐位相同。保持本体actor45/critic48，无高度扫描输入或相位参考奖励。
- 新训练奖励与旧有限任务评测奖励区分；动作网络从旧base复制，critic/optimizer不导入。核验观察和动作字段、冻结权重不变、先计费后step以及有限时限不bootstrap；人工截止案例单列，不作为学习/通过证据。所有失败与不确定批成本保留。

- attempt01在动作前记录配置时失败：当前ObservationManager无get_term_cfg，charged/confirmed/uncertain均0。保留原report/log，改为读取实际传入的cfg.observations.policy项；在相同总520步预算内新建attempt02复核，不覆盖失败。运行退出码不能代替持久report状态。

- attempt02通过：8×65=520控制转移，全部确认、uncertain0；正常终止0，人工截止8/8且无timeout bootstrap；actor45/critic48、均值权重/critic不变、optimizer为空、base哈希不变。attempt01成本0保留，两attempt总520。见[报告](lite3-policy-environment-smoke-2026-10-02.md)及[验证](lite3_policy_environment_smoke_validation_20261002.json)。只完成环境工程，PPO训练和稳定步态仍待登记/执行。

## Policy-20261002-full-actor-pilot

- 预登记[状态相关完整动作网络微调](lite3_full_policy_pilot_protocol_20261002.json)：通过520转移环境工程检查后，新seed223、128环境、rollout32、128次PPO更新，固定lr1e-4/5epochs/4minibatches、std.2、gamma.99/lam.95。训练上限524288控制转移，逐向量step按128条先计费后确认；原actor初始化，critic/optimizer新建，不导入旧replay或teacher。
- 这是完整actor参数学习，动作不受旧残差幅度界约束；保留实际关节scale/绝对clip及崎岖姿态/接触终止，单独报告不同方法及预算，不能声称等资源胜过旧在线方案。训练混合25%平地和两种崎岖tile，固定difficulty1；无相位参考或高度输入。
- 只取完成128次更新后的final checkpoint，保留中间快照但不据测试挑选。随后候选和原base在101/107、seed229相同新起点各5×2000任务＋2×1000平地；最多48000额外控制步，总≤572288。失败与成本全部保留、不自动重试、不晋升aborted状态、不替换评测起点。一个pilot不能完成稳定/多独立学习/新步态验收。

- 首pilot训练数值完成128次更新/524288转移，但轨迹契约审计发现1859行请求触及关节clip，其中1726个非终止后继仍记录裁剪前动作；与原冻结执行链裁剪后上一动作不一致。原模型/轨迹/费用保留，不晋升其final，不运行原48000评估预算。见[契约审计](lite3_policy_history_mismatch_20261002.json)。

## Policy-20261002-full-actor-historyfix

- 单独登记[动作历史修正后的新pilot](lite3_full_policy_historyfix_protocol_20261002.json)：同seed/网络/优化器超参/训练预算，重新从原base初始化，绝不导入前一pilot结果。仅改变env.step实际接收动作先经过共同clip，PPO保存的原始请求/logprob不改；每步断言后继actor/critic的上一动作等于已发送命令（reset行0）。新训练≤524288＋评估≤48000，前一pilot524288成本额外保留，总账不混淆。

## 2026-10-02 — Full-policy symmetry pilot (preregistered before execution)

Protocol: `docs/lite3_full_policy_symmetry_protocol_20261002.json`. Seed233, original-base actor warmstart, fresh optimizer/critic, fixed512 PPO updates =2,097,152 controls; frozen seed239 evaluation cap48,000. Motivated by corrected-pilot lateral drift and degraded flat tracking. Add sagittal mirror policy loss (coefficient1, no PPO data augmentation), keep physics/rewards/action history unchanged, increase training duration. Compound engineering experiment; no causal symmetry-only claim. Final checkpoint only. Preserve prior costs and failures. Require offline reflection and actual RSL update checks before launch; wait for current source-bound evaluation to finish before runtime edits.

## 2026-10-02 — Route-observation engineering smoke (preregistered)

`docs/lite3_policy_route_smoke_protocol_20261002.json`: vector smoke8x65=520 controls, including one explicit deadline step; then frozen expanded-candidate/original-base64 controls each on terrain101/seed251, total648 maximum. Actor48/critic51 with simulator route localization, original-base mean expansion via zero columns, no optimization. Preserve every failure and uncertain charge; no automatic retries. This validates implementation only, not gait or route qualification. No route learning pilot is launched before the smoke gate passes and its separate protocol is registered.

## 2026-10-02 — Route frozen-smoke CLI correction

`docs/lite3_policy_route_frozen_fix_protocol_20261002.json`: both initial commands were rejected before session creation because retention-trials0 is invalid;0 controls. Preserve logs and failed manifest. Set the required retention count to1 but execute evaluation phase only. New frozen-v2 directories, same seed251/terrain101 and64 steps per arm; new cap128 replaces the unused initial128. Vector smoke remains complete520.

## 2026-10-02 — Route-observation full-policy pilot (preregistered)

`docs/lite3_full_policy_route_protocol_20261002.json`: seed257,512 PPO updates=2,097,152 controls; frozen seed263 evaluation cap48,000. Fresh original-base actor initialization with zero route-input columns, fresh critic/optimizer. Actor48/critic51, explicit simulator localization; fixed commands and task criteria retained. Mirror loss1, tracking std.15, heading cost-1. Compound method and extra sensing declared. Final checkpoint only. Vector520 and frozen128 engineering checks passed with report hashes locked; preserve the initial0-control CLI failure. Prior full-policy training/evaluation3,209,837 controls and engineering costs remain separate. No route or gait qualification before frozen tests and independent replication.

## 2026-10-02 — Independent route-policy learning replication (preregistered)

`docs/lite3_full_policy_route_replication_protocol_20261002.json`: first pilot passed5/5 versus base0/5 on both terrains and paired flat retention. Gate report hash locked. New learning seed269 and frozen seed271; same512-update budget2,097,152 controls and evaluation cap48,000. Start again from original base with zero route-input columns, fresh critic/optimizer; no learned-state import. Final checkpoint only. Require repeated gains, retention and aggregate>=9/10 per terrain across two independent policies (each>=4/5); new-development terrain and gait evidence remain required. Preserve prior5,343,851 training/evaluation controls plus1,168 engineering controls.

## 2026-10-02 — New-development terrain membership registration

`docs/lite3_route_new_development_seeds_20261002.json` fixes four new terrain seeds via SHA256(namespace, original route protocol hash, indexed terrain counter), excluding all existing artifact-manifest terrain seeds and current reserved/development membership. No geometry was generated or performance observed. Seeds:834818272,1490653948,1132853762,874689437; evaluation masters1848426001/1315124702. Conditional scope: both independent final route policies and same-start original base on every terrain;5 task and2 flat trials per arm/run/terrain, cap192,000 controls. Requires successful independent replication and a locked executable protocol first. Legacy development allow-list extension must wait until active source-bound jobs finish. No selection of favorable terrain or replacement starts.

## 2026-10-02 — Flat-action anchor engineering smoke (preregistered)

`docs/lite3_policy_flat_anchor_smoke_protocol_20261002.json`: seed277,8x65=520 controls, no updates. Frozen original actor targets only the two flat columns; reward weight-2 times mean action MSE. At registered vector step3 add+.1 normalized HipX command to exercise nonzero flat cost and exactlyzero rough cost; final step explicitly forces deadline. Verify pre-action timing, actual applied reward, clipping/history, unchanged teacher/learner tensors and label count. No extra teacher rollout or deployment blending. Preserve failures; no automatic retry. Existing frozen inference implementation is unchanged.

## 2026-10-02 — Flat-anchor route policy pilot (preregistered)

`docs/lite3_full_policy_flat_anchor_protocol_20261002.json`: fresh original-base initialization, seed283,512updates/2,097,152 controls; frozen seed293 cap48,000. Add flat-only frozen-original emitted-action MSE reward weight-2; other route/symmetry/reward/PPO settings unchanged. Expected524,288 flat target labels from charged training states, no extra teacher rollout. Smoke520 controls passed with130 labels, pre-action timing and rough-zero checks. This explicitly uses a frozen policy target and has no inference fallback. Preserve prior failed replication; require a new same-configuration independent run before new-terrain validation. Prior full-policy training/evaluation7,478,483 plus route engineering648 and environment smokes1,040 retained.

## 2026-10-02 — Independent flat-anchor replication (preregistered)

`docs/lite3_full_policy_flat_anchor_replication_protocol_20261002.json`: first anchor pilot5/5 on both terrains, base2/5 and0/5, paired flat retention passed. Gate report hash locked. Fresh original-base initialization, learning seed307, frozen seed311; same anchor weight2 and full configuration,512updates/2,097,152 controls plus48,000 evaluation cap. No learned-state import and no pooling with failed unanchored replication. Require repeated gains, flat retention and>=9/10 per terrain across two anchor policies (each>=4/5), then registered new-terrain validation. Preserve prior9,613,671 training/evaluation and1,688 engineering controls.

## 2026-10-02 — Conditional executable new-development transfer protocol

`docs/lite3_flat_anchor_transfer_protocol_20261002.json` binds the two final anchored checkpoints (seeds283/307), previously registered four terrain seeds, independent reset masters, all16 paired sessions and192,000-control maximum. Missing/failed anchored replication audit prohibits launch. Registered-development must be a separate frozen-only membership guard; legacy/sealed guards stay intact. No new terrain mesh, reset or rollout has been generated. Existing unanchored failed replication is not a valid gate. Evaluation never trains or selects replacement checkpoints/starts.

## 2026-10-02 — New-development ledger compatibility correction

Initial transfer launch failed in all16 sessions before ledger creation or worker physics: InteractionBudget manifest validator lacked the new split. Failure audit verifies no ledger manifests/episode summaries and preserves all logs; actual/uncertain controls0. `docs/lite3_flat_anchor_transfer_v2_protocol_20261002.json` binds this audit and keeps identical models, terrain/reset seeds, trials and thresholds. Completes manifest validation plus journal phase guard, and stops a slot at its first infrastructure failure. New output directories replace the unused192,000-control allocation; no terrain performance was observed or selected. Actual ledger serialization/reopening is now tested.

## 2026-10-03 — Registered simulation acceptance completed

Anchored seeds283/307 jointly passed original101/107 (20/20 candidate vs3/20 base) and all four preregistered new terrains (39/40 candidate vs18/40 base). All paired flat-retention criteria passed; the one6.12m candidate fall on874689437 remains without replacement. New transfer used152,079/192,000 controls, uncertain0. Full-policy series including registered engineering totals11,900,884; original base pretraining and earlier SAC/C0 costs remain separately recorded. Final report: `docs/lite3-stable-rough-walking-result-2026-10-03.md`; evidence/hash ledger: `docs/lite3_registered_acceptance_20261003.json`. This meets the registered bounded simulation goal, with simulator route localization; it is not a hardware/universal-terrain or novel-gait-family claim.

## FlowDAgger-20261003-pilot-v1

用户授权实现 FlowDAgger。预登记 `docs/lite3_flowdagger_pilot_protocol_20261003.json`：仅使用原 PPO283 的训练观测，按整个环境流划分验证数据；复制其均值网络初始化条件向量场，6000 次共享噪声耦合 flow-matching 更新，生成器具有显式状态/时间输入和8步 Euler 解码。保留 Gaussian 单峰先验的限制，不声称新多模态步态先验。先验证解码误差、逐离散步固定点反演及重建门槛。

随后冻结 flow，以 PPO283 为仿真纠正教师，在学生访问状态按动作偏差触发接管；2轮各64环境×最多2000步，回合成功后才聚合纠正和自主片段标签，严格50/50采样，每轮2500次适配器更新。失败回合、反演拒绝、所有物理转移保留。最后 teacher/flow零噪声/最终适配器各64环境×最多2000步，同一新seed457配对初态，按平地/101/107分别统计首回合完成和横向漂移；总物理上限640000。固定最终checkpoint，不用本次评测选择模型。此为单步关节动作、PPO专家的工程/开发实验，不冒充原论文操作任务复现或原59/60独立验收。当前空闲GPU3用于运行。

### FlowDAgger v1 interrupted / fixed-command v2

v1发现继承的 `rel_zero_vel_envs`/仅单轴覆盖仍会清零前进命令，round00有22/64初态指令为零，违反本轮固定前进测试意图。中断时Kit吞掉SIGINT并退出0，原report/ledger仍显示running，不能按退出码认定完成。保留原始文件，独立 `docs/lite3_flowdagger_v1_interruption_20261003.json`记录charged424064、confirmed424000、uncertain64；不晋升其adapter或任务表现。

预登记 `docs/lite3_flowdagger_v2_protocol_20261003.json`：复用哈希锁定的已验证flow基础模型，重新从零adapter/optimizer/buffers开始；调用已有 `configure_forward_command` 关闭所有静止/单轴覆盖，逐步断言0.3/0/0；增加初态actuator delay配对审计、全部teacher调用计数及安全点信号中断。学习超参不变，新collection461/462、evaluation463，新增物理上限640000；v1成本另计，uncertain64不抹除。仅runner实现和任务指令修正，不改旧PPO源码或已有59/60结果。

### FlowDAgger v2 result

固定指令v2完成：460160/640000控制转移，uncertain0。两轮各2500次噪声策略更新，成功回合纠正标签174+107=281，全部通过离散解码重建；自主片段167030，严格50/50训练。最终相同新初态/actuator delays的teacher、flow零噪声、FlowDAgger分别56/64、64/64、63/64；适配器101为23/24、107为24/24、平地16/16。FlowDAgger最坏中线横偏0.4865m，对比flow0.4528m；未建立适配增益，不晋升为新的正式策略。

独立审计重建全460160条轨迹记录，单独加载最终网络在CPU复现132条动作且不加载teacher。flow/teacher冻结、adapter实际更新、评测零接管及固定指令逐步检查通过。30项相关单元测试通过。v1负结果及64不确定转移保留；本次全部尝试charged884224、confirmed884160。报告 `docs/lite3-flowdagger-result-2026-10-03.md`，证据 `docs/lite3_flowdagger_result_20261003.json`。这是单步动作和PPO专家的FlowDAgger核心机制实现，Gaussian蒸馏先验，无多模态步态、独立训练复现、未知地形或实机结论。

## FlowDAgger-20261004-paired-correction-v3

用户要求继续改进。只读旧训练数据诊断保存于 `docs/lite3_flowdagger_correction_diagnosis_20261003.json`：旧281条标签来自33/43个成功回合，动作重建/拟合本身不足以证明纠正有益；单分支记录没有同状态无接管结果，不能据此认定任何单条纠正改善了任务。

预登记 `docs/lite3_flowdagger_v3_protocol_20261004.json`：保持已锁定flow、2轮、每轮2500更新和50/50批采样，fresh learner479。每轮先无接管当前学生、再带纠正学生，collection487/488全初态及actuator delays配对；首干预前obs/root最大差<=1e-4才能比较。纠正回合须完成，且挽救参考的真实失败/deadline，或RMS至少改善10%且5mm、最大横偏不增、耗时增加<=5%。只接收合格纠正标签，保持数据来自完整无接管成功回合；至少32条各类样本及2个合格回合才更新。数据不足明示跳过，不把零噪声当成学习成功。

最后teacher/flow/原v2/新v3四臂在新seed491配对复测，各64x最多2000步；总新增上限1024000物理转移，包含所有padding、失败和双分支再生成。没有免费快照回滚，额外仿真特权与成本单列。开发晋级要求各地形完成数不低于v2及flow，rough平均回合横偏RMS较flow改善>=5%、最坏横偏不增，且确实更新adapter；如达标，再独立登记第二次学习复现。保留旧v2负结果，不从其权重或replay开始训练。

### v3 result and v4 continual preservation

v3按预算完成733184/1024000物理转移，uncertain0；配对筛选7+10=17个有益干预序列，仅接收60+55=115条反演纠正，5000次更新。独立轨迹/资格审计通过。新seed491测试：flow61/64、旧v2 64/64、新v3 61/64；rough平均回合RMS分别0.1250/0.1206/0.1222m。v3未晋级，不启动其独立复现；筛选质量不能替代最终策略收益，也不能把回合收益归因给每条动作。

另登记 `docs/lite3_flowdagger_v4_protocol_20261004.json`：从锁定的v2 adapter继续学习，fresh optimizer/buffer，不继承v3权重或标签；保持同初态收益筛选和50/50混合，在自主半批增加已执行动作MSE保持项weight25，lr从3e-4降至3e-5，每轮最多1000更新。此为持续学习和保持的组合改进，不作单因素因果归因。learner499，collection503/504，evaluation509，新增最多1024000控制步；原v3成本与负结果保留。各地形完成不低于旧v2及flow、rough RMS较flow改善>=5%、最坏横偏不增且实际更新，才启动单独登记的持续学习复现；未通过不宣称改善。

v4启动前补全增量门槛：除较flow平均rough RMS改善>=5%，还须较继承的旧v2改善>=2%，最坏rough横偏不得高于两对照；避免把继承来的已有收益误算成此次新学习收益。此门槛写入启动前协议，之后不根据结果放宽。

### v4 null result and v5 alternative candidate expert

v4完成735232/1024000转移、uncertain0。仅5个序列合格，共11条纠正，未达预登记32条门槛，两轮均跳过更新；actor与旧v2逐张量相同。seed509旧v2/未更新候选同为63/64、rough平均RMS0.12110m，flow63/64、0.12649m。不能把继承的成绩算成新学习。新增修复：即使数据不足跳过更新，也保存已验证buffer、episode IDs和RNG，避免后续持续收集丢失数据；原v4轨迹/反演/资格文件完整保留，不重写历史。

预登记 `docs/lite3_flowdagger_v5_protocol_20261004.json`：候选纠正教师改为此前独立训练的PPO307（SHA05f5fd60…），不默认它更好，依旧必须通过同初态收益筛选。其余v4方法、初始化和门槛不变；fresh learner547、collection557/558、evaluation563；不读取v3/v4训练或评测数据作为标签。新增上限1024000步，旧累计charged2352640/confirmed2352576/uncertain64保留。仅开发门槛全部通过，才另登记独立持续学习复现。

### v5 outcome and offline capacity diagnosis

v5完成740800/1024000控制步、uncertain0，两个回合各16个合格干预序列，共36918条反演标签、拒绝0，2000次实际更新。新seed563的teacher307为64/64、flow64/64、旧v2 63/64、新v5 59/64；v5最坏横偏0.67667m、rough平均RMS0.12763m，均未达到增量门槛，不晋升也不追加其复现。数据不足已排除，但拟合误差/目标冲突等仍待诊断，不能把教师收益直接等同于学生收益。

新增离线诊断 `docs/analysis/diagnose_flowdagger_capacity.py`：只读v5训练buffer、绝不读取评测轨迹；按完整buffer回合每4个留1个验证，比较256/128与1024/512噪声网络。两者从同一v2函数初始化，宽网络用函数保持复制及零和输出扰动打破对称；各1500更新、batch512、lr3e-5、动作保持25、相同采样序列。新增物理成本0，不直接晋升离线模型。预先门槛：宽网络验证纠正动作RMS较窄网络至少下降20%，自主动作RMS不高于窄网络+0.002，初始函数误差<=1e-5，才值得单独注册物理候选。此为拟合诊断，不代表闭环收益。

### Capacity diagnostic passed; v6 widened adapter

离线1500更新/网络的固定回合留出诊断通过：256/128纠正动作验证RMS0.08686，1024/512为0.06110（约下降29.7%）；自主RMS0.01508→0.01406，初始函数差1.14e-6。见 `docs/lite3_flowdagger_capacity_diagnosis_20261004.json`。不把离线拟合改进当成物理收益，不导入离线权重。

预登记 `docs/lite3_flowdagger_v6_protocol_20261004.json`：只扩大noise actor至1024/512，Net2Wider复制并用零和输出扰动打破对称，保留v2初始函数；保持v5教师307、配对门槛、50/50+动作保持25、lr3e-5和每轮1000更新。fresh learner617，collection619/620，evaluation631。加入未经训练的宽网络第五评测臂，避免将参数化/浮点差异误算为学习；必须额外不退化完成数、rough RMS较该初始臂改善>=2%、最坏横偏不增，原v5所有门槛仍有效。新增上限1152000步，既有charged3093440/uncertain64完整保留。通过才另登记独立持续学习复现。

### v6 failed; v7 disjoint supervision regions

v6完成835776/1152000物理步、uncertain0。seed631 flow64/64，旧v2 62/64，未训练宽网络62/64，训练后v6 54/64；训练后最坏横偏0.66835m。拟合误差下降未转化为闭环收益，保留全部失败，不晋升。

只读v5训练数据发现22组完全相同的起步观测同时进入纠正与自主保持数据，执行动作RMS差约0.237–0.240；当时均为低偏移/正常姿态。这证明目标冲突存在，不证明全部跌倒的因果。36918纠正中17421处于低偏移/正常姿态；157154自主样本中44024已超拟定偏差阈值。证据 `docs/lite3_flowdagger_label_conflict_20261004.json`。

预登记 `docs/lite3_flowdagger_v7_protocol_20261004.json`：继承v6结构和优化设置，但回到同一锁定v2初始函数，fresh learner659、collection661/662、evaluation673；不继承失败模型/数据。只有横偏>0.10m、航向>5°、或姿态倾角>10°之一成立时才允许训练教师接管，仍要求动作差>0.08及完整配对收益；自主保持只收该状态域的补集。严格验证两类标签状态域互斥。该规则仅决定监督来源，推理仍是单一noise actor+冻结flow，未增加部署回退。0.10m是监督区域阈值，不是已达到的漂移成绩；任务1.65m边界和全部v6增量验收门槛不变。新增上限1152000控制步，旧成本/失败不覆盖。

### v7 audited completion but no mean-drift gain; v8 lookahead corrective teacher

v7完成836864/1152000控制步、uncertain0；59个合格干预序列，18203纠正+97107互斥自主保持样本，2000更新。逐步轨迹、教师区域/动作差判据、两缓冲状态域互斥、资格与独立模型回放审计通过。seed673各臂均64/64，v7最大rough横偏0.34139m、平均回合RMS0.12943m；旧v2为0.42936/0.12415，未训练宽网络0.34784/0.12493，flow0.45013/0.12998。平均RMS门槛未通过，未启动复现，不把大部分来自参数化/数值差异的最坏值变化冒充新学习收益。

按用户继续改进指令，登记 `docs/lite3_flowdagger_v8_protocol_20261004.json`：将纠正教师换为同一基础步态的PPO283加几何路线跟踪参考。实际横偏y、航向psi，1m前视中线目标给出期望方向-atan2(y,1)，教师查询航向误差psi+atan2(y,1)，其余观测不变；这是明示的脚本教师控制，不冒充真实传感值。降低动作差采样阈值至0.02，捕获小幅同一步态转向纠正；v7互斥区域和配对收益门槛、网络/优化器设置不变。learner701、collection709/710、evaluation719，新增最多1152000步。记录teacher查询route特征，学生始终读取原始实际观测；推理不增加几何回退。旧累计charged4766080/confirmed4766016/uncertain64保留。原增量验收门槛不变，失败不补试。

### v8 passed audited development gates; independent replication registered

v8完成819776物理控制步、uncertain0，轨迹/配对资格/无教师CPU动作回放审计通过。seed719：训练后64/64（rough48/48），rough平均回合横偏RMS0.09323m、最坏0.31635m；flow63/64、0.12415/0.41752m；旧v2 63/64、0.11871/0.41795m；未训练宽网络62/64、0.11561/0.43921m。实际2000更新，4092纠正标签，所有固定开发门槛通过。仍未宣称独立复现。

在复现实验启动前登记 `docs/lite3_flowdagger_v8_replication_protocol_20261004.json`：使用此前已指定learner727、collection733/734、evaluation743，完全相同方法，从原始锁定v2重新初始化宽网络，fresh optimizer/buffers，不继承v8最终权重。两轮各最多1000更新，五臂配对评测，最多1152000新增物理控制步。原全部门槛保留，失败不换种子/检查点补试。之前累计charged5585856/confirmed5585792/uncertain64。

### v8 independent replication failed fixed gates

复现完成825216物理控制步、uncertain0；2000更新、36个合格序列、7348纠正+120000自主标签。所有轨迹/资格/CPU部署回放审计通过。seed743：flow/旧v2/未训练宽网络均64/64，训练后61/64（flat16/16、101为21/24、107为24/24）。rough平均回合RMS分别0.13297/0.11799/0.11898/0.09835m，最坏0.44808/0.41073/0.33958/0.39639m。完成率与相对初始宽网络最坏横偏门槛失败。三次物理失败在101进度约6.0–6.3m，末端倾角约42度，三个配对对照同初态均成功。不重试、不替换种子、不晋升，不宣称可复现收益。累计charged6411072/confirmed6411008/uncertain64，旧预训练等成本另计。详见 `docs/lite3-flowdagger-v8-result-2026-10-04.md` 与机器可读复现结果及失败诊断。

### 2026-10-05 recovery coverage diagnosis and v9 preregistration

只读v8复现训练数据，未读取评测作为训练标签：7348纠正仅292条倾角>10度、无>15度；120000自主保持无>10度样本。成功无接管参考回合有630+487=1117条>10度（其中107条>15度、16条>20度），被v8区域规则全部排除。证据 `docs/lite3_flowdagger_recovery_coverage_20261005.json`，新增物理成本0；这是覆盖缺口证据，非所有失败因果证明。

预登记 `docs/lite3_flowdagger_v9_protocol_20261005.json`：路线纠正只在倾角<=10度且横偏/航向超原阈值；大倾角改从成功无接管参考回合保留自主动作。自主半批次25%采样恢复状态、75%其他自主状态，纠正/自主50:50仍保持；两自主层各至少32条，否则明示跳过。其余v8设置、所有验收门槛不变，fresh learner751、collection757/758、evaluation761，从原始v2重新初始化，不加载v8权重/数据。新增最多1152000物理步；若通过再登记独立learner773、collection787/788、evaluation797，否则保留失败不补试。另增加逐步碰撞/姿态/越界失败项以便独立追溯，物理终止规则不变。此前累计charged6411072/confirmed6411008/uncertain64。

### v9 failed fixed gates; v10 correction-action anchor preregistration

v9完成822784物理步、uncertain0，轨迹/恢复样本/失败项/CPU动作回放审计通过。2000更新、5395纠正、120183自主（863条大倾角恢复）。seed761 flow62/64、旧v2 62/64、初始宽网络63/64、v9 62/64；v9平地16/16、101为24/24、107为22/24，rough RMS0.100747m、最坏0.459848m。107完成门槛及相对flow/初始宽网络最坏横偏门槛失败，不启动v9复现。env43/49均姿态超限、无接触/越界标志；同初态三个对照均成功。训练参考00上动作变化mean RMS0.00984、p95 0.02223（纠正状态），不能认定动作幅度已过大或证明失败因果。见 `docs/lite3_flowdagger_v9_action_shift_20261005.json`。

预登记 `docs/lite3_flowdagger_v10_protocol_20261005.json`：保留v9全部设置，仅在纠正半批增加相对冻结初始宽网络的执行动作MSE软约束weight25。此为训练正则化，不是安全保证、不增加部署回退、不将初始动作冒充有益教师标签。fresh learner811、collection821/822、evaluation823，从原始v2初始化，不继承v9模型/数据。最多新增1152000物理步，所有门槛不变；通过后才独立登记829/839/840/853复现。累计此前charged7233856/confirmed7233792/uncertain64；保留失败不补试。

### v10 audited failure, no promotion or replication

v10完成829568物理步、uncertain0；2000更新、8030纠正/118604自主（1278恢复）。8条定期损失日志独立核验锚定项实际参与训练。seed823：flow62/64、旧v2 64/64、初始宽网络63/64、v10 60/64（101为20/24）。rough RMS分别0.12335/0.11576/0.12285/0.10823m，最坏0.50027/0.33881/0.81118/0.48415m；完成及相对旧v2最坏横偏门槛失败。逐步轨迹、恢复样本数、细分失败项、无教师/锚定模型CPU部署回放审计通过。v9/v10六次失败全为姿态超限、roll约−43度，接触/越界为false。未认定因果修复，不晋升、不启动独立复现、不换种子补试。累计charged8063424/confirmed8063360/uncertain64，45项相关测试通过。报告 `docs/lite3-flowdagger-recovery-2026-10-05.md`。

### v11 predeclared recovery perturbation collection

前轮六次失败主要为约−43度侧滚，成功恢复训练中大倾角覆盖稀少。只读v10四个失败最后2秒观测：倾角<10度且横向角速率>1rad/s的行数均0，最高角速率约1.9–4.2rad/s发生在已倾斜后，不能据此声称小倾角角速率阈值能提前预警。评测观测仅作诊断，不进入训练标签。

预登记 `docs/lite3_flowdagger_v11_protocol_20261005.json`：保持v10学习算法/软约束/成功恢复采样，单改训练收集分布。zero-based步600/900，对active且rough的偶数env施加world COM roll速度增量±1.25rad/s和相反方向横向速度∓0.12m/s，选中env之间方向交替，第二次反向。两条配对训练分支日程相同；每次状态写入后重算实际观测并记录前后root/增量。各评测臂全部无扰动，物理规则/任务门槛不变。最多192个训练env-impulse，单独统计；仍最多1152000物理控制步，先预留控制预算再写状态，无额外免费仿真。fresh learner859、collection863/864、evaluation877，从锁定v2重新初始化，不继承失败权重/数据。通过全部原门槛才另登记881/883/884/887独立复现。旧累计charged8063424/confirmed8063360/uncertain64保留。

### v11 failed; v12 severe-recovery collection preregistration

v11完成824576物理步、uncertain0，192训练env-impulse、所有评测0扰动。2000更新、5590纠正/120198自主，>20度成功恢复仅27行、>30度为0。seed877三个对照均64/64，v11为62/64（101为22/24），rough RMS0.10722m、最坏0.45930m，完成和最坏横偏门槛失败；不晋升、不启动复现。累计charged8888000/confirmed8887936/uncertain64。

预登记 `docs/lite3_flowdagger_v12_protocol_20261005.json`：沿用v11训练扰动日程/选择，只将roll增量提高至±3.0rad/s、横向增量∓0.25m/s；自主恢复半批内25%层改为倾角>20度，至少32条成功样本才更新（另一层也>=32）。纠正排除阈值仍10度，不纳入失败回合、不改变评测或验收门槛。fresh learner907、collection911/912、evaluation919，从原始v2初始化；最多1152000物理步/192训练env-impulse。通过后才另登记929/937/938/941独立复现。针对训练覆盖不足的数据候选，不预先宣称稳定收益。

### v11/v12 audits complete; v12 null learning result

v11/v12逐步轨迹、扰动日程、前后root状态、实际body角速度观测、配对资格、失败原因与CPU部署回放审计通过。v12完成825344物理步、uncertain0、192训练env-impulse，所有评测无扰动。8309纠正、116685自主样本中没有>20度的成功恢复状态，两轮都因恢复层不足跳过；0更新，最终actor逐张量等于初始宽网络，64个配对评测回合指标也完全相同。seed919 flow63/64、旧v2 62/64、initial/candidate均61/64；候选rough RMS0.114923m、最坏0.427394m。零学习结果不晋升、不启动复现。累计charged9713344/confirmed9713280/uncertain64。47项相关测试通过，报告 `docs/lite3-flowdagger-perturbation-2026-10-05.md`。下一阶段需先研究有效的大倾角恢复示范来源，不能在现有不足数据上堆优化次数。

### Recovery teacher: training-only kinematic bank and engineering smoke

从v11/v12的reference/round训练失败中，每完整回合只取首个20–30度实际观测及配套pre-root/joints；不读取评测。27个初态按地形内每第4个完整回合留出：train20（101为8、107为12），validation7（101为3、107为4）。这些是运动学重置，不恢复求解器/接触/执行器记忆或旧动作历史，不声称精确快照，也不把失败动作当标签。首个离线提取尝试因关闭npz后取长度报错，0物理步；修复后的bank-v1-attempt02保留哈希和来源，旧目录不覆盖。

登记 `docs/lite3_recovery_teacher_smoke_protocol_20261005.json`：PPO283冻结执行，仅训练split初态，64环境×64步上限4096，seed947；验证重置顺序、实际倾角/观测、命令与动作历史，无优化、不做资格结论。之后计划以PPO主动学习恢复，仍必须独立验证完整跑道与未参与训练的恢复初态，未验证前不接入FlowDAgger纠正标签。

### Recovery PPO teacher development preregistration

smoke-v1完成4096物理步、uncertain0，重置实际倾角>=19.9度、joint/event顺序、固定命令与动作历史检查通过；48个训练恢复初态试验中44个在64步内物理失败，不能把状态初始化成功当成恢复成功。59项相关测试覆盖重置训练/验证隔离、完整初态遍历、48D平地锚定、actor-only初始化及既有FlowDAgger契约。

预登记 `docs/lite3_recovery_teacher_train_protocol_20261005.json`：learner953，PPO283 route48 actor-only起步，fresh critic/optimizer，std0.2、lr3e-5、512迭代/2097152物理步，粗糙地形回合起点交替普通/20个训练恢复初态，flat始终普通。横偏reward改-4，保留原姿态奖励、flat anchor2和mirror1。恢复动作通过真实物理PPO收益学习，不用失败动作作教师标签。最终检查点唯一，不重试择优。

同时预登记 `docs/lite3_recovery_teacher_eval_protocol_20261005.json`：seed971，PPO283/旧v2/候选三模型分别普通全程与7个未参与训练的恢复初态（48rough槽重复，另16flat），最多768000物理步。候选完整跑道逐地形完成不低于两基线、rough平均RMS较v2下降>=2%、最坏不增；恢复须连续25步倾角<=10度且完成终点，逐地形不低于两基线、rough总>=80%、每个留出来源初态至少成功一次。全部通过才另登记FlowDAgger接入，不默认新教师更好。新增开发上限2869248含smoke；旧FlowDAgger9713344/9713280/64与更早PPO成本分别保留。

### Recovery teacher result: full-course improvement, recovery gate not met

训练512迭代/2097152物理步，覆盖全部20训练初态、0验证泄漏；恢复回合控制109998、重置1444、完成224/失败1206。完整跑道seed971：旧PPO283为54/64、旧v2为63/64、新PPO教师64/64；rough平均RMS分别0.12723/0.11834/0.04562m，最坏0.50833/0.42922/0.13767m。留出恢复48槽（7来源初态重复）基线均0/48，新教师29/48（101为17/24、107为12/24）；case0/19均0/6。全部完整跑道门槛及相对恢复基线门槛通过，但80%恢复率/每来源至少一次成功门槛失败。不晋升、不接入FlowDAgger、不宣称独立训练复现。

59项相关测试通过；训练/冻结轨迹、重置分区与状态、动作历史、CPU动作回放独立审计通过。实际新增2622592控制步（smoke4096+训练2097152+评测521344），无不确定步。加此前FlowDAgger累计charged12335936/confirmed12335872/uncertain64，更早PPO基础成本另列。最终SHA67006951ebb2ed337aeac7ede2a21709ba6bcc985e28220429a9368d2c17cdb9，报告 `docs/lite3-recovery-teacher-2026-10-05.md`。

### 2026-10-06 requested video and experiment planning

用户询问成功录像并要求后续计划。原64/64运行只有状态/动作轨迹，没有相机视频。登记 `docs/lite3_video_demo_protocol_20261006.json`：冻结新PPO教师SHA67006951…、原seed971、原64环境普通跑道，提前固定env16（首个rough101槽）为录制对象，960×540/25fps，首回合成功或失败都保留。仅渲染隐藏其他机器人，物理仍执行并计入最多128000控制步；相机帧不额外推进物理。明确新运行演示而非原评测录像，不训练、不替换种子、不宣称独立复现。此前累计12335936 charged/12335872 confirmed/64 uncertain，原PPO更早基础成本另列。

录像attempt01继承DISPLAY=:3，Kit相机启动报GLXBadFBConfig并退出2，0 charged/confirmed/uncertain，原running日志保留，sidecar `startup_failure.json`补充退出原因。随后同模型/协议/seed/env16改用unset DISPLAY离屏后端，输出新目录attempt02；这是0物理步的图形初始化修复，不是择优重跑成功样本。

录像attempt02完成86080控制步、uncertain0，64/64，选定env16成功；编码659帧/26.36秒。图像检查发现env0的visibility通过克隆继承隐藏了选中机器人，画面无机器人，不作为交付录像。保留原MP4/轨迹并修复选中env显式MakeVisible。另登记2×64=128步可见性工程probe，先看预览再进行完整补录；不是成功样本筛选。

visibility-probe完成128步，预览确认选定机器人已可见。登记delivery协议：同一冻结模型/seed971/env16，换较近跟随镜头与balanced渲染，仍最多128000步，任何结果保留。前两次图形故障/空画面结果均保留，不作独立复现或替换旧评测。

最终delivery完成86080控制步、uncertain0，64/64；预选env16/rough101成功，1318步，RMS0.0262197m、峰值0.0962181m。可见机器人MP4为659帧/25fps/26.36秒，SHA7402a723f1924d27b83aee2e59cfe792f38739e67744d6af8f05d0191cc67740；起点/中后段画面人工检查，帧遥测与物理轨迹/CPU动作回放独立审计。新运行演示，不冒充原评测或独立训练复现。本轮工程总172288步，累计charged12508224/confirmed12508160/uncertain64。后续方案 `docs/lite3-next-experiments-plan-2026-10-06.md` 为分阶段计划，除P0未启动新训练或新场景矩阵。

### P1 frozen multiseed verification preregistered

按用户继续指令，登记 `docs/lite3_teacher_frozen_multiseed_protocol_20261006.json` 及5个子协议，固定新初态seed2017/2027/2029/2039/2053、当前PPO教师SHA67006951…与旧v2 SHA3e2e7a86…，每模型每seed64回合（16flat+24rough101+24rough107），全部在GPU3，无相机、无训练。共最多1280000控制步。汇总门槛沿计划：总完成>=98%、各地形>=95%、逐地形完成不低于旧v2、rough平均回合RMS<=0.06m、最坏<=0.20m。逐seed及试验级Wilson区间同时报告，不将同地形重复试验当成新地形泛化；不替换失败seed，不用其结果改旧恢复门槛。此前累计charged12508224/confirmed12508160/uncertain64。

同时预登记P1独立学习复现：learner2063从原PPO283开始，原方法/20训练初态/512更新/2097152物理步，fresh critic与optimizer，不加载teacher-v1学习后参数。验证seed2081，仍为原7个留出来源初态及普通跑道，三臂最多768000步；保持原80%恢复及每case成功门槛。见 `docs/lite3_recovery_teacher_replication_train_protocol_20261006.json` 和对应eval协议。新学习种子与评测种子均在训练启动前固定，不替换失败尝试。该复现是P1的一部分，不会因原教师恢复门槛未过而自动称作资格通过。

P1冻结五种子复测完成并独立审计：旧v2为314/320（flat80、101为118、107为116），当前教师320/320；rough平均回合RMS0.118715→0.045637m，最坏0.444770→0.178652m。全部原冻结复测门槛通过；试验级Wilson95%教师[0.98814,1]，不解释为独立地形泛化。实际889792控制步、uncertain0。随后启动此前登记的learner2063独立重训，未改训练方法/门槛。

P1独立训练512轮完成并审计。新模型seed2081普通跑道60/64（旧v2为63/64），恢复0/48，未复现原教师的恢复收益；恢复回合仅50785/2097152=2.42%控制步。独立评测508224控制步，uncertain0；失败模型/结果保留，不替换种子。

按既定P2计划登记 `docs/lite3_p2_recovery_sampling_protocol_20261006.json`：复用已完成learner2063为原采样对照，新候选同seed/初始化/PPO/奖励/数据，只改为固定32恢复槽（101/107各16），期望恢复回合步数25%；32flat和64普通rough保持。60项相关测试通过。新训练上限2097152；候选在同seed2081仅跑普通/恢复两条件，上限256000，只有初态/执行器延迟哈希完全匹配才复用旧三臂对照；新增上限2353152。明确收益门槛预先固定：恢复完成比控制增加>=8/48，普通逐地形完成不降、RMS<=控制1.05倍、最坏不增；原教师资格单列，不能用消融收益替代80%门槛。通过才另登记第二配对种子复现。

P2首组learner2063训练恢复占比524288/2097152=25%，审计通过；同seed2081初态/延迟哈希一致。对照→固定恢复槽：普通60→64/64，RMS24.44→8.02cm，峰值55.64→15.61cm，恢复0→37/48。全部消融收益门槛通过，但原资格80%及case0至少成功一次仍失败。新增2271488控制步；累计charged18274880/confirmed18274816/uncertain64。

条件触发第二配对组，登记 `docs/lite3_p2_sampling_replication_protocol_20261006.json`：learner953从原PPO283重新初始化，仅采样改为32固定恢复槽，512轮；复用原teacher-v1作为控制，eval971两条件最多256000控制步，须同初态哈希。新增上限2353152步。收益门槛不变：恢复比29/48增加至少8、普通逐地形完成不降、RMS<=原1.05倍、峰值不增；原教师资格单列。两组全部结果保留，不替换失败种子。协议生成时系统python不存在导致首次命令在读取缺失协议时退出，未创建仿真环境或输出目录，0物理步；保留启动错误日志后使用python3生成协议。

P2第二配对组learner953完成512轮，恢复控制524288/2097152=25%；eval971两条件172352控制步，训练/评测及初态延迟配对审计全部通过。原对照→固定槽：普通64→64/64，RMS4.562→7.147cm、峰值13.767→22.872cm，恢复29→39/48；恢复增益/完成保持通过，RMS及峰值保持失败。原资格80%通过但case0仍0/6而失败；不晋升教师、不接入FlowDAgger。第二组SHA3df38326…，实际新增2269504，无不确定步。本轮P1+P2共8036160，累计charged20544384/confirmed20544320/uncertain64；原PPO基础成本另列。60项相关测试通过。两组报告及下一项未启动的单变量课程设计见 `docs/lite3-p2-sampling-result-2026-10-06.md`，审计完成于本地2026-10-07，文件名沿用预登记日期。

### 2026-10-07 P2 inclination curriculum preparation

按用户继续指令推进已列单变量课程：仅从原bank manifest中20个TRAIN来源回合提取首次10–15°、15–20°且早于原20–30°状态的活动帧，全部来源须具备两个层级，否则停止而不剔除来源。原27行及验证划分不变，禁止使用7个验证来源的相邻状态。纯离线提取0新物理步。先检查覆盖、重置与物理smoke，再登记512更新的正式课程训练；未改奖励/终止/网络。

课程库提取完成：保留原27行，新增40训练状态（每来源每较轻层级1个），全部20训练来源齐全、0验证相邻状态。预登记三个分层smoke，seed2099、原PPO283、每层64环境×64步=4096，总上限12288控制步；任何失败保留，smoke仅检查状态重置与物理执行，不以恢复成功筛选初态。

三层smoke及CPU回放审计通过，共12288控制步、uncertain0；64步截止前物理失败数为0/29/46，非成功率评测。训练案例诊断：20原训练起点roll均为负，case11/22/26在两固定槽学习种子均零任务完成。登记课程主协议 `docs/lite3_curriculum_ablation_protocol_20261007.json`：learner2063相同PPO283初始化/32恢复槽/优化器/奖励/512更新，仅前128更新10–15°、中128更新15–20°、末256更新原20–30°，跨阶段正在执行的回合不强制重置。新增训练2097152+eval2081两条件最多256000；使用原验证bank与同初态配对。相对固定槽对照37/48要求增加至少8（>=45/48）、正常逐地形完成不降、RMS<=1.05倍、峰值不增；原教师资格单列。全部增益门槛通过才另登记第二种子，不挑中间权重。

课程seed2063完成512更新，训练审计确认3层实际重置271/436/1190、60训练状态全暴露、验证0、恢复控制524288。最终SHA901d22a5…；eval2081初态/延迟配对与CPU轨迹审计通过，普通64/64但横偏RMS8.02→15.31cm、峰值15.61→25.49cm，恢复37→36/48，case0/19均0/6。消融增益失败，原教师资格失败，不晋升、不触发第二组、不接FlowDAgger。61项相关测试通过；旧配对汇总回归逐字段一致。本轮smoke12288+训练2097152+评测163840=2273280；累计charged22817664/confirmed22817600/uncertain64，旧PPO基础训练另列。报告 `docs/lite3-curriculum-result-2026-10-07.md`。

### 2026-10-08 recovery dynamics diagnosis

按用户继续指令，登记两个短诊断，seed2111、固定32槽learner2063最终模型：原20训练来源初态与仅root角速度乘0.5的反事实初态。64环境×128步×2=16384控制步上限，16flat与48rough槽全部计费，未结束样本按截断记录，不称恢复成功。新增可选遥测在控制步末/自动重置前记录足端位置、足端接触净力、基座质心位置及既有根姿态/角速度；不是静态支撑多边形或全机器人质心。所有验证来源与其他状态量不变，反事实库只用于诊断，不转为验证集或训练标签。

两个角速度诊断完成并通过轨迹及反事实初态审计：共16384步，无不确定步；2.56秒内原训练起点失败14/48、角速度减半失败9/48，case11/22/26在两组均全部失败，不能宣称恢复合格或把角速度作为唯一原因。训练曝光诊断显示这些来源每个仅约3000控制，而容易来源约5–6万。

因此登记按来源固定槽工程smoke：128×64=8192上限、seed2113、PPO283、原20–30°库不变；每rough地形16固定恢复槽按原bank顺序取模永久分配来源，101的8来源各2槽，107的12来源各至少1槽、顺序前4来源另1槽。不得按结果重排或移除来源；普通32flat+64rough不变。后续正式实验只改变恢复来源分配，不改角速度/倾角/奖励/网络。GPU2/3有其他用户作业，后续使用空闲GPU0（同型RTX3090），不干预其他进程。

来源固定槽smoke完成8192控制步、uncertain0，初态分配/轨迹审计通过。正式登记 `docs/lite3_source_stratified_ablation_protocol_20261008.json`：learner2063与原固定32槽同初始化/预算/PPO，仅将恢复来源固定在槽上，不减角速度、不降倾角；512更新/2097152训练步，eval2081两条件最多256000。收益门槛保留相对37/48至少增加8、普通逐地形完成不降、RMS<=1.05倍、峰值不增；原教师资格单列。全部增益门槛通过才另登记953复现。

按来源固定槽learner2063完成512更新，永久映射全轨迹审计通过；case11/22/26曝光32768/16384/16384，任务完成仍全0。GPU0评测与旧GPU3初态/延迟哈希一致：普通64→62/64，RMS8.02→6.75cm，峰值15.61→25.65cm；恢复37→37/48，case0仍0/6。收益与原教师资格均失败，不晋升、不触发第二组、未接FlowDAgger。SHA d18cbb36…，62项测试及轨迹/CPU回放审计通过。本轮诊断16384+smoke8192+训练2097152+评测170112=2291840，uncertain0；累计charged25109504/confirmed25109440/uncertain64，旧PPO基础成本另列。结果与下一步受约束动作搜索设计见 `docs/lite3-source-stratified-result-2026-10-08.md`，后者尚未启动。

### 2026-10-08 bounded recovery sequence search engineering

用户要求按计划继续；先登记三臂工程检查：原训练case11固定在64环境中的24个rough107槽，其余40槽正常起步并全计费；seed2143，固定PPO恢复模型SHA065c1c3b…。无前缀/零前缀/非零前缀各64控制步，上限3×64×64=12288。前缀为3×4控制步的12维临时动作修正，12步后严格归零；保持原动作绝对限幅与物理终止。记录末物理子步/自动重置前的computed/applied关节力矩和接触净力。先验证零前缀物理等价、非零执行与CPU回放，所有失败保留，不以工程表现选择正式动作。

短动作搜索工程三臂均完成并通过审计，12288步/uncertain0，无/零前缀公共轨迹逐字段精确相等。审计脚本插入块缩进错误曾导致离线启动失败，原错误日志保留，修复后仅重跑审计，未重跑物理。65项相关测试通过。

正式登记 `docs/lite3_recovery_sequence_search_protocol_20261008.json`：原TRAIN case11/22/26，每case24候选×6代CEM，原冻结PPO SHA065c1c3b…，3段各4步的12维临时残差、幅度界2，执行仍受原绝对动作限幅。固定search_seed2153、proposal_seed2183+case，精英6、初始std1、下限.25；每代含零前缀与incumbent，保留全部失败。每次128步，回正正例须交回原PPO后连续25步<=10°且该窗口至少5步有2足Fz>5N，同时存活到截止或完成路段。每case先付费baseline，搜索172032上限；仅发现正例的case用2161/2179两新初态种子固定前缀对原PPO复核，各>=20/24且均优于base，最多98304。正式上限270336，加工程总上限282624，不称短期正例为完整恢复资格。

首轮短动作搜索完成204800控制步、uncertain0，独立CEM/轨迹选择审计通过。case11/22无短恢复正例；case26第4/5代有6/15个正例，但最高分固定前缀在新seed2161/2179仅4/24和3/24（base均0），未通过20/24复核。成功全部集中Hip/Knee延迟1/0组，其他组全失败；不晋升、不启动完整路段跟进。

据此另登记稳健性筛选 `docs/lite3_recovery_robust_screen_protocol_20261008.json`：原训练搜索中全部20个不同正例参数固定后，逐个在seed2153全部24槽重放，不改延迟/动作，不用复核结果挑候选池。按四种原有Hip/Knee延迟组合的最差成功率排序，再比较总成功数及均值分；须每组>=80%且总>=20/24。通过才用新seed2203/2207固定动作对原PPO复核。新增上限163840+32768=196608；原失败搜索完整保留，仍只为TRAIN case26诊断，非新教师资格。

稳健筛选及独立审计完成196608控制步、uncertain0，选中固定候选15；两个新seed2203/2207的原PPO均0/24、前缀方案均24/24，四延迟组合均通过。原单最高分方案4/24、3/24失败保留，不用新结果替换原结论。

条件触发完整路段跟进，登记 `docs/lite3_recovery_prefix_fullcourse_protocol_20261008.json`：仅TRAIN case26，固定候选15与冻结PPO，seed2213/2221，每seed原PPO/前缀两臂、64环境×最多2000步，总上限512000。24目标槽需每seed>=20完成原终点且满足交还后持续回正/支撑条件、四原延迟组各>=80%，并优于base。其余40槽与结束padding计费。该阶段不更新权重、不替换前缀、不触碰留出case0/19，也不称作新教师资格或通用策略。

完整路段跟进四臂完成并独立审计：seed2213/2221，原PPO TRAIN case26均0/24，固定候选15＋PPO均24/24且四延迟组全通过。前缀仅前12步，后续冻结PPO完成原终点；模型权重不更新、未参与训练的来源未使用。横偏RMS58.15/58.73cm、最坏68.43cm（起点已偏32.65cm，相对起点最大额外35.78cm），不宣称精密循迹或通用恢复。完整跟进353152步；本轮工程12288+搜索204800+稳健筛选196608+跟进353152=766848，uncertain0；累计charged25876352/confirmed25876288/uncertain64，原PPO等前置成本另列。65项相关测试通过。报告 `docs/lite3-recovery-sequence-search-result-2026-10-08.md`；下一项新的训练正例采集和状态相关策略学习尚未启动，不把本轮复核轨迹直接回流训练。

### 2026-10-08 state-conditioned prefix imitation pilot

用户继续后，登记 `docs/lite3_prefix_imitation_plan_20261008.json`：新seed2237采集TRAIN case26固定候选15示范，新seed2243采普通行走保持，各64×2000上限，总256000；不回用此前搜索/复核轨迹作标签。只用成功完整首回合，整环境流id%4==0留作离线验证。learner2247从固定槽PPO2063 actor初始化，完整48D actor监督MSE微调2000步、lr1e-5、批512（128纠正/128恢复后保持/128flat/64rough101/64rough107），仅最终checkpoint。不输入case或时间，不使用前缀作推理后备；这是监督微调，不是PPO或FlowDAgger更新。

pilot eval2251原PPO/学习actor各普通与TRAIN case26两条件，最多512000。须case26>=20/24且四延迟组各>=80%、最坏横偏<=.70m，普通逐地形完成不降/RMS<=1.05倍/峰值不增；.70m仅该TRAIN案例自主迁移工程门槛，不代替旧教师资格。全部通过才对原7来源/seed2081做候选两条件验证（最多256000），原资格不变；本组新物理总上限1024000，监督训练不增加物理步。

新示范采集完成：普通course64/64、prefix条件64/64含case26为24/24；各88448步，共176896、uncertain0。一次过早发起的离线审计被running状态检查拒绝，日志保留，等待完成后两轨迹均审计通过，未重跑物理。数据集逐行重建及整环境流划分审计通过。锁定 `docs/lite3_prefix_imitation_train_protocol_20261008.json` 及无前缀pilot协议，按既定2000步/学习率进行监督微调。

首个自主Actor pilot完成并审计：无脚本前缀的case26恢复24/24，但普通64→61/64、RMS8.07→16.01cm、最坏19.27→79.20cm；case26峰值78.57cm也超过70cm工程上限。原pilot门槛失败，不启动原7来源验证。本组采集176896+pilot352512=529408，监督训练2000更新、0新物理，失败模型SHA29194d51…保留。

按用户持续改进授权，另登记单一保持权重消融 `docs/lite3_prefix_imitation_preserve10_plan_20261008.json`：全部非纠正样本损失系数1→10，其余数据/划分/原PPO初始化/seed2247/2000更新不变，不从失败模型续训。candidate-only两条件eval2251最多256000，已付费base评测须同初态/延迟哈希才复用；原pilot门槛不变，只有全过才另跑原7来源验证（最多256000），不自动花完未触发预算。

权重10训练和两臂pilot完成并独立审计：完整2000次Adam更新、SHA80c8fbe3…；复用base的两条件初态/执行器延迟联合哈希完全一致。普通64/64（16+24+24），case26自主恢复24/24、四延迟组7/7、7/7、7/7、3/3；普通rough RMS8.77cm、峰值41.95cm，对照8.07/19.27cm，恢复峰值71.30cm超过70cm。三个成功/完成门槛通过，三个横偏门槛失败，不晋升、不启动原7来源验证、未接FlowDAgger。新增173696物理步、uncertain0；新采集加两版pilot总703104、两次监督训练共4000更新而新物理0。累计charged26579456/confirmed26579392/历史uncertain64，完整PPO旧系列11900884另列。

只读轨迹诊断定位第二版普通最坏为rough101/env22（7.04秒、纵向3.66m处41.95cm），恢复最坏env57（3.98秒、纵向7.21m处71.30cm）。case26起点已偏32.65cm，相对起点额外最大38.65cm、结束平均绝对横偏56.88cm；回正不等于回线。36项相关回归通过、两模型轨迹/CPU回放和训练审计通过；保持失败checkpoint与原判定。结果 `docs/lite3-prefix-imitation-result-2026-10-08.md`，对比及预算 `docs/lite3_prefix_imitation_comparison_20261008.json`。后续普通状态覆盖的新TRAIN采集与单列恢复后回线优化尚未启动，不把本轮pilot作为训练标签。

### 2026-10-08 learner-state retention coverage ablation

用户继续授权后登记 `docs/lite3_retention_coverage_plan_20261008.json`：新seed2267采集冻结权重10候选的普通起步首回合，64×2000上限128000；不干预动作。原冻结PPO2063在实际48D动作前观测上离线生成限幅标签，仅abs(y)<=0.30m普通域进入保持数据，整环境流id%4==0仍留作离线验证。包含失败前活动行，全部失败和padding计费；旧纠正/恢复后数组不变，原7来源不参与。新增行按组拼接、组内均匀采样，其余原初始化/seed2247/2000更新/lr1e-5/保持权重10/批512不变，只用最终checkpoint。标签是原策略保持参考，不能预先宣称为已验证纠正。

pilot仍用2251候选两条件上限256000，复用付费base需初态/延迟联合哈希一致；原成功/横偏门槛不变，全部通过才触发原7来源验证上限256000。本轮总新物理上限640000，监督更新与离线教师查询不新增物理；之前累计charged26579456/confirmed26579392/uncertain64。不得按结果更换seed/checkpoint或转用pilot轨迹作训练标签。

新采集64/64完成、87488控制步，轨迹与CPU回放审计通过。实际首回合活动82913行全部在±30cm内，无按域排除行；追加普通保持拟合62208行、离线验证20705行（原恢复纠正及恢复后数组不变）。全部新标签经独立原PPO查询重建通过。锁定同2000更新监督协议及2251无前缀pilot协议，开始数据覆盖消融拟合。

状态覆盖消融完成2000监督更新，最终SHA6a470cec…，训练/数据审计通过。eval2251候选两臂173760步，普通64/64、rough RMS7.91cm、峰值18.03cm，配对原PPO为64/64、8.07/19.27cm；普通三项门槛全部通过。case26自主恢复24/24，各延迟组7/7、7/7、7/7、3/3，RMS59.73cm，但峰值71.62cm仍超过70cm，比权重10的71.30cm略高。六项门槛通过五项，完整pilot失败，未触发原7来源验证或新FlowDAgger集成，不以72cm新阈值替代原判定。

本轮新增采集87488+pilot173760=261248控制步，监督2000更新及离线查询/汇总新物理0，uncertain0；累计charged26840704/confirmed26840640/历史uncertain64，旧完整PPO系列11900884另列。36项回归及全部标签、训练、轨迹/CPU动作回放审计通过。保留最终模型、失败判定和原基线；报告 `docs/lite3-retention-coverage-result-2026-10-08.md`。下一项仅完成设计：保护普通崎岖行走/恢复前段动作的PPO回线工程，现有flat_anchor不足以直接承担该范围；尚未启动新PPO工程、训练或有条件验证。

### 2026-10-08 recovery return-to-route PPO pair

用户继续后登记 `docs/lite3_recovery_return_plan_20261008.json`。先工程smoke seed2273、64×400=25600上限，冻结当前6a470cec… Actor加逐步交替±0.01动作探针，只验证奖励目标/掩码/自动重置时序与实际奖励贡献。扩展保持约束到所有普通状态和恢复前12控制步；训练期12固定case26槽，保留16flat+24rough101+12rough107普通槽，评测仍case26全部24槽。

工程通过后，固定learner2281、64×32×128=262144步/臂，两个PPO臂同监督Actor初始化、fresh critic/Adam、std0.05、lr3e-6、5epochs×4minibatches、entropy0、无镜像损失、保持惩罚系数100；仅横偏奖励系数control=-4、primary trial=-8不同。只取最终128更新模型。每臂eval2251两条件最多256000；主臂全部通过原pilot门槛才触发原7来源验证256000，不因对照结果更好而替换主臂。总上限1317888，未触发预算不自动运行；之前累计charged26840704/confirmed26840640/uncertain64。原7来源、pilot轨迹不参与训练，软奖励约束不声称执行端强制保护。

工程smoke完成25600步并独立审计，普通/恢复前段/尾段掩码与非零奖励贡献、自动重置年龄均通过。监督Actor缺探索方差的问题在正式训练前修复：显式allow_mean_only，仅新建log_std=.05，均值/critic/optimizer/RNG保持初始化契约；32项相关回归通过。两个正式PPO训练与pilot协议已锁定，按既定128更新/臂运行，不更改训练奖励或门槛。

两组PPO/评测均完成：对照普通64/64、RMS12.96cm/峰值28.77cm，case26为24/24、峰值74.52cm；主组普通63/64（rough101/env30姿态终止）、13.39/35.97cm，case26为24/24、峰值75.94cm。两组均退化，原7来源验证未触发。首个32步物理rollout逐字段相等，源快照相同；训练/动作保持/评测审计通过，32项回归通过。最终对照99b55642…、主组16388461…保留，不替换原监督候选6a470cec…。本阶段新增920000、uncertain0；累计charged27760704/confirmed27760640/uncertain64，旧完整PPO11900884另列。结果 `docs/lite3-recovery-return-ppo-result-2026-10-08.md`；下一项改用直接可微均值保持损失，另行登记，不回流pilot。

### 2026-10-09 direct mean-action replay loss

用户继续后登记 `docs/lite3_mean_anchor_plan_20261009.json`：从原监督候选6a470cec…重新初始化，保留上一主组learner2281/std0.05/lr3e-6/128PPO轮/奖励保持100/横偏−8，仅新增直接均值MSE系数100。每次PPO小批另取128correction+128flat+128rough101+128rough107 TRAIN观测，目标由冻结初始化Actor给出原始均值；不使用post_recovery组、离线验证行或pilot轨迹。私有CPU采样seed2293、记录索引流SHA；参考网络构造恢复主RNG，保证首rollout可配对。单次合并损失反传、仍2560次Adam步，零系数与原PPO参数/损失/RNG完全一致的测试通过。

本轮训练262144、pilot最多256000、只有全部原门槛通过才另跑原7来源最多256000，总上限774144，原付费主组复用不重复计费。此前累计charged27760704/confirmed27760640/uncertain64。不从失败PPO checkpoint续训，不换seed/中途检查点，不放宽70cm恢复峰值门槛。

直接均值保持组完成128PPO轮/2560Adam步，最终SHA88e8b292…；1310720回放抽样索引流独立重建通过，原验证行零使用，首32步物理与上一主组完全一致。eval2251普通64/64、RMS8.43cm/峰值19.04cm；case26恢复24/24、RMS59.49cm、峰值70.62cm。原六门槛通过五项，恢复峰值仍超70cm，原7来源验证未触发。普通均值偏差MSE降至对照约13%，但仍不能称作精密回线或稳定泛化。新增训练262144+评测178176=440320、uncertain0；累计charged28201024/confirmed28200960/历史uncertain64，旧完整PPO11900884另列。36项相关测试和训练/奖励保持/直接更新/冻结轨迹审计通过；训练恢复144成功/29失败完整保留。报告 `docs/lite3-mean-anchor-result-2026-10-09.md`。下一项仅拟定128→256更新长度消融，其余参数不变，尚未启动。

### 2026-10-09 fixed256-update mean-anchor length ablation

用户继续后登记 `docs/lite3_mean_anchor_long_plan_20261009.json`：只将PPO最终预算128→256更新，训练524288步；保持原监督初始化6a470cec…、learner2281、std0.05、lr3e-6、均值回放100/seed2293/原TRAIN数据、奖励保持100/横偏−8不变。不从128轮checkpoint续训，重新初始化actor/critic/optimizer，只取最终256轮。pilot最多256000，全部原门槛通过才触发原7来源最多256000，总上限1036288。先前charged28201024/confirmed28200960/历史uncertain64；不得因仅差0.62cm而调整70cm阈值，也不得挑中间checkpoint补过。

256轮长度消融完成并审计：524288训练控制、5120Adam步、2621440回放查询，最终SHA3893dddf…。前128轮policy/critic观测、根状态、请求/执行动作与旧128轮逐元素完全一致。eval2251普通64/64、RMS7.04cm/峰值15.84cm；case26恢复24/24、RMS59.22cm、峰值70.46cm。五门槛通过，恢复峰值仍超70cm，未触发原7来源验证。新增评测179200，阶段合计703488、uncertain0；累计charged28904512/confirmed28904448/历史uncertain64，旧PPO11900884另列。运行时代码未改，沿用前轮36项回归；只读审计对旧结果兼容核验及全部新轨迹审计通过。报告 `docs/lite3-mean-anchor-long-result-2026-10-09.md`。下一项拟检查correction组用成功TRAIN示范目标替代初始化Actor保持目标，普通保持不动；尚未实现/启动，不放宽70cm阈值。

### 2026-10-09 correction replay demonstration-target ablation

用户继续后登记 `docs/lite3_demo_target_plan_20261009.json`：只把correction回放目标由初始化Actor均值换成既有新TRAIN示范采集的执行动作，216条TRAIN状态不变、target数组SHA1849d787…；普通回放仍用冻结初始化Actor，不用post_recovery或验证行。原在线奖励保持100（含恢复前12步）仍保留，目标间可能存在竞争，不同时改系数。保持原6a470cec…初始化、learner2281/std0.05/lr3e-6/256轮/均值系数100/回放seed2293/横偏−8，最终checkpoint规则不变。40项相关测试通过，新增对齐/隔离与旧PPO一致性检查。

训练524288、pilot上限256000、仅所有原门槛通过才触发原7来源256000，总上限1036288；此前charged28904512/confirmed28904448/历史uncertain64。目标来自已审计训练数据，禁止回流pilot或原7验证来源，不放宽70cm恢复峰值。

纠正前一条测试状态：首轮实际39通过、1失败，代理未先检查返回结果就发起训练，随后发现并发送停止信号。失败用例揭示参考输出若与输入共享内存，标签替换可误改输入；修复为替换前clone。实际FrozenPPOExpert输出独立张量，但通用契约必须先修复。原训练目录完整保留：51次PPO更新、104576控制步（含末段2个向量步），已安全停止、账本无pending、uncertain0，无最终checkpoint；独立按轨迹核账通过。该工程误启动全部计费，不作为策略试验结果或免费重试，后续重跑另登记预算。

修复后的40项测试实际全部通过，测试日志固化为 `docs/lite3_demo_target_tests_20261009.txt`。重试补充登记 `docs/lite3_demo_target_retry_addendum_20261009.json`，中止成本104576完整计入，阶段上限1036288→1140864；重试前累计charged29009088/confirmed29009024/历史uncertain64。重用原不可变训练协议、相同seed/初始化/数据/参数，从头运行到新目录train-retry01；新pilot协议只更换最终模型/报告路径，不作结果驱动调整。

重试已完成固定256轮、524288控制步/5120Adam步，最终SHAb256c42e…。训练恢复288成功/57失败完整保留；训练、奖励保持及直接更新三审计通过。2621440回放查询中655360次correction示范抽样逐条重建通过，首32控制rollout逐字段与旧256轮相等，抽样索引SHA仍f23e011a…。离线纠正示范MSE从对照TRAIN0.010452/验证0.009175降至0.005390/0.005490；只作诊断，不以此替代闭环门槛。新最终模型开始固定eval2251普通/恢复两条件评测，原pilot协议所指中止目录不再使用，保留为未运行协议。

2026-10-09纠正示范目标消融完成：只把correction回放目标换为成功TRAIN执行动作，其余256轮PPO设置不变；普通64/64、RMS6.32cm，但峰值22.16cm超原19.27cm；case26恢复24/24、RMS62.44cm/峰值75.94cm，超70cm。六门槛过四项，新候选不采用，原7来源验证及新FlowDAgger集成未触发。示范验证MSE下降40.2%，恢复后状态的均值变化却为旧对照3.12倍，下一项先查尾段保持不足；新消融尚未启动。已修复参考输出view导致标签替换误改输入的问题，40项测试及全部审计通过。代理未检查首轮失败测试就提前启动的中止成本104576也完整计入；正式训练524288+pilot179200，阶段总808064。累计charged29712576/confirmed29712512/历史uncertain64。见[目标消融结果、两张图及错误记录](lite3-demo-target-result-2026-10-09.md)。

### 2026-10-09 additive post-recovery mean preservation

用户继续后，先仅用TRAIN数据离线诊断：已有5681条post_recovery状态覆盖横偏43.82–68.43cm，此前PPO回放未使用；示范目标候选在该TRAIN域相对初始化均值MSE0.000581，对照0.000187。尾段保持与旧辅助项梯度余弦0.817，与纠正项0.055，未见强烈局部对抗，但这不是闭环收益证明。

登记 `docs/lite3_tail_anchor_plan_20261009.json`：只加25×post_recovery均值MSE，每小批独立CPU seed2309采128条原TRAIN状态，目标为冻结初始化Actor原始均值，不读取尾段存储动作或验证数组。原四组批量/系数100/分母/seed2293完全不动；同6a470cec…初始化、learner2281/std0.05/lr3e-6/256轮、奖励保持100/横偏−8。仍5120Adam步；原2621440回放+新655360尾段查询，新增回放本身物理0。

42项测试实际通过，新增零尾段系数完整参数/RNG一致和非零尾段梯度严格相加检查。首测41过1失败来自测试误选Actor首参数log_std，显式选mlp权重后通过；未提前启动任何物理训练。预算训练524288+pilot上限256000+条件验证上限256000，总1036288；此前charged29712576/confirmed29712512/历史uncertain64。只有全部原门槛通过才触发原7来源验证；不续训失败checkpoint、不换seed、不放宽70cm。

固定256轮完成：524288控制步/5120Adam步，最终SHAb714dea5…。训练恢复288成功/60失败、0截止完整保留；原四组2621440回放与新增655360尾段回放逐条索引重建通过，原纠正标签流不变，首32控制与旧示范目标对照逐字段相同。训练/奖励/直接更新三审计通过。离线验证尾段均值变化MSE0.000572→0.000313，下降约45%；示范验证拟合MSE0.005490→0.005584略增。开始固定eval2251两条件冻结评测，尚不据离线损失宣称达标。

2026-10-09尾段保持消融完成：先核对5681条TRAIN尾段覆盖和辅助梯度，再只加25×尾段均值MSE，原四组系数/分母/采样不变。固定256轮及eval2251结果：普通64/64，但RMS9.89cm/峰值33.10cm；TRAIN case26恢复24/24、RMS61.54cm/峰值73.68cm。原六门槛仅过三项，不采用新候选，不触发原7来源验证或新FlowDAgger集成。离线尾段误差降低45%、实际普通访问状态的均值误差也降低18%，但普通路线退化；env44约−1.18cm/s世界横移在26.58秒累积至−33.10cm，提示应检查持续方向偏差与路线奖励，而非继续堆叠保持损失。42项测试及全部审计通过，无物理重试；新增524288+183936=708224，累计charged30420800/confirmed30420736/历史uncertain64。后续奖励/课程消融尚未启动。见[完整报告与两张图](lite3-tail-anchor-result-2026-10-09.md)。

### 2026-10-09 centerline-directed heading reward

用户继续后，仅用既有新TRAIN采集（env_id%4==0仍排除）重建路线几何：普通横偏成本均值0.04899，对应纯机身项0.00773，足端支配100%；原航向成本均值0.002664。偏离>5cm的TRAIN普通行中83.96%航向有向外分量，零航向目标未直接表达向路线回转。这些是离线描述，不是新奖励收益。

登记 `docs/lite3_route_heading_plan_20261009.json`：唯一学习改动为航向目标0→clip(atan2(-y,1m),±20°)，仍权重−1；原足端/机身横偏惩罚−8、普通/前段在线保持100、四组均值100和尾段25均不变。同6a初始化、learner2281、lr3e-6、std0.05、256轮，只取最终checkpoint，不接续失败模型。输入48D/固定命令不变，没有推理期外部转向器。53项测试已实际通过，覆盖左右反射、向内目标、转角边界、旧默认精确一致及重置前快照。

先seed2311工程smoke64×400=25600，冻结Actor加±0.01探针，独立核对实际奖励和重置时序后才锁定训练协议。后续训练524288、pilot最多256000、全部原门槛通过才触发原7来源最多256000，阶段上限1061888。此前charged30420800/confirmed30420736/历史uncertain64。GPU0有其他用户任务，使用空闲同型号RTX3090 GPU1，要求首物理rollout与GPU0对照相同，保留任何差异。

工程smoke完成25600控制、uncertain0，新航向目标/实际dt奖励/重置前状态及旧动作保持分别独立审计通过。正式训练协议和pilot协议已冻结，仍按既定256轮、GPU1运行，原门槛不变。工程检查不作为模型成绩，也不回流标签。

固定256轮训练完成524288控制、5120Adam步，最终SHAc7c1d64d…。训练恢复288成功/53失败、0截止保留；训练、原保持、航向几何/重置时序、双回放四审计通过。GPU1首32步物理与GPU0旧尾段保持对照逐元素相同，总奖励最大差0.00475049、仅由航向变化解释的残差最大1.43e−8；两回放索引及纠正目标流完全复现。新冻结eval2251两条件评测已启动，尚不据训练损失判定达标。

2026-10-09回线航向目标消融完成：只将航向目标0改为clip(atan2(-y,1m),±20°)，权重仍−1，其余尾段保持PPO设置不变。53项测试、25600工程smoke及训练/实际奖励/回放/评测审计通过；GPU0有其他用户任务，使用GPU1，首32步物理逐元素匹配GPU0旧对照，奖励差由航向变化解释（最大残差1.43e−8）。固定256轮/eval2251普通64/64、RMS7.15cm/峰值32.50cm，TRAIN case26恢复24/24、RMS61.26cm/峰值73.14cm。六门槛过四项，候选未晋升、原7来源验证未触发。普通48槽中42个RMS改善、41个峰值改善，但env29从16.47→32.50cm成为新最坏；恢复结束平均横偏60.15→57.72cm仍大。下一项拟单独检验航向权重−1→−4，尚未启动。新增25600+524288+183936=733824，累计charged31154624/confirmed31154560/历史uncertain64。见[完整结果、方法图及对比图](lite3-route-heading-result-2026-10-09.md)。

### 2026-10-09 centerline heading weight-4 ablation

用户继续后登记 `docs/lite3_heading_weight4_plan_20261009.json`：只将回线航向权重−1→−4，1m/20°几何、原横偏−8、在线保持100、均值回放100及尾段25全部不变。同6a监督初始化、learner2281、std0.05/lr3e-6/256轮，不续训c7c1d64d…，只取最终模型。运行时代码未修改，已逐文件确认与前轮完成训练源哈希相同，沿用实际通过的53项测试；后续扩展只读审计另做兼容检查。

先在同一空闲GPU1重跑配对工程smoke，固定原seed2311/64×400/冻结Actor/±0.01探针；应物理完全相同、航向实际奖励4倍，全部25600步重新计费。之后训练524288，pilot上限256000，只有所有原门槛通过才触发原7来源256000，总上限1061888。之前charged31154624/confirmed31154560/历史uncertain64；不换种子/中途checkpoint，不放宽70cm或普通精度门槛。

配对工程完成并通过航向/动作保持/全轨迹配对三审计：新25600控制，所有冻结物理字段逐元素相同，航向实际奖励为4倍，总奖励差只由该项解释。新的完整运行全部计费，未将复现扣为零成本。正式训练与pilot协议已冻结，开始既定256轮。

固定256轮完成524288控制/5120Adam步，最终SHAb0432834…。训练恢复287成功/57失败、0截止保留；训练、原动作保持、航向奖励几何、双回放四审计通过。第一32控制的物理和航向目标与权重−1对照逐元素相同，奖励差只由权重变化解释，最大残差1.49e−8。新冻结eval2251两条件开始；所有3276800次回放查询及两索引流、纠正目标流符合原协议。

2026-10-09航向权重−1→−4消融完成：运行时代码未改，核对源哈希后沿用53项测试。配对工程25600步所有物理字段相同、航向奖励4倍；固定256轮及eval2251完成，普通64/64但RMS8.72cm/峰值33.71cm，TRAIN case26恢复24/24、RMS62.54cm/峰值79.55cm、结束平均横偏61.64cm。原六门槛只过三项，不采用新候选、不触发原7来源验证。普通48槽仅13个RMS、16个峰值改善，env62成为新最坏；不再增加航向权重。下一项拟以权重−1为对照，单独将correction示范目标改回冻结初始化均值，检验目标与回线奖励组合；尚未启动。新增25600+524288+180224=730112，累计charged31884736/confirmed31884672/历史uncertain64。见[完整负结果与图](lite3-heading-weight4-result-2026-10-09.md)。

### 2026-10-09 self-mean correction targets with centerline heading

用户继续后登记 `docs/lite3_self_target_heading_plan_20261009.json`：以权重−1的c7c1d64d…配方为对照，唯一变化为correction目标从示范执行动作回到冻结初始化Actor原始均值。原216条TRAIN观测、两采样流/批量、主回放100/尾段25、在线保持100、横偏−8、1m/20°航向目标及权重−1全部不变。仍6a初始化、learner2281/std0.05/lr3e-6/256轮，fresh critic/optimizer，不接续失败模型，只评测最终checkpoint。

运行时代码未改，已核验逐文件源哈希与先前完成运行相同，沿用已通过的53项测试和相同奖励工程证据；自均值路径已有实际128/256轮及测试验证，新增工程物理0。新训练524288、pilot上限256000、全部原门槛通过才触发原7来源256000，总上限1036288；此前charged31884736/confirmed31884672/历史uncertain64。普通/尾段/纠正全部回放目标来自冻结6a，不读取存储动作作优化标签，不回流pilot或原7来源。GPU1与对照相同，首32控制物理及总奖励必须完全一致。

2026-10-09自均值纠正目标＋回线奖励消融完成：以权重−1配方为对照，仅将correction目标从示范动作改为冻结6a均值，保留尾段25和其余设置。运行时代码未改，复用经源哈希核对的53项测试/奖励工程证据，新增工程0。固定256轮及eval2251完成：普通64/64、RMS7.67cm/峰值27.78cm；TRAIN case26恢复24/24、RMS61.31cm/峰值72.47cm，结束平均残差59.37cm。原六门槛过四项，候选未晋升、原7来源验证未触发。虽然全局峰值下降，普通48槽仅15个RMS/10个峰值改善，结束恢复残差增大；env39近零平均航向下仍持续侧移。首32控制所有字段含奖励逐元素相同，两回放流一致，训练优化存储动作标签0。新增524288+179776=704064，累计charged32588800/confirmed32588736/历史uncertain64。下一项拟只去掉尾段保持，尚未启动。见[完整报告与图](lite3-self-target-heading-result-2026-10-09.md)。

### 2026-10-10 remove tail preservation with fixed self-mean targets and heading

用户继续后登记 `docs/lite3_no_tail_heading_plan_20261010.json`：只移除单独尾段保持项（有效系数25→0，协议省略tail_anchor），其余四组自均值回放100/seed2293、航向−1/1m/20°、在线保持100、横偏−8均不变。仍从6a监督Actor重新初始化，learner2281/std0.05/lr3e-6/256轮，只取最终checkpoint。对照为上一轮self_target_heading，不跨旧零航向模型推断因果。

176个运行源文件哈希与完成运行相同，复用已实际通过的53项测试及无尾段/奖励工程证据，本轮新增工程0。训练524288、pilot上限256000、只有全部原门槛通过才触发原7来源上限256000，总1036288。此前charged32588800/confirmed32588736/历史uncertain64；不回流pilot或原7来源，不换seed/中途checkpoint，不放宽普通精度或70cm。主回放2621440次、尾段0次；要求首32控制所有字段含奖励和主采样流与对照完全一致。

固定256轮训练完成，524288控制、5120Adam更新、2621440主回放查询、尾段查询0；最终SHAf9cb2e41…。训练恢复288成功/72失败/0截止全部保留。训练、实际动作保持、航向几何和更新四审计通过；首32控制全部26字段含奖励逐元素相同，主回放索引流完全复现，优化存储动作标签0。开始固定eval2251普通/恢复两条件评测，原门槛不变。

2026-10-10去掉尾段保持消融完成：仅移除单独尾段25，四组自均值100及回线航向−1等不变。同6a初始化/learner2281/最终256轮，524288训练控制、5120Adam更新、2621440主回放查询、尾段0；首32控制全部26字段含奖励相同，主抽样流相同。普通64/64、RMS7.15cm/峰值18.88cm，TRAIN case26恢复24/24、RMS59.35cm/峰值70.71cm、结束平均横偏54.85cm。六门槛过五项，恢复峰值仍超70cm，不晋升、不触发原7来源验证。普通48槽28个RMS/37个峰值改善，恢复24槽全部RMS/峰值改善；更接近初始化动作并不保证更好路线。只读TRAIN速度诊断显示1–2秒仍向外约10.18cm/s，下一项先校准世界横向回移速度奖励，再登记单项消融，尚未实施或启动。运行源176文件未改，复用53项测试，全部训练/评测审计通过。新增524288+176896=701184，累计charged33289984/confirmed33289920/历史uncertain64，旧PPO11900884另列。见[完整结果和图](lite3-no-tail-heading-result-2026-10-10.md)。
