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
