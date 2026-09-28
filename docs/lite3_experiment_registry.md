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
- 状态：方案已登记，尚未启动本方案新一轮实验。
- 当前硬门槛：先完成 runner parity、SAC 逐步轨迹、观察/终止/预算契约；G0 未通过前不得把适应结果写成学习收益。
- 路线：A LoRA-PPO、B Residual SAC、C 结构化低维修正，独立晋级；不预设主算法。
- 主 session 预算：40,000 个实际控制 transition；最终冻结评估 10 个共同扰动起点；平地保持 5 个共同起点。所有预热、候选筛选、探针和更新样本均计入。
- G3 候选：G0 通过且路线具备开发晋级证据后，最多 8 个失败池地形 × 3 次独立 session × 3 臂；具体地形和代码协议哈希锁定后再启动。
- 立即动作：P0/P1/P2 工程包；禁止直接重启高度扫描 teacher、重训 diffusion base 或启动封存实验。
