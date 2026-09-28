# Lite3 实验登记表

依据：执行计划 v2.1 及 `lite3-plan-audit-2026-09-28.md`。
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
- 状态：已登记，未启动。
