# Lite3 现有平地 PPO 训练链审计

审计日期：2026-09-06。依据保存的训练配置、checkpoint 文件和 manifest，不以旧说明文档中的 rough 实验描述代替当前模型证据。

## 结论

当前 `model_1998.pt` 的已保存训练链支持纯平地训练来源，可以继续作为冻结 PPO 使用。无需仅因训练来源问题安排从头重训。基础配置参考本地 `unitree_rl_gym` Go2 的速度跟踪训练结构，实际在本项目 Isaac Lab / RSL-RL 中针对 Lite3 训练。

## 训练链

训练目录根路径：`/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/`。

| 训练目录 | 核对 checkpoint | 恢复来源 | 保存的地形配置 |
|---|---|---|---|
| `2026-08-05_11-04-10_lite3_height_balance_deployable_seed42_20260805` | `model_300.pt` | `resume=false`，算法为 PPO | `plane`，generator 为 null |
| `2026-08-05_11-38-22_lite3_height_balance_resume300_seed42_20260805` | `model_1499.pt` | 上一目录的 `model_300.pt` | `plane`，generator 为 null |
| `2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805` | `model_1998.pt` | 上一目录的 `model_1499.pt` | `plane`，generator 为 null |

三份 checkpoint 均存在，三轮 seed 均为 42；每轮的 `params/env.yaml` 中 policy 的 `base_lin_vel`、`motion_phase`、`motion_reference`、`height_scan` 均为 null。当前训练入口只有 resume 或 Distillation 分支会加载已有权重；保存的最早配置为 PPO 且 resume=false，与从随机初始化训练一致。

文件 SHA256：

```text
model_300.pt  d7a1c02b0989251b1325dfbc8d903bee729358cc265bc2c4d8510c5d52a0d01a
model_1499.pt 1d20865c1700de2b97b348fff8639767ea5eed3d50c63785daa5b238d88d2c64
model_1998.pt 87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c
```

`artifacts/online_validation_20260805/base_manifest.json` 中最后一轮的模型、agent.yaml、env.yaml 三项哈希均与当前文件一致。该 manifest 只登记最后一轮，本审计补充了前两轮的恢复关系；这不等于对历史执行做了逐步重放。

## 已有平地数据

`artifacts/online_validation_20260805/flat_replay_model1998_formal/flat_replay_manifest.json` 已记录 100000 条平地 transition、frozen=true、falls=0、safety_invalid_rows=0。此次读取了 manifest，尚未重新计算大型 replay 文件哈希及逐条复验，正式复用前需完成这一步。

`base_flat_seed7_v3/summary.json` 记录一次 300 步、无跌倒、无安全违规、无 Actor 更新的历史平地运行。它不能替代多速度、多种子长程复测。此前反复引用的“源地形五种子平均误差 0.103”不能在未核对模型身份时直接归属于本 checkpoint。

## 下一步复测范围

1. 使用现有 model_1998，不先训练新模型；复测固定 0.1、0.3、0.5 m/s 和停走切换。
2. 每个速度先做一个短程诊断，再执行预先登记的五个平地种子，每次保存视频和 JSON。
3. 报告实际位移、前向速度误差、侧向漂移、姿态、四足支撑与摆动统计、终止原因及控制延迟。不要采用误差 0.35 m/s 作为 0.3 m/s 指令下唯一通过条件。
4. 核对 flat replay 的哈希、模型绑定、观测维度和奖励语义，确认后复用。
5. 对 residual 独立核对：v10/v24 含台阶经验，不用作无台阶经验的 residual 初始化；这不影响 PPO 本身的平地来源结论。

本次仅完成训练链审计，没有运行新的物理 rollout，也没有修改训练模型或既有实验产物。后续月计划中的“必须从头训练 PPO”和“必须重新采集平地 replay”应改为有条件任务。
