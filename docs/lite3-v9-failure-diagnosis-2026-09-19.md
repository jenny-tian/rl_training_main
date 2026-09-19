# Lite3 v9 失败诊断与 context 配对优化

## 已完成实验的证据

对 `artifacts/semester_2026/09/lora_ppo_multiepisode/seed101_source_v9_*` 三组
各 100 回合的 summary、逐步 checkpoint rows、更新日志及冻结 probe 做全量读取。
审计脚本和原始 JSON 在 `diagnosis_context_20260919/audit.py`、`audit.json`。

| 组 | 成功 | 横向越界 | 超时 | 摔倒终止 | 严重非足接触 | 接受/全部更新 | 冻结 probe 成功 |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor1 | 11 | 43 | 24 | 9 | 13 | 562/709 | 5/20 |
| anchor100 | 22 | 36 | 24 | 10 | 8 | 703/716 | 2/20 |
| std100 + anchor100 | 29 | 37 | 5 | 17 | 12 | 598/662 | 3/20 |

1. **主要瓶颈是横向偏离。** 300 回合中 116 次越界，占 238 次失败的 48.7%。
   三组运行的是旧 v7 输入结构：Actor 45 维不含累计航向/横移，Critic 48 维
   虽有瞬时机身 XY 速度和时间，但没有累计横移。奖励与通关资格却依赖赛道横向位置。
   这是可观测性缺口；它是否是主要因果因素仍需配对实验，而非仅凭失败统计断言。
   上次已提交的 `--proprioceptive-context` 并未被这三组旧代码快照使用。
2. **增大探索有明显代价。** std100 每步至少一个动作分量被投影；每回合逐步
   `max(abs(requested-executed))` 的均值再对回合平均为 0.18279，
   std030 两组分别为 0.04232、0.04224。std100 的 requested 最大残差均值 0.19850，
   executed 为 0.08661。高投影率不是 PPO likelihood 错误的证据
   （当前按 requested-action 固定安全映射建模），但说明探索与执行差异较大。
   std100 超时下降而摔倒增加；原组同时改变了学习率，不能把因果全归于 std。
3. **强 anchor 没有普遍阻止学习。** anchor100 更新接受率 98.2%；anchor1
   有 145 次因 `anchor_mse` 超限被拒绝，另 2 次因 KL。继续放松硬限额没有依据。
4. **成功 checkpoint 需要保留，不能用末回合代替。** std100 的第 95 回合
   冻结 probe 通过，开发地形 103/107/113 通过、109 失败（3/4）；
   第 100 回合 probe 失败。优化从第 95 回合的确切文件出发，保留恢复分支。

## 本轮优化

复用已实现、已通过单元测试的 context head 和 checkpoint migration，先做 20 回合开发试验。
新增 `scripts/reinforcement_learning/online_adaptation/run_lora_ppo_context_arm.py`
使训练、冻结验证、三种子物理平地保持检查连续执行并持有 GPU 锁。

共同初值为 `seed101_source_v9_std100_anchor100_100ep_20260919_0532/episodes/episode-0095/latest.pt`，
SHA256 `f6e34b96b19d455008b3416002e86d251113638b95b9f318fa2f95fd7e4cc80e`。
两组共同使用 std=0.03、actor lr=1.5e-4、critic lr=3e-4、anchor coef=100、
source-flat MSE 上限 1e-4、KL 上限 0.01、地形种子 101、模拟种子从 7 开始递增。
相同安全限制、2000 步截止、每 5 回合冻结 probe 和原稳定验收规则不变。

| 组 | GPU | 唯一区别 |
|---|---:|---|
| control | 1 | 旧 45D Actor / 48D Critic |
| context | 3 | 开启本体航向与横移 context，48D Actor / 51D Critic |

与旧 std100 的比较同时包含降噪和降学习率，不能分离两者作用；本轮两组间比较
专门检验 context。新增参数初始化消耗 RNG，匹配种子不保证逐步探索轨迹一致。
仅一个训练种子序列，也不足以做统计显著性结论。

训练不逐回合录视频，冻结地形 probe 仍录视频。每组结束后，对 supervisor
选择的 `final_checkpoint`（可能仍是初始恢复模型）做平地 seed 7/8/9 配对检查。
同时保存其路径和 SHA，不能把恢复模型的保持能力归因给训练末回合。
平地门槛沿用原计划：2000 步、1750 稳态样本、无摔倒/违规/reset、无更新，
稳态世界 X 位移至少 8.4 m；X MAE 和 XY L2 误差增量分别不超过
`max(0.005 m/s, 5% * baseline)`。保留 XY 各误差原值，失败不做自动晋升。
源 anchor 的零 context 约束不能替代这些非零累计 context 的真实平地测试。

实验从 Git 快照执行，运行目录与命令写入 `arm_plan.json`，失败写入
`arm_status.json`。程序不会复用已有 `arm_plan.json` 的目录覆盖数据。
配对运行位于 `artifacts/semester_2026/09/lora_ppo_multiepisode/context_pair_v10_20260919/`。

## 验证边界

本轮重跑 context/里程计/worker/监督器相关回归 115 项通过；新增平地验收门槛
测试覆盖摔倒、重置、短轨迹、仍在学习、位移不足及误差超限。
真实 context 优化效果以本轮输出为准，不能把 CPU 测试或启动成功当作地形学习达标。
