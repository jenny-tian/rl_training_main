# 2026 年 9 月执行状态

更新时间：2026-09-07。

## 已完成

- 已确认 `model_1998.pt` 的训练链为 `model_300 → model_1499 → model_1998`，三轮保存配置均为 `plane` 且 `terrain_generator=null`，最早一轮 `resume=false`。
- 基础模型、agent.yaml、env.yaml 的 SHA256 与 `artifacts/online_validation_20260805/base_manifest.json` 一致。
- 已重算 `flat_replay.pt` SHA256，与 `flat_replay_manifest.json` 一致；replay manifest 的 baseline manifest SHA256 也匹配。
- replay 检查结果：100000 条 transition、frame_dim=49、history_length=25、action_dim=12、frozen=true、全部浮点字段 finite、falls=0、safety_invalid_rows=0、residual_action 全为零。
- 在线适应回归测试：`958 passed, 7 skipped`。
- 已生成 `artifacts/semester_2026/09/asset_validation.json`。

## T02 首条视频探针：已通过

此前两次失败均为执行资源问题：第一次混用了 `CUDA_VISIBLE_DEVICES=3` 与 `--device cuda:0`；第二次直接使用 `cuda:3`，但 Vulkan 在多 GPU 资源探测阶段显存不足。失败记录保留在
`artifacts/semester_2026/09/base03_seed7_short/rollout_status.json`，没有停止其他用户进程。

随后改用官方 experience，并通过 `--kit_args` 覆盖 Kit 的多 GPU 设置：

```text
--/renderer/multiGpu/enabled=false
--/renderer/multiGpu/autoEnable=false
--/renderer/multiGpu/maxGpuCount=1
```

在 GPU 3 成功完成：

| 项目 | 结果 |
|---|---|
| 任务/模式 | `FlatOnline-Deeprobotics-Lite3-v0` / `base` |
| 速度、种子、步数 | 0.3 m/s、seed 7、300 步 |
| 跌倒/重置/控制超时 | `false` / 0 / 0 |
| 安全回退/参数更新 | 0 / 0（冻结 PPO 保持不变） |
| 平均绝对速度误差 | 0.02375 m/s（前后半段） |
| 倾角 RMS | 0.01408 → 0.01404 rad |
| 地形有效比例 | 1.0 |
| 视频 | `artifacts/semester_2026/09/base03_seed7_short_kitargs_v3/videos/rl-video-step-0.mp4`，169394 bytes，5.98 s |

完整结果见：
`artifacts/semester_2026/09/base03_seed7_short_kitargs_v3/{episode_summary.json,summary.json,metrics.jsonl}`。

## 当前执行命令约束

后续保持以下条件：

```text
worktree: /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
device: cuda:3（不设置 CUDA_VISIBLE_DEVICES）
experience: 官方 /home/tjx/IsaacLab/apps/isaaclab.python.headless.rendering.kit
kit_args: renderer.multiGpu.enabled=false, autoEnable=false, maxGpuCount=1
video: enabled，每条正式 rollout 必须保存
```

T02 下一步按计划扩展到 0.1/0.3/0.5 m/s × seed 7/11/19/23/31，共 15 条轨迹，每条最多 2000 步；速度与 seed 同时写入独立输出目录。

批量执行已启动（GPU 3，串行，保持官方 rendering experience + 单 GPU Kit 参数）：

| 已完成案例 | 步数 | 跌倒 | 重置 | 通过 | 平均绝对速度误差 | 视频 |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 m/s / seed 7 | 2000 | 否 | 0 | 是 | 0.0293 / 0.0292 m/s | `flat_0p1_seed7_2000/videos/rl-video-step-0.mp4` |
| 0.1 m/s / seed 11 | 2000 | 否 | 0 | 是 | 0.0293 / 0.0292 m/s | `flat_0p1_seed11_2000/videos/rl-video-step-0.mp4` |
| 0.1 m/s / seed 19 | 2000 | 否 | 0 | 是 | 0.0298 / 0.0298 m/s | `flat_0p1_seed19_2000/videos/rl-video-step-0.mp4` |
| 0.1 m/s / seed 23 | 2000 | 否 | 0 | 是 | 0.0256 / 0.0256 m/s | `flat_0p1_seed23_2000/videos/rl-video-step-0.mp4` |
| 0.1 m/s / seed 31 | 2000 | 否 | 0 | 是 | 0.0293 / 0.0292 m/s | `flat_0p1_seed31_2000/videos/rl-video-step-0.mp4` |
| 0.3 m/s / seed 7 | 2000 | 否 | 0 | 是 | 0.0270 / 0.0268 m/s | `flat_0p3_seed7_2000/videos/rl-video-step-0.mp4` |
| 0.3 m/s / seed 11 | 2000 | 否 | 0 | 是 | 0.0270 / 0.0268 m/s | `flat_0p3_seed11_2000/videos/rl-video-step-0.mp4` |
| 0.3 m/s / seed 19 | 2000 | 否 | 0 | 是 | 0.0368 / 0.0361 m/s | `flat_0p3_seed19_2000/videos/rl-video-step-0.mp4` |
| 0.3 m/s / seed 23 | 2000 | 否 | 0 | 是 | 0.0346 / 0.0339 m/s | `flat_0p3_seed23_2000/videos/rl-video-step-0.mp4` |
| 0.3 m/s / seed 31 | 2000 | 否 | 0 | 是 | 0.0270 / 0.0268 m/s | `flat_0p3_seed31_2000/videos/rl-video-step-0.mp4` |
| 0.5 m/s / seed 7 | 1803 | 是 | 1 | 否 | 0.0465 m/s | `flat_0p5_seed7_2000/videos/rl-video-step-0.mp4` |
| 0.5 m/s / seed 11 | 1803 | 是 | 1 | 否 | 0.0465 m/s | `flat_0p5_seed11_2000/videos/rl-video-step-0.mp4` |
| 0.5 m/s / seed 19 | 1797 | 是 | 1 | 否 | 0.0347 m/s | `flat_0p5_seed19_2000/videos/rl-video-step-0.mp4` |
| 0.5 m/s / seed 23 | 1737 | 是 | 1 | 否 | 0.0437 m/s | `flat_0p5_seed23_2000/videos/rl-video-step-0.mp4` |
| 0.5 m/s / seed 31 | 1803 | 是 | 1 | 否 | 0.0465 m/s | `flat_0p5_seed31_2000/videos/rl-video-step-0.mp4` |

T02 多速度矩阵已完成 15/15 条，所有案例均有非空 MP4 视频。0.1 m/s 和 0.3 m/s 共 10/10 通过；0.5 m/s 为 0/5，通过率为 0，但这五条都运行到约 1.7--1.8k 步后才触发物理安全终止，不是启动或显存失败。

0.5 m/s 的终止原因分为两类：seed 7/11/31 为 `bad_orientation_2`（同时出现 height safety reason）；seed 19/23 为 `illegal_contact`，终端接触体为 `FL_SHANK`。所有 0.5 m/s 案例 `control_timeout_preserved_actor=false`，因此当前证据不支持“控制超时导致跌倒”的解释。视频帧数分别为 1804、1804、1798、1737、1804，与终止步数一致。

基线结论：论文主实验速度固定为 0.3 m/s；0.1 m/s 作为低速保持性测试，0.5 m/s 作为 stress test 单独报告，不立即修改或重训冻结 PPO。后续 residual 学习的验收速度先采用 0.3 m/s，并保留 0.5 m/s 失稳作为待解决问题。

## 当前阶段判断

T01 资产核验已通过，T02 多速度物理 rollout 和视频已完成。下一阶段启动 T03/T04：先验收 49 维纯本体感知 residual 接口，再从零初始化 residual SAC；不得把已有 v10/v24 的 step 经验当作新实验的“干净初始化”。

## T03/T04：已完成第一轮验收

- 49 维 proprio frame、SAC 网络、publisher、checkpoint 和 adapter 回归：`126 passed`。
- online_adaptation 完整回归：`958 passed, 7 skipped`。
- 新增初始化工具：`scripts/reinforcement_learning/online_adaptation/initialize_residual_sac.py`。
- 正式干净初始化产物：
  `artifacts/semester_2026/09/clean_residual_initializer_v1_retry/`。
- Critic 使用当前已核验的 100000 条 flat replay 预热 `1000/1000` 次；flat replay SHA256 为
  `65530abb816757bb354f490825bc783d2e1e91c011c386ef14d1cdd95218f881`。
- Actor、Critic、Actor/Critic/alpha optimizer 均为新建；Actor version 为 0，确定性零输入残差最大绝对值为 0。
- online replay 和 adaptation replay 初始大小均为 0；未读取 v10/v24 checkpoint，禁止来源已写入 `initializer.json`。
- 可直接作为后续 online worker 的 residual checkpoint：
  `clean_residual_initializer_v1_retry/residual_sac_initial.pt`。

正式 T04 产物清单：

| 文件 | 用途 |
|---|---|
| `initializer.json` | schema、来源、哈希、预热次数和空 replay 证据 |
| `residual_sac_initial.pt` | SAC updater checkpoint，供 `--residual-checkpoint` 使用 |
| `agent_initial_state.pt` | 含空 online/adaptation replay 的完整 agent 初态 |

下一步进入 T05：先用该 clean initializer 在 0.3 m/s 平地做一条带视频的 residual-frozen/online 路径烟测，再启动最多 20 回合的 2 cm 台阶闭环；不续接 v10/v24。

## T05：平地闭环烟测已通过

使用 `clean_residual_initializer_v1_retry/residual_sac_initial.pt`，0.3 m/s、seed 7、300 步、每条均保存视频：

| 模式 | 结果 | Critic 更新 | Actor 更新 | 跌倒/重置 | 产物 |
|---|---|---:|---:|---|---|
| `residual_frozen` | 通过 | 0 | 0 | 否 / 0 | `t05_residual_frozen_flat03_seed7_300/` |
| `online` | 通过 | 118 | 0 | 否 / 0 | `t05_online_flat03_seed7_300_v2/` |

online 烟测中 `actor_version=0`、`adaptation_replay_size=0`，因为前 256 步处于 novelty calibration，且本次 residual gain 为 0；Critic-only 更新和 flat replay 混合比例均被记录。在线 experiment checkpoint、视频和 `episode_summary.json` 均存在。

第一次 online 命令因缺少 `--experiment-checkpoint-out` 被 CLI 正确拒绝，没有启动仿真；修正后的 v2 目录为正式结果。

视频：

- `t05_residual_frozen_flat03_seed7_300/rl-video-step-0.mp4`
- `t05_online_flat03_seed7_300_v2/rl-video-step-0.mp4`

下一步是 T05 的开发闭环：2 cm 台阶、最多 20 回合、每回合最多 2000 步，使用同一 clean initializer；每回合保存视频和 checkpoint，不要求本月底前必须学会台阶，但要求更新、恢复、回滚和终止原因可审计。

台阶闭环结果（`t05_step2cm_geometry_v1`）：

- 前 10 回合均完成 2 cm 台阶轨迹，单回合 638--712 步，无跌倒、无回滚，每回合均有 MP4；Actor version 从 21 增至 221。
- 第 11 回合执行 2000 步并越过完成边界，但出现 4 次 `control_timeout` 安全回退，因此 `passed=false`；没有跌倒、非法接触或姿态终止。
- supervisor 按 `no_improvement` 停止，总计 11 个物理回合、8857 步、250 次 Actor 更新；没有继续耗尽 20 回合上限。
- 第 10 回合后的自动 learned-flat retention 原先因 59 维 geometry Actor 装入 49 维 flat runtime 被严格配置检查拒绝。这是评测接口问题，不是平地物理退化证据。

已修复 retention 契约：新增 `--residual-frame-dim=59`，在严格平地为 geometry Actor 提供恒为零的 10 维几何特征；基础 PPO 对照仍使用原接口。相关测试 `130 passed`。同时 curriculum CLI 默认控制超时改为 50 ms，避免视频渲染和调度抖动被误判成策略跌倒。

使用第 10 回合 checkpoint（Actor version 221）重新执行 0.3 m/s、1000 步、五种子 learned-flat retention，五条均有视频：

- 5/5 无跌倒、无重置、无安全事件；物理安全保持通过。
- 五条 second-half velocity error 均为 0.03621 m/s，tilt RMS 为 0.01723 rad。
- 相比冻结 PPO 的相应平地基线，超过预设 5% 性能退化门槛，因此 version 221 暂不能作为最终冻结策略。

下一步只筛选较早 Actor checkpoint，不继续追加台阶更新：先单 seed 比较 episode 1/5 等较早版本，选出满足平地保持门槛者，再做五种子确认。所有筛选 rollout 继续保存视频。

## T05 结果纠偏：纯本体感知为主路径

59 维 geometry 闭环保留为工程诊断，但不作为论文主结果。原因是它包含精确台阶边缘距离、高度、足端相对边缘等 10 维特征，不符合九月计划的 49 维纯本体感知主路径。严格同协议 base-only 对照也显示冻结 PPO 本身可在 669 步完成 2 cm 台阶，因此不能声称该地形由在线学习首次学会。

已新增显式 `terrain_feature_source=proprio`：复杂地形仍存在，但 residual 仅接收 49 维本体状态，不追加 geometry/height scan。接口测试 `140 passed`。

使用 clean 49 维 initializer 完成 3 回合短闭环：

| 回合 | 步数 | 完成/跌倒 | Actor version | 观测 | 视频 |
|---:|---:|---|---:|---|---|
| 1 | 696 | 完成 / 否 | 26 | 49 维 proprio | 有 |
| 2 | 704 | 完成 / 否 | 50 | 49 维 proprio | 有 |
| 3 | 681 | 完成 / 否 | 77 | 49 维 proprio | 有 |

version 77 冻结后仍在 688 步完成 2 cm 台阶，无更新和安全事件，说明 checkpoint 可复用；但平地筛选显示 version 26/50/77 的速度误差分别比同长度 base 对照退化约 12.6%/24.8%/19.2%，均未通过 5% 保持门槛。因此当前不封存任何纯本体学习版本，clean zero-residual 仍是冻结起点。

策略漂移日志显示第 1 回合 `policy_residual_mse` 从 `9.40e-7` 增至 `8.02e-5`；未配置 `max_policy_residual_mse` 是漂移门槛未生效的直接原因。下一条开发试验使用 `1e-5` 单回合门槛，并在回合后立即做冻结平地检查；该阈值仅用于开发验证，不能未经多种子校准直接作为最终冻结配置。

受控漂移试验 `t05_step2cm_proprio49_drift1e5_v1` 已完成：

- 49 维 proprio、2 cm 台阶、0.3 m/s、seed 101，689 步完成；无跌倒、无安全事件、无回滚。
- `max_policy_residual_mse=1e-5` 下接受 8 次 Actor 更新，最终 Actor version 8，最大记录 MSE `9.398e-6`，没有超限发布。
- novelty baseline 和台阶回合均保存 MP4。
- 完整回归：`961 passed, 7 skipped`。

当前工程结论：安全门控、49 维接口、checkpoint 恢复和视频审计链路均已可执行；但尚无通过五种子平地保持门槛的纯本体在线策略。下一步应围绕 `1e-5` 漂移信任域和跨回合平地 retention 联合验收，不能仅以单回合完成台阶作为策略冻结依据。

## 连续粗糙地面与草地扩展（2026-09-07）

根据阶段决策，后续主地形不再使用台阶，而采用连续相关随机粗糙地面。新增
`ContinuousRough-Deeprobotics-Lite3-v0`，其粗糙段使用共享顶点的三角网格高度场，
高度由固定 seed 的局部相关随机场生成；因此没有独立方块之间的垂直断边，且每个
seed、幅值和相关长度都可审计复现。默认扫描范围为 ±1/±2/±3/±4 cm，相关长度
0.25--0.50 m，轨道为 3 m 平地进入、0.5 m 连续过渡和 6.5 m 粗糙段，主速度
0.3 m/s，在线 residual 仍保持 49 维纯本体感知接口。

另新增 `Grass-Deeprobotics-Lite3-v0`，作为独立物理表面条件：连续起伏约 ±1 cm，
静摩擦 0.55、动摩擦 0.45、回弹 0.05，用于评估草地/湿草地上的滑移和能量损失，
不与高度粗糙度结果混报。

本轮代码完成静态编译和地形 API 单元测试文件；Isaac Sim 视频探针尚未产生正式结果，
原因是当前 shell 找不到 Isaac Sim 的 `pxr`/Kit runtime（`env_isaaclab` 能导入
`isaacsim`，但 `isaaclab.sh` 启动时缺少 Kit runtime 路径）。在 runtime 恢复后，
先运行冻结 PPO 的 ±1/±2/±3/±4 cm 三种 seed 探针并保存视频，再选择冻结成功率
约 20--40% 的难度进入 online SAC 因果组。

受控试验的 Actor version 8 随后完成冻结平地验收：0.3 m/s、1000 步、seed 7/11/19/23/31，5/5 无跌倒、无重置、无安全事件，每条均有 19.98 s 视频。与同长度 base 对照相比：

- 平均 second-half velocity error：0.02878 → 0.02757 m/s（比值 0.9582）；
- 平均 second-half tilt RMS：0.01421 → 0.01312 rad（比值 0.9236）。

因此 version 8 被标记为“开发候选冻结点”，不是论文最终策略。候选清单为
`artifacts/semester_2026/09/proprio49_drift1e5_candidate.json`，绑定 base、clean initializer、experiment checkpoint、SHA256、台阶视频和五种子平地视频。严格限制仍成立：当前 base-only 在同协议下也可完成 2 cm 台阶，本结果只证明受限在线更新和保持性，不证明地形能力提升。
