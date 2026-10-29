# 多尺度粗糙 L1 单条件策略安全检查

日期：2026-10-29。结果：**本次固定用例安全通过**。

用户批准后，从干净提交 `75589cb07f72dfe6758c070963b2d9708561a632` 执行一次单环境检查。使用原平地冻结 `model_1998.pt`，没有训练、更新权重或重试。原始运行规格见 [manifest.json](manifest.json)，执行结果见 [outcome.json](outcome.json) 和 [result.json](result.json)。

## 固定条件与结果

用例 `multi_scale_rough_L1_forward_lane` 对应 shared-bank row0 / column1。geometry/state seeds 为 **10001/20030**，固定 difficulty 为 **0.6294702331215974**。轨道 12×4 m，登记局部峰谷差候选 0.02 m、空间尺度 0.2/0.4 m；单格网格此前已与实际 bank 对应列精确匹配。本次使用单格环境，未进行 shared-bank 策略运动。

GPU0 准入线为 12288 MiB 可用显存；命令 `(0.3,0,0)`，控制周期 0.02 s、物理周期 0.005 s，最多 2000 控制步，启动／总时限 120／240 秒。运行读回的几何 seed、难度、17 刚体 guard 映射及安全阈值与冻结规格一致。

| 项目 | 结果 |
| --- | --- |
| 完成步数／控制仿真时长 | 1493 步 / 29.86 s |
| 首次前进至 9 m | 第 1444 步 |
| 最终前进／连续稳定保持 | 9.313489 m / 50 步 |
| 最低地形相对安全高度 | 0.307143 m（下限 0.20 m） |
| 最大姿态 XY 范数 | 0.051734（上限 0.65） |
| 最大关节速度 | 17.300009 rad/s（上限 30 rad/s） |
| 安全拒绝／无效状态／非足端接触事件 | 0 / 0 / 0 |
| 动作投影／策略权重更新 | 0 / 0 |
| 横漂峰值／最终 yaw | 0.325888 m / −0.256339 rad |
| 平均速度误差 | 0.037860 m/s |
| 足端滑移代理计数 | 78 / 3335 接触足采样，约 2.3388% |
| 监督器用时 | 92.809928 s |
| GPU 采样峰值／退出后 GPU0 | 2750 MiB / 18 MiB |

滑移比例是接触足的世界水平速度超过 0.35 m/s 的采样占比，是运动代理指标，不是直接测量的摩擦系数。报告的 0.4 m 方窗局部峰谷差最大值为 0.02 m，这是重建网格统计，不是足端接触测量。

## 离线复核

[post_run_review.json](post_run_review.json) 记录以下检查：

- 9 个原始文件逐字节归档，225 项冻结输入哈希一致。执行清单只包含本次单回合规格，没有继承准备阶段 status、simulation 标记或多回合范围。
- 1493 条状态和 1493 条动作记录连续完整；策略动作裁剪、请求动作与实际执行动作一致，没有投影或 SafetyFilter 暂停。
- 从逐步记录重算 9 m 目标与 50 步保持条件，以及进度、安全高度、姿态、关节速度、横漂、速度误差和滑移计数，与结果汇总一致。
- 新增四元数、世界／机体速度、投影重力及完整命令记录均有限且形状正确。四元数范数最大误差 `1.04e-7`，机体速度旋转到世界后的最大误差 `2.86e-8 m/s`，投影重力最大误差 `2.98e-8`。
- actor 执行前后哈希完全相同，PPO 未初始化。GPU0 退出后无计算进程，按精确脚本参数匹配无运行残留；GPU3 的既有他人任务未被使用或操作。

本阶段没有修改生产代码，沿用冻结源码上已完成的 589 项 CPU 回归；没有为归档重复整套测试。可独立复核本次归档：

```bash
CUDA_VISIBLE_DEVICES='' /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  docs/lite3-source-policy-prequalification-rough-l1-20261029/review_run.py
```

原始日志仍有 4 条足部视觉 USD 引用缺失警告和 6 条 CUDA 设备枚举跳过警告。没有 Error、Python traceback、fatal exception、cooking failure 或超时记录；本结果不表示上述警告已被普遍解决。

## 证据范围与基线进度

这说明**平地策略已经具有通过这个轻度粗糙实例的迁移能力**，不是 Source 训练产生的新技能。该单回合不能估计整类地形通过率，也不覆盖更高难度或其他地形／状态 seed。横漂和 yaw 偏差仍是质量观察；本次没有追溯添加直线质量门槛。后续训练的改善空间仍应通过跟踪、滑移、接触及更难几何上的独立对照评估确定。

本次正常完成没有触发 manager termination 或自动 reset，所以新增终止前运动学记录的运行路径未在本回合被覆盖。日志也没有逐物理子步接触对或完整刚体动力学轨迹；不能据此宣称 shared-bank 运动隔离或完整 G3/G4 通过。

当前已知的 Flat 和本次粗糙单条件安全证据、以及其余 **13 个未运行的准备用例**，登记于 [baseline_progress.json](baseline_progress.json)。原始准备清单的 `not_run` 是准备时快照，不改写其哈希；本索引记录后续进度。历史坡地／台阶诊断继续单独保留，见[六族基线审查](../lite3-flat-ppo-six-family-review-20261029/README.md)，不与当前用例合并计算通过率。

## 下一阶段待批准规格

按此前冻结的 bank 列顺序，建议下一阶段只执行一次 **5° 上坡 L1** 单环境检查：

- `case_id=slope_L1_uphill`，bank row0 / column2。
- geometry/state seeds **10001/20060**，固定 difficulty **0.3384848514104325**。
- 12×4 m 轨道，5° 上坡、2 m 末端平段；同一冻结 actor 和 `(0.3,0,0)` 命令。
- GPU0 至少 12288 MiB 可用显存；2000 步、120／240 秒时限；安全前进 9 m + 稳定 50 步；失败保留证据并停止，不自动重试、换种子或更新策略。

此前旧 10 m 上坡诊断的横漂越界结果不被改写。本提案是早已登记的当前 12 m bank 第 2 列条件，不是依据本次结果临时改种子。[next_stage.json](next_stage.json) 保存具体规格，[next_case_review.json](next_case_review.json) 记录 225 项输入及单回合清单离线核对；两者均不授予执行授权。

以下命令在下一阶段获批后，从本归档提交后的干净工作树执行：

```bash
PYTHONPATH=source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_policy_prequalification.py \
  --case-id slope_L1_uphill \
  --run-dir /tmp/lite3_source_policy_prequalification_slope_L1_uphill_bank_20261029
```
