# Flat 横漂审查与后续检查离线准备

日期：2026-10-29。状态：`offline_review_passed_runtime_not_run`。

本阶段完成已批准的离线审查和准备，没有启动 GPU 仿真、重试实测或初始化 PPO。589 项 CPU 回归通过；15 个单地形网格与 shared-bank 对应列精确匹配，其中剩余 14 列已冻结为候选单回合规格。下一阶段仅提议一次新版 **两环境 shared-bank G2** 检查，等待用户按此前“阶段完成后汇报并确认”的约定批准。

## 1. Flat 横漂的运动学解释

输入为[已完成的单条件 Flat 归档](../lite3-source-policy-prequalification-flat-20261029/README.md)，未重放仿真，未修改原始证据。该回合 1514 步前进 9.295170 m，横漂峰值 0.719969 m；它仍只通过已登记的单条件安全标准，不能据此认定直线跟踪质量合格。

对 0.02–30.28 s 的 1514 个控制采样作梯形积分，用 yaw 平面近似 `vy_world ≈ vx_body·sin(yaw) + vy_body·cos(yaw)` 得到：

| 项目 | 数值 |
| --- | ---: |
| 实测世界 Y 位移（末采样减首采样） | −0.720003811 m |
| 机体前向速度经航向旋转的积分贡献 | −1.395832897 m |
| 机体横向速度的积分贡献 | +0.676557571 m |
| 两项合计与实测的最终残差 | −0.000728485 m |
| 全轨迹最大绝对重建残差 | 0.001492819 m |
| yaw 变化 | −0.286440411 rad（−16.411827°） |
| 平均机体 vx / vy | 0.309010486 / 0.022658972 m/s |
| 平均机体 wz | −0.010768350 rad/s |

航向逐渐向负方向偏转，使前向运动产生较大的负 Y 分量；正机体横向速度抵消了其中一部分。两项基本解释了实测横漂。平均机体 wz 的积分为 −0.326300953 rad，**机体角速度 Z 不是 Euler yaw 的导数**，不能将两者混为一谈。

冻结命令为机体系 `(0.3, 0, 0)`，`heading_command=False`；actor 观测没有绝对航向和横向位置。该控制配置允许小的转向／侧向偏差随时间累积，但本次分析没有区分策略偏差、接触动力学、材料或初始条件的因果作用。旧日志不含普通采样的完整四元数和世界速度，重建使用 yaw 平面近似及 50 Hz 积分，因此不能用亚毫米最终残差声称获得精确三维运动模型。本阶段没有修改策略或引入航向控制器，也没有追溯修改通过标准。

数值见 [drift_analysis.json](drift_analysis.json)，图见 [PNG](drift_analysis.png) / [PDF](drift_analysis.pdf)，分析入口为 [source_flat_drift_audit.py](../../scripts/tools/source_flat_drift_audit.py)。

## 2. 地形实例与剩余用例

shared-bank 使用 seed **10001**、`1×15` 等权列、难度范围 `(0,1)`。实际 Isaac Lab 生成循环为每列抽取一个 difficulty，并把子地形 seed 设为 generator seed；Source 再由该 seed、地形族和 difficulty 派生随机几何。因此历史登记的单格 seed 与默认 difficulty=0，不能直接代表该 bank 列的实际粗糙网格。

[source_bank_policy_cases.py](../../scripts/tools/source_bank_policy_cases.py) 在 CPU 上运行已安装生成器的真实 AST 生成循环，再通过实际 `source_subterrain_from_cell` 构造单格配置。仅替换 Isaac 配置对象、缓存／USD 等非几何接口。固定单格 seed=10001、难度范围 `(d,d)` 后，15/15 列的变换前顶点、三角面、面颜色／材料编码及局部 origin 的精确字节哈希相同。详见 [geometry_equivalence.json](geometry_equivalence.json)。此检查不包含 PhysX cooking、接触或运动。

历史 Flat 的 seed10000/difficulty0 网格也与 bank 第 0 列精确相同；Flat 不列入新的默认执行候选。剩余 14 列按 bank column 1–14 固定顺序登记，状态均为 `not_run`，初始状态 seed 使用原始登记范围的第一项。具体 ID、seed、完整难度值见 [用例矩阵](case_matrix.md)，权威运行输入见 [preparation.json](preparation.json)。

几何 seed 的变更依据是**运行前对齐实际 bank 实例**，不是看结果后换 seed 或重试。每次只允许显式选择一个 `--case-id`；没有自动 sweep。单回合安全通过不能替代 shared-bank 运动隔离证据，也不能支持未测几何／状态 seed 的泛化结论。

## 3. 运行规格和记录修正

- 新单格入口固定 `qualification_geometry_difficulty`，环境和指标几何使用同一值；历史未指定该字段的扫描仍保持 difficulty=0。
- supervisor 使用显式单回合 run spec，不再展开准备清单。因此旧 raw manifest 中沿用的 `simulation_run=false`、准备阶段 status 和 5 回合 seed 范围不会出现在新执行规格中；实际结果写入 `result.json` / `outcome.json`。原始 Flat 归档不改写。
- 每个控制采样补充完整四元数、世界／机体速度、投影重力、三维命令以及采样阶段。发生自动 reset 时，回调先保存终止状态；日志不会将终止采样与 reset 后速度混配。
- 冻结 checkpoint、SafetyFilter、终止阈值、命令、9 m + 连续稳定 50 步标准保持既有设定。候选单格仍为 GPU0、至少 12288 MiB 可用显存、1 环境、2000 控制步、120/240 s 时限，无自动重试或权重更新。

以上是离线代码和用例准备。新日志字段与固定地形入口尚未经过新的 GPU 实测。

## 4. 新版 shared-bank G1/G2 复验方案

历史 shared-bank G2 绑定的 `a34b1b7` 之后，Source 基座 height scanner 周期从 0.02 s 改为 0.005 s。历史结果继续保留，但新配置须重新获得启动、接触和 reset 运行证据。命令及固定预算见 [shared_bank_recheck.json](shared_bank_recheck.json)。

拟申请的下一阶段：从本准备阶段提交后的干净工作树，仅执行一次两环境检查并离线复核、归档其结果：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --reset-lifecycle --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_g2_2env_height_recheck_20261029
```

从仓库根工作目录执行。启动前检查干净 revision、子模块、依赖／资产哈希、GPU0 进程和资源；输出目录必须不存在。两环境 supervisor 实际显存门槛为 **8192 MiB**，八环境为 **12288 MiB**。不得把单格策略入口的 12288 MiB 门槛误称为两环境 G2 的实现。

| 规模 | 启动接触探针 | 直接 reset | 自动 timeout | 总票数 / reset 批数 |
| --- | --- | --- | --- | --- |
| 2 环境（下一项待批准） | 20 个零动作控制步 / 80 子步 | 120 次 / 90 批 | env0，随后一个恢复步 | 123 / 91 |
| 8 环境（2 环境通过后另行批准） | 20 个零动作控制步 / 80 子步 | 120 次 / 32 批 | env7，随后一个恢复步 | 129 / 33 |

两种规模各有 2 个生命周期控制步（timeout 步与恢复步），不计入前面的 20 步接触探针。应核对实际地面接触阳性证据、跨机器人接触为零、扫描有限且归属正确、票序与分配计数、未 reset 环境不变性及自动 reset 后首步。未 reset 环境的相等比较严格发生在 `_reset_idx` 前后，不含物理积分。

现有扫描检查支持 hit 有限性、地面归属及 reset 刷新，不逐子步测量传感器刷新时间戳；不能以本次 G2 复验宣称动态传感器时序或 G3 已通过。历史缺失足部视觉 USD 警告仍需如实记录。任何失败保留证据并结束阶段，不自动重试、延长预算、推进八环境、运行剩余策略用例或初始化 PPO。

## 5. 离线验证和可追溯性

最终回归 **589 passed、0 failed、0 skipped**，包括固定几何实例配置、单用例选择和 provenance、终止前采样、SafetyFilter、高度故障隔离、shared-bank 接触／reset 和真实生成循环的 CPU 测试。354 条警告来自已安装 tensor API 源码的 `invalid escape sequence`；不涉及本阶段实际物理运行。

首次合并执行为 588 passed / 1 failed：严格离线 mesh audit 拒绝了共用测试进程内残留的仿真命名空间。已将该集成用例放入全新 CPU 子进程，保留其不得导入仿真模块的断言；没有放宽生产几何或安全检查。初次失败日志／JUnit／命令以 `regression_initial*` 保留，最终证据为 [regression.log](regression.log)、[JUnit](regression.junit.xml) 和 [命令](regression_command.json)。

[post_review.json](post_review.json) 记录 225 项当前输入哈希验证、14 个候选单回合规格核对、15 列几何匹配、历史 Flat 归档哈希、漂移重算以及两种 G2 预算与 supervisor 实现一致。旧归档保持原哈希，当前准备单独冻结新输入。新报告的校验和见 [SHA256SUMS](SHA256SUMS)。

可重复运行以下离线审查，输出到终端，不发起任何 runtime 命令：

```bash
CUDA_VISIBLE_DEVICES='' /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  docs/lite3-source-followup-preparation-20261029/review.py
```

本阶段结论止于离线准备完成。其余 14 个实例的策略安全、当前版本 shared-bank G2、完整 G3/G4、120 环境以及 PPO 仍没有获得授权或资格。
