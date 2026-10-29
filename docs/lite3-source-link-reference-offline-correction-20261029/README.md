# Lite3 边界预测参考点离线修正（2026-10-29）

状态：`offline_correction_verified_runtime_not_run`。本阶段承接用户同意的[离线提案](../lite3-source-uphill-boundary-diagnostic-recheck-20261029/next_stage.json)，完成代码修正、诊断、CPU 验证和下阶段准备。没有启动仿真、重试上坡或初始化 PPO，没有更新策略、奖励、观测、安全阈值或历史失败结论。

边界 wrapper 已改为同时读取 `body_link_pos_w` 与 `body_link_lin_vel_w`。原先 `body_pos_w` 是 link 原点位置，`body_lin_vel_w` 却是 COM 速度。对旋转且 COM 偏移非零的刚体，应满足：

`v_link_w = v_com_w − omega_w × (R_link_w · r_com_link)`

实际安装的 IsaacLab 已提供这一换算；本次直接使用其明确接口，缺失接口立即报错，不回退到旧别名。`body_link_lin_vel_w` 的文字注释误写 COM，但实现确实返回经过换算的 `body_link_vel_w`。纯边界谓词文件及 0.27 m 半径、0.001 m 余量、0.02 s 外推、XY 非有限判定均保持原哈希。一个控制周期的线性外推仍不是连续时间安全证明。

新版 `tile_bounds.jsonl` 使用 v2 格式，在实际终止判定时保存全部 17 个刚体的 link 位置／速度、link 四元数、link 坐标系内 COM 偏移、世界 COM 位置／速度和世界角速度，并复制快照以避免 reset 覆盖。CPU reviewer 用独立的 float64 旋转矩阵和分量叉积核对 COM 位置与 link 速度，再按原 dtype 精确回放谓词。float32 换算容差为绝对／相对各 2e-6，四元数范数容差 1e-5；float64 均为 1e-10。这些是诊断核验容差，不改动边界阈值。

非有限参考数据会保留并标记不可核验，不算已验证换算，不能获得后续运行通过。历史 v1 日志仍支持原格式复核；未来 worker 和监督器都强制要求 v2。启动回读新增实际 COM 与 link/root 参考点一致性核对。每步 root 运动学另存 `root_reference_state`（单刚体维度，COM 偏移来自 root/body0），保留旧字段并明确其 COM 速度含义，为后续同参考点位移积分提供数据。

验证结果见 [post_review.json](post_review.json)、[测试日志](regression.log)与[独立复核脚本](review.py)：

- 针对性 CPU 测试 **131 项通过**；21 个相关测试文件的回归 **653 项通过**，无失败或跳过。354 条 warning 来自读取的上游 PhysX Python 源码中既有转义序列弃用提示。
- 在 CPU 上从已冻结的实际依赖提取速度换算和 quaternion 函数，仅执行算术函数体，没有导入 IsaacLab 或启动 PhysX。float32／float64 共 **8192** 组随机旋转／COM 状态通过独立几何核验，body 和 root 接口换算一致；float32 最大速度误差 3.564e-7 m/s，float64 为 8.882e-16 m/s。
- 合成平面运动例同时覆盖零偏移、纯平移、消除旧误触发、补上旧漏检及旋转后的 COM 偏移。下表使用 link Y=1.726 m、上界 1.729 m、外推 0.02 s，**不是旧上坡末步的重建结果**。

| 合成例 | COM Y 速度 | link Y 速度 | 旧预测 Y | 修正预测 Y | 边界标记变化 |
|---|---:|---:|---:|---:|---|
| 正向旋转、X 偏移 0.01 m | 0.3 m/s | 0.1 m/s | 1.732 m | 1.728 m | 触发 → 不触发 |
| 反向旋转、X 偏移 0.01 m | 0 m/s | 0.2 m/s | 1.726 m | 1.730 m | 不触发 → 触发 |

随机样例中 650 个旧触发消失、1378 个旧漏检出现；这些是人工近边界状态分布，不能当作实际策略的误报率或通过率。测试验证了缺失明确接口、损坏或过期参考字段、非单位四元数、非有限数据、自动 reset 前快照、多个环境和乱序 body ID 的行为。

新准备清单冻结 **237 项输入**。相对上一版增加参考点模块、测试 fixture、实际 ArticulationData、quaternion 数学模块及 editable 包解析文件，共 5 项；另更新 5 个已审查文件的哈希。[依赖审计](dependency_audit.json)和[输入变更](input_changes.json)保留完整细节。旧准备清单及 5 份历史归档的 **102 个文件**逐项哈希一致，旧上坡 **1059 次谓词**回放及原 review 输出完全一致。[基线进度](baseline_progress.json)逐字节复制最新旧账本：上坡两次同 seed 诊断失败，其余 12 个已准备条件未运行。

旧 FL_THIGH 末步没有完整四元数、角速度及实际 COM 回读，因此现在仍不能重建其修正后预测，也不能据此宣称原失败是误报或修正后一定通过。此前约 **0.350 mm** 的躯干位移积分残差是另一问题：50 Hz 速度梯形积分与 200 Hz 物理积分／求解器状态并不完全相同，具体分量尚未隔离；躯干静态 COM 为零，不能用大腿参考点错误解释它。详见[既有残差解释](../lite3-source-uphill-boundary-diagnostic-recheck-20261029/residual_explanation.json)。此前偶发的 descendant 清理测试失败仍未定位原因；本轮回归通过不表示该问题已修复。

[下一阶段具体规格](next_stage.json)已准备，**尚未获运行授权**：仅一次相同 5° 上坡 L1，geometry/state seeds 10001/20060，difficulty 0.3384848514104325，原 checkpoint，命令 `(0.3,0,0)`；GPU0 需重新检查空闲且可用显存不少于 12288 MiB，单环境最多 2000 控制步、120/240 s 时限，不自动重试。新目录 `/tmp/lite3_source_policy_uphill_link_reference_20261029` 在本次准备时不存在。条件通过仍要求前进 9 m 并稳定保持 50 步，安全条件与完整参考点证据分开报告。结束或失败即停止，不自动转其他地形、G3/G4、训练或 PPO。

可用以下命令仅复核本离线归档；不执行 next_stage 中的 GPU 命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python docs/lite3-source-link-reference-offline-correction-20261029/review.py
```

执行记录见 [execution_scope.json](execution_scope.json)。本记录仅支持离线实现与证据契约已验证；实际物理状态、日志开销、六类地形通过率、能否学到新步态、G3/G4 与训练准备情况都未获得新的运行结论。
