# Lite3 固定平地参考集

连续训练 v8 的第 8、9、10 回合分别有 5、5、6 次更新因 reference anchor 超限被拒绝。原 worker 把当前复杂地形 rollout 同时作为基础策略保持参考，实际要求新地形上的均值也接近平地 PPO。这会限制新能力形成；它也没有直接测量平地分布上的保持情况。

新增可选 `--flat-anchor-observations`，把同一 `1e-4` MSE 上限施加到固定、独立的源平地观测上。每个 worker 在相同输入上重新计算原冻结 PPO 的教师均值；当前复杂地形数据仍只用于正常 PPO 与 Critic 更新，并保留原在线 KL 限制、SafetyFilter 和严格完成资格。默认没有提供资产时仍使用原有参考方式。

## 参考集来源与限制

资产位于 `artifacts/semester_2026/09/lora_ppo_multiepisode/source_flat_anchor_v1/`。从已登记的 100000 条 frozen、安全、plane-only replay 中，以固定均匀索引选择 4096 帧，将原始 49 维本体帧重排、缩放成 45 维 Actor 输入。模型、replay、baseline manifest、配置文件及 tensor 哈希均核验，详细映射和索引写入 manifest。

- `source_anchor.pt` SHA-256：`539ee7eba9301a2f0cbff6aaf109f72b37a7b6e298eefe04e299cdfc6f1de2cb`
- 原 PPO SHA-256：`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`
- schema：`reconstructed_clean_flat_policy45_v1`

原 replay 保存 float16 原始帧，没有保存当时 Actor 的观测噪声。因此这是重构后的干净平地参考集，不是原始带噪 45 维观测的精确重放；没有把旧 base_action 与重构输入配对。教师均值必须重新计算。旧奖励和转移不会作为 PPO 样本混入本轮。固定观测上的保持也不能替代实际平地行走复测。

## 起点核验

同一固定参考集、同一教师、同一阈值下：v8 E10 的 MSE 为 1.04702e-4，E18 为 1.15301e-4，均不合格。没有修改样本或提高门槛。在已审查的 E1–18 中，最近合格的是 E15，MSE 为 9.32535e-5，checkpoint SHA-256 为 `028793db43e8d13cfa62c20dfe73b50709640c4ae282a5069b36833fe55ead0c`。初始 v7 E5 与验证备份 v6 E15 分别为 9.67820e-6、4.18550e-6。

完整结果保存在 `candidate_teacher_mse.json`。这里的合格只表示源参考输出约束，未将它标记为新的地形验证。后续仍分离未验证训练候选和已验证恢复模型，保持严格冻结探针及稳定窗口标准。

## 诊断与运行

更新日志新增 `candidate_anchor_mse`，包括被拒绝候选的实际 MSE；拒绝后参数仍按原规则恢复。完整回归 1312 项通过、7 项跳过；新增候选 MSE 诊断后对应 34 项定向测试通过。

另增加 `--single-gpu-renderer`：保持指定 CUDA 物理设备，在 Kit 中关闭自动多 GPU 渲染。此前日志显示 GPU1/3 的 PhysX 已正确分配，但 Vulkan 渲染仍落在物理 GPU0。该选项传递标准 `--kit_args`，是否实际改善分配需用下一次运行的 Vulkan Active 行、UUID 和进程设备验证，不能只看设备组内的局部编号。

source-flat 实验清单使用 supervisor v9，保留 worker 的现有奖励、观测和采样合同；资产路径、哈希和渲染设置另行登记。
