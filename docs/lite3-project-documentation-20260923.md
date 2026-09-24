# Lite3 在线地形适应项目说明

**更新时间：2026-09-23**
**项目目录：** `rl_training_main`
**适用对象：** DeepRobotics Lite3 仿真训练、在线适应算法开发和结果审计

## 1. 项目目标

本项目研究四足机器人从已掌握的平地行走技能迁移到未见崎岖地形的方法。系统以平地 PPO 策略作为稳定的基础控制器，只允许一个受约束的适应模块根据机器人本体观测进行更新，从而降低在线学习对原有步态的破坏。

当前项目的工程主线是：

1. 冻结并资格审查平地 PPO 基础策略；
2. 通过 LoRA-PPO 或 Residual SAC 学习有限的适应量；
3. 将策略请求动作送入安全投影器，得到实际执行动作；
4. 记录请求动作、执行动作、投影差异、终止状态和策略版本；
5. 对候选策略执行平地保持、目标地形冻结探针和安全审计；
6. 只有通过审查的候选才可发布为恢复检查点；
7. 封存测试使用独立入口，关闭更新并隔离测试数据。

项目目前是 **Isaac Lab 仿真研究代码**。现有结果不能直接证明实体机器人安全，也不能把单种子或开发集结果写成泛化结论。

## 2. 当前实现边界

| 模块 | 当前状态 | 说明 |
|---|---|---|
| 平地 PPO 基础策略 | 已有 | 作为冻结名义步态控制器和行为参考 |
| Direct LoRA-PPO | 已实现 | 只更新低秩适配参数和独立 Critic |
| Residual SAC | 已实现并持续修复 | 基础 PPO 冻结，Critic 学习请求残差，Actor 受新异状态门控 |
| 请求/执行动作分离 | 已实现 | PPO/SAC 的学习语义使用保护前请求动作，同时保存实际执行动作 |
| 终止与时间截断处理 | 已实现 | 真实终止不 bootstrap，时间截断使用末帧价值但切断优势递推 |
| 候选检查点、回滚和违规封存 | 已实现 | 训练期违规候选不能因后续探针通过而晋升 |
| 平地保持和目标地形冻结探针 | 已实现接口 | 用于候选验收；结果须按实验协议单独报告 |
| 本体积分上下文 | 已实现 | 适配 Actor 可接收本体历史积分特征 |
| 高度扫描教师接口 | 已实现接口 | 高度扫描教师训练和特征摘要可运行，但不代表完整 RGLS 已验证 |
| 潜变量动作反演、低秩教师解码器 | 已实现工具 | 支持离线初始化和监督接口 |
| 风险门控潜变量残差端到端方案 | 部分实现 | 风险缩放和潜变量组件存在，统一训练、对照和多种子效果仍待验证 |
| FlowDAgger | 未直接复现 | 项目只借鉴潜空间动作反演思想，当前 PPO/SAC 代码不是 Flow policy |

## 3. 系统结构

```text
本体观测/历史 ──> 冻结 PPO ──> 基础动作 a_base
      │                              │
      ├──> LoRA-PPO 适配器 ───────────┤
      └──> Residual SAC/潜变量模块 ──> 请求动作 a_req
                                         │
                           动作幅值/变化率/关节范围/
                              接触与姿态安全投影器
                                         │
                                         v
                                  执行动作 a_exec
                                         │
                          Lite3 仿真环境产生转移和奖励
                                         │
                      记录 (a_req, a_exec, 差值, 安全原因)
                                         │
                      PPO rollout / SAC replay / 候选审查
```

基础策略输出、适应模块输出和安全投影后的动作必须保持语义区分：

- `a_req`：策略提出、用于策略概率和 Critic 训练定义的请求动作；
- `a_exec`：安全投影后真正发送给执行器的动作；
- `a_exec-a_req`：动作被保护机制修改的差异，用于诊断、惩罚和候选拒绝。

把 `a_exec` 误当成 PPO 采样动作，或把保护后的残差直接当作 SAC 的 Critic 输入，会破坏更新语义。当前代码已对此建立测试和检查点契约。

## 4. LoRA-PPO 原理与实现

对基础 Actor 的指定线性层使用：

```text
W = W0 + (alpha / r) B A
```

其中 `W0` 冻结，`A` 和 `B` 是低秩可训练矩阵，`r` 为秩。适配器初始化时 `B=0`，因此确定性动作初始时与基础策略一致。当前实现支持：

- 注入、合并和解除合并 LoRA；
- 原始线性层检查点严格加载；
- 仅标记 LoRA 参数和独立 Critic 为可训练；
- 连续 rollout、GAE 和 clipped PPO；
- 源平地行为锚定、梯度裁剪、有限值检查和近似 KL 拒绝；
- 候选拒绝时同步恢复 Actor、Critic 及两个优化器。

LoRA 是当前已验证的软件适配路径。它不等于 FlowDAgger，也不自动包含高度扫描或潜变量编码器。

## 5. Residual SAC 与潜变量扩展

Residual SAC 保留冻结 PPO，学习受限动作残差。残差经过幅值、变化率、关节目标和接触保护后才执行。系统同时维护全局回放和新异状态回放；Critic 可以先学习价值，Actor 只有在新异门控允许时更新。

潜变量扩展提供以下离线工具：

1. 从安全教师残差样本按动作限幅归一化；
2. 使用截断 SVD 拟合固定低秩解码矩阵；
3. 对教师动作执行 `u* = argmin_u ||a(u)-a_teacher||² + beta||u||²` 动作反演；
4. 监督训练历史/高度扫描编码器预测 `u*`；
5. 再将低维潜变量送入受限残差路径。

这些工具支持后续的 RGLS 类实验，但项目当前不能把“已实现工具链”表述为“已经完成端到端潜变量适应验证”。

## 6. 安全、回滚与数据隔离

候选策略必须经过有限值和参数范围检查、策略漂移或近似 KL 检查、源平地行为锚定、目标地形冻结探针和安全违规审计。训练期发生安全违规时，候选被永久标记为不可晋升；系统恢复最近一个无违规且冻结验证通过的检查点，并同步恢复网络与优化器状态。

安全机制包括：

- 请求动作幅值和相邻周期变化率限制；
- 关节目标范围裁剪；
- 姿态、机身高度、关节速度和非足接触检查；
- 高度扫描无效时将适应特征置零并暂停残差；
- 严重安全事件时残差置零、暂停更新或回滚；
- 记录策略版本、检查点哈希、协议哈希、投影原因和回滚事件。

训练、开发验证和封存测试使用不相交的数据划分。封存测试入口为：

```bash
python scripts/reinforcement_learning/online_adaptation/run_lora_ppo_sealed_test.py \
  --base-checkpoint /path/to/model.pt \
  --checkpoint /path/to/selected/latest.pt \
  --output-dir artifacts/final_test/run
```

该入口只读取选定检查点，关闭梯度和参数写入，输出独立的 manifest/report，不参与候选选择。

## 7. 当前实验结论

现有证据支持以下工程结论：

- LoRA-PPO、Residual SAC、请求/执行动作分离、终止边界处理、候选回滚和封存测试隔离已经具备可测试实现；
- 单种子配对实验中，LoRA 和 SAC 曾比冻结 PPO 获得更大前向位移，但均出现真实非法接触，不能据此宣布已经学会陌生地形；
- 近期小规模 PPO/SAC 试验尚未形成稳定的多种子成功率提升；
- 较高 Actor 更新预算曾导致平地保持和开发地形表现下降，因此不作为当前默认方案；
- 历史实验若使用旧奖励、旧终止协议、旧动作语义或旧 replay schema，不得与新协议结果直接合并；
- 当前最可靠的下一步是按固定训练/开发/封存划分完成多种子对照，而不是继续扩大单次运行长度。

## 8. 推荐运行顺序

### 8.1 安装与组件测试

```bash
python -m pip install -e source/rl_training
pytest -q tests/online_adaptation
```

仿真训练还需要已安装 Isaac Lab/Isaac Sim，并使用与项目匹配的 Python 环境。

### 8.2 基础策略资格审查

```bash
python scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py \
  --base-checkpoint /path/to/flat/model.pt \
  --task Flat-Deeprobotics-Lite3-v0 \
  --headless \
  --output logs/online_adaptation/base_qualification.json
```

资格审查不通过时，应先修复基础检查点、环境或协议，不应依靠适应模块掩盖问题。

### 8.3 LoRA-PPO

```bash
python scripts/reinforcement_learning/online_adaptation/train_lora_ppo.py \
  --base-checkpoint /path/to/flat/model.pt \
  --task FlatOnline-Deeprobotics-Lite3-v0 \
  --headless \
  --run-dir logs/online_adaptation/lora
```

多回合候选监督和续训使用 `run_lora_ppo_multiepisode.py`；参数应由 manifest 固定，不能在续训时静默改变。

### 8.4 Residual SAC

```bash
python scripts/reinforcement_learning/online_adaptation/train_residual_sac.py \
  --base-checkpoint /path/to/flat/model.pt \
  --task OnlineAdapt-Deeprobotics-Lite3-v0 \
  --update-schedule control-step \
  --headless \
  --log-dir logs/online_adaptation/sac
```

算法配对应优先使用 `control-step` 调度；`wall-clock` 仅用于实时并发和 deadline 检查，两者结果分开解释。

## 9. 结果记录要求

每个实验目录至少保存：

- 运行配置、环境和算法协议；
- 基础检查点、适配器/残差检查点及 SHA-256；
- 训练、开发和封存数据划分；
- 每步或每回合 summary；
- 请求/执行动作统计与安全原因；
- 更新、拒绝、回滚和恢复事件；
- 终止与时间截断标记；
- 评测参数、种子、地形几何和协议哈希。

报告结果时同时给出成功判据、失败判据、前向位移、速度误差、姿态/接触违规、动作投影差异、平地保持和回滚次数。没有统一协议时，只报告为工程观察，不报告为方法优越性。

## 10. 主要代码入口

| 用途 | 路径 |
|---|---|
| 在线适应公共组件 | `source/rl_training/rl_training/online_adaptation/` |
| LoRA-PPO 训练 | `scripts/reinforcement_learning/online_adaptation/train_lora_ppo.py` |
| 多回合 LoRA 监督 | `scripts/reinforcement_learning/online_adaptation/run_lora_ppo_multiepisode.py` |
| Sealed 测试 | `scripts/reinforcement_learning/online_adaptation/run_lora_ppo_sealed_test.py` |
| 高度扫描教师 | `scripts/reinforcement_learning/online_adaptation/train_height_scan_teacher.py` |
| 教师解码器拟合 | `scripts/reinforcement_learning/online_adaptation/fit_teacher_decoder.py` |
| 单元与契约测试 | `tests/online_adaptation/` |
| 研究进展报告 | `deliverables/lite3_research_progress_20260922/` |
| 专利交底书 | `deliverables/lite3_online_adaptation_patent_20260923/` |

## 11. 已知限制

当前没有完成实体机器人闭环验证；没有完成特权教师轨迹的大规模采集、完整动作反演数据集、端到端高度扫描编码器训练和潜变量残差多种子效果验证。项目文档、专利交底书和研究报告均应保留这些边界。

