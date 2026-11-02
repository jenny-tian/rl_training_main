# Source PPO 入口准备与 CPU/GPU 合成更新检查

状态：`entrypoint_prepared_synthetic_cpu_cuda_passed`，2026-11-02。

用户在询问 GPU 与零更新的原因后，同意继续按调整后的范围推进：入口实现、CPU
边界测试、小规模 GPU 合成 PPO 更新检查。本阶段没有启动 Isaac Sim，没有真实
地形交互或正式 Source 试训。当前阶段结束后暂停，下一阶段另行批准。

## 已完成的实现

- 审阅并完成原有三个未提交草稿：`training_ppo.py`、`training_runner.py` 和
  `source_training_entrypoint.py`；加入合成检查入口和 23 项针对性测试。
- actor-only 初始化保持已有均值权重，std 重置为 1.0；critic、optimizer 和迭代
  计数全新创建。actor/critic 输入宽度分别为45/393，六类/15个L1 cell保持。
- rollout 保存高斯采样动作及其原始 log probability；执行端仍使用生产安全桥，
  超时使用真实 pre-reset terminal observation bootstrap。普通失败和复合
  failure+timeout不会错误获取timeout bootstrap。既有回归覆盖这些数值区别。
- 修复草稿中 `self.run` 数值覆盖 `run()` 方法，以及数值阶段默认checkpoint
  节点超预算的问题。输出目录禁止覆盖；中断不自动重试或恢复。
- 在已安装的 RSL-RL 更新内部，使用局部 TorchFunctionMode 检查 backward 前的
  总loss；optimizer pre/post hooks检查梯度、参数及优化器状态。正常更新和原版
  PPO精确一致。失败时同时报告已尝试交互、返回交互、已存储交互、完整PPO更新和
  部分Adam minibatch步，避免把部分权重变化写成“没有更新”。
- 训练入口在启动Isaac前校验批准清单、远端实时SHA、工作区、模型和生产输入哈希、
  所列运行依赖，以及哈希绑定的选定规模运行证据和Source-only选模协议。提供
  `--validate-only`，拒绝路径不启动Isaac或CUDA。Isaac相关任务在应用启动后导入。
- 启动/训练/清理有独立持久记录；应用关闭前先记录`closing`，不把发出关闭请求
  写成Python层关闭已返回。

入口包含真实数值阶段的收集/更新代码，但**尚无真实仿真端到端运行证据**，也没有
生成可执行的正式训练lock。运行依赖清单的完整构成、运行监督及后续选模协议仍须
在真实运行方案中固定，结构和哈希检查不等于外部证据已通过。

## 本阶段实际验证

| 检查 | 结果 |
|---|---|
| 已登记回归 | 433 passed，0 failed，0 skipped；原410项加新增23项 |
| CPU合成更新 | 三seed各2次PPO更新，共6次，1,152 synthetic transitions |
| GPU0合成更新 | 三seed各2次PPO更新，共6次，1,152 synthetic transitions |
| 每个seed、每种设备 | 8个合成环境 ×24步 ×2次更新，384 transitions |
| 优化器 | 每个seed 40次Adam step；每种设备120次 |
| 观测与网络 | policy45/critic393；512/256/128 ELU；真实登记actor初始化 |
| 超参数 | 5 epochs、4 minibatches，其余使用登记候选runner_config |
| 设备 | CPU；GPU0 NVIDIA RTX3090，PyTorch2.7.0+cu128，RSL-RL5.0.1 |
| GPU进程 | 退出0，未超时，进程组2306994为空 |
| GPU时长 | 子程序约14.40秒，外部计时约18.14秒；120秒上限 |
| GPU张量内存 | peak allocated 45.0542MiB，peak reserved 50MiB |
| 原始checkpoint | SHA固定；源文件未更新；合成输出未选为训练候选 |
| 真实Source训练 | transitions=0，PPO更新=0，Isaac启动=0 |

每种设备的三次运行初始actor哈希相同，初始critic哈希互不相同；每次结束actor和
critic均改变，optimizer从空状态变成17个参数状态项，loss均有限。每个seed均经过
一个纯timeout和一个failure+timeout。GPU使用生产SourceTrainingVecEnv、安全桥、
lifecycle和SourcePPO，但底层状态/奖励是合成fixture，**没有物理引擎**。

CPU和CUDA RNG/浮点内核不同，不要求训练后的参数逐位相同。合成奖励的改善不能说明
地形技能改善；合成耗时不能用作真实训练吞吐。GPU卡级显存启动/退出均为18MiB；
张量allocator峰值和卡级显存是不同口径，详见原始CSV与采样。

CPU故障测试覆盖loss溢出、NaN/Inf梯度、损坏参数/optimizer、初始观测宽度错误、
非有限观测/奖励、采集KeyboardInterrupt、Adam已经执行但完整PPO尚未完成、预算
耗尽、已有目录、未批准/缺失/变更lock与依赖等。正常更新与安装版PPO的loss及参数
哈希精确相同，证明新增检查没有改变通过路径的更新公式。

## 测试范围限制

一次`tests/source_scan`整目录收集失败：已有`source_family_runtime_smoke.py`导入
仓库缺失的`scripts.tools.runtime_progress`。该历史模块未在本阶段修改。完整失败
日志、JUnit和命令与最终433项已登记回归结果一并保留；不能写成整目录全绿。

pytest使用`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避开现有typeguard与typing_extensions
的自动插件兼容问题，未更改安装依赖。433项运行中的359条warning来自原有PhysX
源文件文档字符串转义提示。最终回归命令保持上一阶段14个测试文件并增加入口测试。

## 预算与尚未完成的真实运行工作

120环境×24步候选不变，262144/1048576/2097152是transition硬上限；完整rollout
实际对应91/364/728次更新和262080/1048320/2096640次交互，不向上补足。

当前真实入口仅支持从全新状态开始的numerics节点。midpoint/pilot明确拒绝：跨阶段
续训和累计预算台账尚未实现，不能简单重新初始化并再次运行364或728次而重复计入
同一seed预算。合成检查使用独立的新输出目录，未消耗正式Source训练预算。

选定规模G3/G4、真实环境live critic参与更新、实际吞吐/显存和Source-only六类
质量/选模协议仍待完成。本阶段没有读取Development/Sealed反馈，没有选模或发布。
旧平地策略在六类上的失败仍是学习目标，不新增“旧策略必须全过”的训练门。

## 归档说明

`raw/`的27个JSON/XML/CSV/日志文件与临时原件逐字节一致；`provenance.json`记录哈希。
12个合成actor输出仅在归档中登记哈希和字节数，原件仍位于其中记录的临时目录；
它们不是正式训练checkpoint，未在仓库发布。每次更新的参数哈希、loss、计数和
保存校验保留在run_report及receipts。`review.py`只做CPU上的归档与计数核对，
不是再次执行或独立重放GPU更新。220项生产/资产输入在CPU与GPU检查开始/结束一致。
原始日志保留RSL-RL输出的尾空格；代码/文档的Git空白检查仅排除`raw/`，不改写原始
证据。8项已安装框架源码的哈希在检查结束后另行登记，不声称是运行前的依赖快照。

## 下一阶段提案，待用户同意

准备小规模真实Source PPO端到端联调：候选8环境×24步×2次更新，共384条真实交互。
先实现并审阅独立诊断模式、运行监督器、时限/进程组清理、GPU资源检查、原始采集与
验收，以及这384条交互的预算归属和seed。它们尚未授权，也不得悄悄排除在未来累计
训练预算之外；不把合成模型当起点、不覆盖原始actor、不自动扩到120环境。

该下一阶段先交付可审阅清单和命令，再单独请求真实仿真启动批准。小规模联调也不
替代120规模G3/G4与正式Source-only选模锁定。见`next_stage.json`。
