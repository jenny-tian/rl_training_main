# G1.2 SAC控制时延与冻结候选执行修复

本轮修复了两个实际问题：小张量GPU同步/传输造成的控制耗时尖峰，以及managed冻结评估把学习残差乘以零的配置错误。原20 ms控制计算阈值、动作边界及安全条件保持不变。**修复后的9696个控制步无超时；这不等于G1学习收益或路线晋级。**

## 定位证据

两个预定冻结base诊断分别在第1、705步超时。回退前控制计算24.14/26.61 ms，当前线程CPU时间24.14/26.60 ms，GC重叠均为0。旧`control_compute_ms`还包含超时后的安全回退，变成39.09/41.28 ms；新增诊断明确区分回退前耗时、线程CPU时间、GC与宿主子阶段。

0物理步的保存观测重放复现了慢调用。cProfile显示：8次`Tensor.item`读取累计约16 ms，两次每步新建固定动作边界`torch.tensor`约4 ms。开销主要落在CUDA标量同步/小额传输，不支持把这些已观测尖峰归因于GC。共享GPU的具体调度原因尚未单独证明，没有停止其他项目或调整其资源。

同样的1000次独立GPU重放，旧实现190次超过20 ms，优化后1次；唯一优化后超时是未预热的首次冷调用，实际worker仍执行原有启动预热。由于系统负载会变化，这不是超时概率的统计估计。另做同进程交错512步：两个实现当时均无超时，优化后CUDA中位数约2.965 ms，旧实现约3.505 ms。CPU/CUDA各512步的动作、残差、安全原因、历史及待写replay字段均逐位相同，包括无效地形与reset边界。

## 修复内容

- 固定动作边界按device/dtype缓存，避免每个动作调用都重新创建并传输两个常量tensor。缓存有容量上限，并在普通tensor模式下创建，避免第一次由inference模式填充后破坏LoRA反向传播。
- 同一步的terrain-valid只读取一次；五个安全原因合并为一次回传，保留顺序、跨环境并集及所有安全条件。
- worker manifest增加共享动作/稳定性源码的哈希；异常路径保存checkpoint时也保留target_warmup进度。新旧source-bound session不静默混用。
- 增加控制时延诊断并在退出时移除GC回调，不禁用GC、不移动或提高20 ms阈值。该阈值衡量控制计算，不包含整个仿真、日志和持久化循环，不能据此宣称端到端硬实时。

学习计时回归又发现：适应有3次actor更新，但冻结地形和平地的`action_residual_gain`恒为0，执行动作等于base。现managed learned frozen候选使用已注册的`adapting_residual_gain`（当前0.5）；online等待态仍为原gain=0，base模式为0，bare worker保留显式gain语义。运行配置和summary记录解析后的等待/部署增益。

## 实际运行与预算

所有运行先登记于[实验登记表](lite3_experiment_registry.md)，同一base、development rough_4cm seed101、learner7、perturbed-v1、速度0.3，最多一张GPU运行本组仿真。不按结果更换起点；保留两个baseline失败。总预登记上限14280步，实际**10402步、uncertain=0**。

| 组 | 控制步 | 最大控制耗时 | 超时数 | 结论 |
|---|---:|---:|---:|---|
| baseline诊断 |706|41.283 ms（含回退）|2|保留原失败|
| 优化后冻结base |5416|10.280 ms|0|两个地形、两个1000步平地窗口完成|
| 学习/恢复计时回归 |2640|9.081 ms|0|预热1000次不重复；3次actor更新；暴露冻结gain=0|
| 冻结增益修复回归 |1640|6.974 ms|0|2次actor更新后，真正执行学习残差并冻结|

优化后的两个地形起点分别在1416步完成、2000步到任务deadline；两个平地起点各1000步，均有完整稳态数据。它们是冻结base结果，不是适应成功率。

学习/恢复回归使用256步短warmup和1000次critic-only更新，仅作工程检查；两个1000步物理trial分别有2/1次actor更新，critic更新1372/500次，第二trial不重复预热。该组后续两个320步零gain冻结窗口不作学习后策略评估证据。

冻结增益修复另开独立session：1000步适应后有2次actor更新，随后地形/平地各320步，gain每步均为0.5，最大执行残差分别约0.001699/0.001847（归一化动作单位）。两段均updates=critic_updates=replay_writes=0，冻结前后actor张量相同，平地数据满足comparison_eligible。重复evaluation/retention命令均未新增trial或控制步。

历史上使用默认managed gain=0的learned-frozen记录只视为base执行证据。保留原始记录，不重新命名为有效学习后表现，也不据此比较算法优劣。

## 验证与复核

- 完整测试：**1602 passed, 7 skipped**。新增覆盖缓存后梯度有效性、多dtype/非连续输入、多个安全原因的并集和顺序、计时回调生命周期，以及冻结增益和非零发布均值执行。
- [验证JSON](lite3_g1_latency_validation_20260929.json)：控制时延分位数、逐trial成本、源码SHA、预热/actor/critic计数及冻结残差。
- 原始数据与验证脚本：`artifacts/g1_control_latency_20260929/`，包括`baseline/`、`optimized/`、`learning/`、`frozen_gain/`、离线profiling、交错数值对照和`validate.py`。
- [baseline到最终源码补丁](lite3_g1_latency_optimization_20260929.patch)与[冻结增益修复补丁](lite3_g1_frozen_gain_fix_20260929.patch)：分别从最终源码用`git apply --unidiff-zero --reverse`重建baseline和中间计时修复源码。两版本均已重算并匹配各自manifest SHA；两个补丁分别使用，不连续反向应用。
- base权重文件SHA始终为`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`；奖励分项与终止bootstrap重算通过。

## 下一步

优先执行G1的完整回合留出价值诊断：两个调试地形各6个完整冻结策略回合，固定4/2训练验证划分，比较常数/时间基线与学习critic；SAC另做固定策略和alpha的soft-target尺度诊断。短warmup与训练内TD误差不替代该验收。随后才按原计划比较UTD、Q主干LN等候选，候选晋升和最终封存规则仍需锁定。
