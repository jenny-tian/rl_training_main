# G1.7 SAC 成功奖励守卫与 Huber 候选（2026-09-29）

**本轮完成了非零残差采样和冻结执行核验，但没有确认适应收益，并发现、修复了预热目标增益依赖瞬时控制状态的问题。**物理实验共19129步，属于增益修复前数据；修复通过真实agent集成/单元回归，尚未追加修复后的物理实验。

## 发现的问题与单一改动

有限任务的合法成功终止奖励为305、bootstrap=0；原SAC双Q采用MSE，critic loss守卫为800。当128行minibatch只有一条成功样本、预测接近0时，该样本对双Q loss的贡献已约为`2*305²/128=1453.52`，超过守卫。该冲突不意味着该回报本身非法。

先登记零物理步合成检查：真实49维frame/25历史/12动作网络，初始化seed7，128行中127行reward1.5及1行success305，固定seed23随机动作、alpha=.01、gain=.5。原MSE loss1458.2755、mean TD3.9484，因critic_loss_limit拒绝；相同目标、初始化的既有Huber(delta=1) loss6.8721通过，grad norm2.0445，actor/alpha不变。两次合成attempt、一次接受，模型均丢弃。首次脚本因无用helper导入失败，在任何更新前退出，成本0步/0更新。

Huber改变了大TD误差的梯度权重，因此两个loss数值不能作为价值质量的直接比较。它只解决本例因单个合法成功奖励触发MSE守卫的问题，不保证成功状态拟合充分，也不保证任务表现改善。TD上限10、梯度上限3000、grad clip1、奖励、终止语义和安全规则均保持不变；测试还验证大范围TD异常仍拒绝并回滚。

managed supervisor现允许已有`critic-huber-delta`参数并绑定到manifest；未设置该参数的原默认MSE行为不变。预热逐更新日志新增`residual_gain`，避免用检查点末尾等待态增益推断先前更新的目标策略。先完成数值契约核查，再决定是否开展原计划中的LN/UTD实验，未把这些因素一起改动。

## 注册的真实验证

[协议](lite3_g1_sac_huber_protocol_20260929.json)固定development rough_4cm101/107、learner43、perturbed-v1。每地形从同一base独立开始，B5000：前4000步冻结确定性base采样，恰好1000次critic-only预热，然后最多1000步后续学习。保持n=1、原网络、alpha=.01、critic/actor更新额度.5/.005；唯一候选损失为Huber1。

每地形候选只取最后提交的适应checkpoint，固定一次完整任务评估（≤2000步）及一次真实平地保持（≤1000步）；另起fresh base执行相同phase-local reset seeds的任务/平地对照。总上限22000控制步，不用以前诊断轨迹作免费训练，不按结果补抽、不选最佳checkpoint。预热失败不晋升aborted checkpoint，本组不追加救援候选。

代码、测试与协议在采集前提交推送`d43af03`。最多GPU0/1各一个worker，依次适应→冻结任务→平地→base任务→base平地；各阶段返回码与命令写入`execution-{terrain}.json`。旧source-bound session不续接。

## 验证边界

实际 **19129 / 22000 控制步**，四session均完成、uncertain=0，无替换起点。两地形各5000适应步，均完成4000步冻结base采样及1000次critic-only预热；随后各500次常规critic更新，合计3000次真实critic更新（含2000次预热），actor更新合计2次。两地形的预热replay均包含成功终止样本，预热未因loss守卫退出。

| 地形 | 预热后控制步 | 非零随机请求 / 进入critic replay | 预热目标gain（修复前） | actor更新 |
|---|---:|---:|---:|---:|
| 101 | 1000 | 361 / 361 | 0.5（1000次） | 0 |
| 107 | 1000 | 578 / 578 | 0（1000次） | 2 |

101预热后的有效actor replay样本分为两回合的65/140行，未触发当前每200条有效actor样本一次更新的额度；回合结束会清除该回合剩余额度。107共有579条有效actor replay样本，完成2次actor更新。没有更新拒绝，但名义1000个后续控制步远不等于1000次学习或足够多的actor更新。

| 地形 / 策略 | 冻结任务结果 | 前进距离(m) | 冻结请求RMS | 平地稳态XY MAE(m/s) |
|---|---|---:|---:|---:|
| 101 / 候选 | completion，1483步 | 8.01275 | 0 | 0.02263693 |
| 101 / base | completion，1483步 | 8.01275 | 0 | 0.02263693 |
| 107 / 候选 | physical_failure，1106步 | 5.08205 | 0.00057017 | 0.02259661 |
| 107 / base | physical_failure，1057步 | 5.12575 | 0 | 0.02263693 |

各候选/base对的reset seed和初态匹配。101没有actor更新，冻结执行等于base，不能把其完成任务归为学习收益。107确实执行了非零已学均值残差，但仍物理失败，前进距离略小于base；多存活49步不等于完成率或前进距离提升。

四个平地窗口均完成1000步，排除前5秒后各有750个稳态样本，均无跌倒/安全违规，满足窗口可比较条件。107微小MAE差异只是单起点观察，不构成保持能力改善的统计证据。

冻结候选执行gain均为.5、base均为0；actor输入检查点与冻结before/after张量逐位一致，actor/critic更新计数和replay写入数为0，replay大小恒定。冻结worker按设计不输出完整learner checkpoint，因此没有声称取得冻结后的完整critic/replay逐位快照。原始输入checkpoint哈希保持不变。8次冻结phase重复调用新增0步/0试验。

## 预热增益修复与剩余限制

新增日志揭示：预热函数原来调用`agent.update()`时读取的是当时控制gain。101第4000步处于适应状态，目标gain=.5；107当时处于等待状态，目标gain=0。同一个注册适应gain并未保证同一个预热目标策略。

已修复为`pretrain_target_critic`必须显式接收注册适应gain，并通过仅允许critic-only使用的覆盖参数传入更新器；worker和预热日志均使用`args.adapting_residual_gain`。控制gain不被临时改写，常规更新保持原行为。managed source/config哈希阻止旧session跨源码续接。

集成测试在控制gain为0/.25/.5时，均核对三次预热更新实际传入目标gain=.5、target critic收到非零请求、控制gain及actor/alpha保持不变；验证非法覆盖参数及用于actor更新的覆盖请求会被拒绝。**本轮物理结果没有重写成修复后证据，也没有重跑候选。**下一步必须登记新的source-bound小预算复核，再审查有效actor更新调度，不能据当前单次结果增加长训练掩盖低更新覆盖。

本轮还保留了1次控制超时：101适应trial00001第951步，回退前wall34.319 ms、线程CPU5.025 ms、GC0，adapter/safety阶段wall28.654 ms；回退后日志总值44.072 ms不能用作原始触发耗时。20 ms阈值未改变。这个事件提示等待/调度或设备同步开销，单次记录不能确定具体原因，不宣称硬实时问题已消失。

## 证据与范围

本组只有一个独立适应seed和每地形一个冻结起点，用于学习/动作覆盖和冻结执行检查，不能满足G1要求的两个独立session、每session五个保留起点。平地窗口必须完成且有稳态数据、无跌倒/安全违规才可比较，不能拿提前结束的低误差宣称保持更好。

最终回归测试 **1636 passed, 7 skipped**，覆盖成功批次守卫、TD守卫回滚、supervisor真实参数解析、预热gain日志和显式目标gain集成。原始结果目录为`artifacts/g1_sac_huber_coverage_20260929/`，合成检查位于`artifacts/g1_sac_success_guard_20260929/`；[精简验证JSON](lite3_g1_sac_huber_validation_20260929.json)保留完整报告哈希、逐回合结果、计费和限制。

对账脚本初版错误地要求冻结阶段存在`session.pt`，在读取冻结证据时失败。已按既有冻结产物契约改为检查actor before/after、输入checkpoint哈希、零更新/写入计数及replay大小；首次失败保留在`audit/failure.json`，完整通过结果位于`audit_validated/report.json`。该修正新增物理步0，没有修改实验数据。
