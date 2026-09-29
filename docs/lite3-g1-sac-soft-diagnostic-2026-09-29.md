# G1.5 SAC 固定策略 soft-Q 诊断（2026-09-29）

**旧预热 critic 已较好拟合慢更新目标，但当前证据不足以认定 Q 收敛或 G1-B 合格。**预热后的 target-critic Bellman RMSE 为0.120；改用冻结当前critic构造相同策略的下一状态目标，RMSE仍为1.515。共同replay全部是零残差请求，且没有终止/截断行；不能据此判断随机残差动作价值或失败边界。

## 固定数据与计算契约

[首轮协议](lite3_g1_sac_soft_protocol_20260929.json)锁定G1.1旧SAC session的trial00009和trial00010检查点、manifest及SHA256。before已采3881物理步、0 critic更新；after完成4000步预热采样和1000次critic-only更新，随后又有43次常规critic更新。因此after是**1043次更新后的状态**，不是恰好预热结束的状态。trial00010因旧日志序列化问题仍记aborted，未恢复、重标或晋升。

使用before replay全部3871有效行；after保存4076行，工具要求其前3871行逐字段完全相同。两个快照actor/alpha逐位一致，actor/alpha优化器为空、actor版本0。网络和数据从checkpoint只读加载，没有optimizer调用或物理采样。replay半精度观测按实际`ReplayBuffer.sample`规则转float32送入网络。

策略固定为原direct residual actor，温度约0.01，统一使用注册适应gain=.5；after保存的瞬时等待gain为0，只作记录，不能将其混入前后策略比较。每行16次随机动作，batch128、CPU单线程、seed23；两快照和后续算子检查使用相同随机数。随机数状态在诊断函数退出后恢复。

soft目标为`r + mask*discount*(min(Qnext)-alpha*log_prob)`，n=1、gamma=.99。熵密度属于12维normalized tanh action坐标，实际请求由action_limits和gain变换，再送入包含安全过滤动力学语义的Q。当前target_entropy=-6是原配置；不把它改解释为6维动作，也不凭空加入缩放Jacobian。这里不使用纯reward-only MC作为SAC soft-Q真值。

首轮结果只衡量对target critic的拟合，因此在不改写原结果的前提下，另行登记[当前Q算子协议](lite3_g1_sac_soft_self_protocol_20260929.json)：仅将bootstrap网络换为冻结的当前critic，其他数据、策略、温度和抽样都不变。它衡量这些replay点上的固定点一致性，仍不是实际环境多步soft回报校准。

两轮各2快照×3871行×16动作，共247744个随机动作样本用于前向诊断；新增控制步0、更新0、模型候选0。原始物理成本仍计在G1.1，不重复计费。各快照前向耗时见[验证JSON](lite3_g1_sac_soft_validation_20260929.json)。

## 结果

表中残差为当前`min(Q1,Q2)`减去16次采样平均soft目标。

| 快照 / bootstrap网络 | 当前minQ均值 | soft目标均值 | Bellman RMSE | 偏置Q−target | 双Q平均绝对分歧 |
|---|---:|---:|---:|---:|---:|
| before / target critic | −0.005857 | 1.578051 | 1.603419 | −1.583908 | 0.126645 |
| after / target critic | 8.597847 | 8.606985 | 0.120417 | −0.009137 | 0.013634 |
| before / 当前critic | −0.005857 | 1.578051 | 1.603419 | −1.583908 | 0.126645 |
| after / 当前critic | 8.597847 | 10.094397 | 1.514500 | −1.496550 | 0.013634 |

before的当前/目标网络相同，因此两算子结果一致。after中，bootstrap网络选择使平均目标差约1.487，远大于单状态目标采样标准差约0.013。低target-critic loss主要反映拟合滞后目标，不能证明当前Q已满足同策略Bellman固定点关系。这不是增加训练轮数就一定能解决的结论。

两快照熵bonus均值0.077306，有效折扣熵贡献0.076533；对照抽样一致。采样请求RMS约0.068226，而训练请求全部为0。after中，随机请求相对零请求的target-Q平均绝对变化约0.002953；该小变化可能反映有限动作覆盖下的弱敏感性，**不能解释为实际残差动作无影响或Q已校准**。

## 边界覆盖与限制

共同3871行及after完整4076行均没有terminated/truncated样本，对应分层只报告samples=0。逐回合检查旧原始轨迹发现11个末尾均为learning_terminated=1；前10个可用summary都标记safety_failure。原始步数与replay行数分别相差10/11，和回合末尾计数一致；这些旧session处于控制时延修复之前。

现有agent把学习有效性与安全mask共同用于replay准入，这解释了为何不能从这组replay检查终止目标。**缺少边界覆盖不等于所有安全失败行都应直接放回replay。**原始末尾状态、安全回退、请求/执行动作和终止语义需要一起审核；不能为降低诊断误差放宽安全规则。

这只是development101的历史训练内数据，未覆盖新起点、107、自然完整任务边界或实际随机SAC策略的多步回报。actor/alpha没有学习；双Q分歧变小也不是价值正确性的充分证据。原aborted记录和原首轮报告保持不变。

## 工程改进与后续

新增只读soft诊断工具，固定随机数、显式区分两种bootstrap网络、核验actor/alpha和共同replay前缀，并报告目标采样波动、有效熵贡献和空边界分层。没有改变在线SAC目标、网络、预热次数或安全mask。

测试覆盖终止时同时屏蔽Q与熵、Q使用请求动作、归一化熵不随动作限幅缩放、固定种子复现、参数/RNG不变及非法mask拒绝。首轮测试夹具错误共享terminated/truncated张量，已改成独立张量；最终全量回归 **1629 passed, 7 skipped**。

下一步先登记并采集控制时延修复后的SAC完整任务数据，核对终止/安全mask及残差动作覆盖，再决定是否需要改变训练配置。当前不增加LN、UTD或训练轮数，不以旧timeout数据直接放宽replay准入，也不宣布G1-B路线晋级。
