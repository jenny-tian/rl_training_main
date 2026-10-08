# G1.8 预热增益修复的物理复核（2026-09-30）

**等待控制增益为0时，修复后的SAC预热实际使用了注册目标增益0.5，且检查点恢复没有重复预热。**两个地形共1152控制步、0控制超时；这是短预算工程验证，不是正式4000步预热效果或G1路线晋级。

## 协议与隔离条件

[预登记协议](lite3_g1_sac_warmup_gain_protocol_20260930.json)固定development rough_4cm101/107、learner43、perturbed-v1。每地形新source-bound session：B512、warmup256、trial256、task deadline2000；先执行一个trial并检查checkpoint，再恢复剩余256步，最后执行一次64步冻结检查并重复该phase。未执行retention。

沿用Huber1、alpha=.01、replay5000/min256、n=1及1000 critic-only更新。novelty calibration设为1024，使这512个适应控制步保持等待gain0，用来触发并隔离上轮发现的问题；该设置不改变正式默认预热方案，也不是探索或学习效果候选。物理采集前已提交推送`5430457`。

日志新增`control_residual_gain`，与既有`residual_gain`并列记录；前者是当时控制状态，后者是预热更新使用的注册目标增益。核心修复仍为G1.7提交的显式critic-only增益参数，本轮未重新调整学习算法或安全阈值。

## 物理结果

| 地形 | 首trial / critic更新 | 预热目标gain / 控制gain | 恢复trial / 新增critic更新 | 冻结窗口 / 更新 | 最大控制耗时(ms) |
|---|---|---|---|---|---:|
| 101 | 256步 / 1000次 | 0.5 / 0（全部1000次） | 256步 / 128次 | 64步 / 0次 | 7.991 |
| 107 | 256步 / 1000次 | 0.5 / 0（全部1000次） | 256步 / 128次 | 64步 / 0次 | 6.525 |

实际 **1152 / 1152 控制步**，uncertain=0，无失败替换、无控制超时，20ms阈值不变。两地形共2000次预热更新及256次常规critic更新，actor/alpha更新均为0。

首个checkpoint的critic优化器步数为1000，warmup计数为256/1000；恢复后优化器为1128，但warmup计数仍为256/1000，第二个trial没有任何新的warmup日志。两次物理trial的输入/输出checkpoint链一致，actor和温度状态没有改变。

所有物理动作与base一致（atol=1e-6）。冻结阶段控制gain为0.5，但actor未更新，因此均值残差仍为0；不能把它解释为已学残差的效果。冻结actor before/after及输入actor张量逐位一致，更新attempt、critic更新和replay写入均为0，replay大小不变，且按设计未写完整learner checkpoint。两次重复evaluation均新增0步/0试验。

六个窗口均以collector_cutoff结束，不作为完整任务MC、完成率或平地保持证据。本次证明目标gain与等待控制gain分离、恢复不重复预热和短冻结执行契约；不证明4000步数据上的价值精度或长期收益。上轮的控制超时和修复前gain数据继续保留，不被本轮结果覆盖。

## 有效actor更新次数的只读核对

同时检查G1.7原始manifest、逐步actor replay写入数和回合末额度，未运行新优化器或更改调度。原actor额度为每有效actor transition 0.005，即200条才获得一次更新；critic额度为0.5，在同样有效行数条件下名义比率为100:1，actor额外门控还可能降低实际更新次数。

| 地形 | 预热后有效actor样本（按物理回合） | 分回合更新上界 | 实际actor更新 | 各回合末剩余额度 |
|---|---|---:|---:|---:|
| 101 | 65、140 | floor(0.325)+floor(0.700)=0 | 0 | 0 |
| 107 | 579 | floor(2.895)=2 | 2 | 0 |

观察数与该上界一致，没有证据表明更新器额外漏执行。101若不清除跨回合额度，单从总数看理论floor为1，但这不是实测更新次数，也不表示应直接改变清除规则。不能将名义1000个后续控制步当作足量actor训练，更不能仅因更新少就先提高学习率。

下一步应依照主计划单独登记更新调度对比，明确记录有效样本、额度、更新次数和计算成本；保持Huber、目标gain和安全契约固定，不同时改变LN、学习率和n-step。本轮没有启动新的调度候选，也不改变既有G1接受条件。

## 证据与回归

完整报告和输入哈希位于`artifacts/g1_sac_warmup_gain_20260930/audit/report.json`，actor覆盖核对位于同目录上一级`actor_coverage.json`。[精简验证JSON](lite3_g1_sac_warmup_gain_validation_20260930.json)保留完整报告哈希、源提交、每trial计数和覆盖分析。

物理审计逐一核对原始trace的base动作等价、日志/账本步数、目标与控制gain、optimizer步数、空actor/alpha状态、检查点恢复链、冻结产物及重复调用。全量回归 **1636 passed, 7 skipped**；首次实际对账通过，无新增采样重试。
