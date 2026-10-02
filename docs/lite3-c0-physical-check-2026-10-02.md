# C0 固定代码物理执行验证（2026-10-02）

工程验证通过，但尚未证明地形收益或学会步态。两个地形101/107、seed71，base/zero/前后腿±1共6臂，各执行64步任务和64步平面窗口：24个窗口全部完成，实际1536/1536控制步，uncertain=0，无跌倒、安全违规或控制超时。

两地形两phase的base/zero初始root、root_state、base_action、executed_action、reward和learning_terminated逐位一致，最大差异0。非零请求满足原限幅与gain .5下的L*tanh(Bz)公式，最大绝对误差7.45058e-9；实际执行残差已保存，未放宽安全投影或残差速率限制。

全部冻结actor前后张量一致，actor/critic更新和replay写入为0，base文件哈希不变。随后24次completed phase重复调用，所有session_summary内容完全不变，新增试验/控制步均0。

本组所有控制步记为新工程成本；child ledger使用evaluation/retention名称不意味着这是学习后的独立性能评价。64步不足完成任务或形成750个平地稳态样本，不据本组宣称平地保持合格、headroom、稳定步态或算法晋级。

[协议](lite3_c0_physical_protocol_20261002.json)，[精简验证](lite3_c0_physical_validation_20261002.json)。原始轨迹、24次重复记录与完整哈希：`artifacts/c0_fixed_code_physical_20261002/`。代码此前完整回归1680 passed/8 skipped，新增短窗口调度测试1 passed；本次实际运行和审计均一次完成，无失败重抽。

下一步按主计划单独登记有限完整回合headroom：固定二维代码网格初筛、预定义排序锁定候选、新起点与base比较。只有在完整回合中证实收益，才投资后续回合级学习与独立确认；本次不扩大动作幅值或增加代码维度。
