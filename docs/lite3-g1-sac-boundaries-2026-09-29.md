# G1.6 时延修复后 SAC 任务边界核验（2026-09-29）

**当前 SAC 能保留本组自然任务终止，不需要依据旧 timeout 数据放宽 replay 安全准入。**两个地形、4个预定回合共6501控制步全部进入replay；4个终止行均保留、bootstrap=0。未发生控制超时，最大控制计算耗时7.614 ms，阈值仍20 ms。本组零学习，不证明SAC价值函数或适应策略已合格。

## 协议与执行

[预登记协议](lite3_g1_sac_boundary_protocol_20260929.json)固定development rough_4cm101/107、learner seed31、perturbed-v1，每地形两次attempt，任务deadline与每回合collector上限均2000。两个reset seeds为1375524730和86170031，未因成败替换。GPU0/1各一个worker，总实际控制步上限8000。

为验证真实replay路径，使用冻结确定性base的warmup采集：managed B=4001、warmup=4001，而两次attempt最多4000步，故不会触发critic预热或actor/alpha更新。replay5000、min256、n=1、alpha=.01；多出的1步只是防止边界处自动启动学习，没有执行。该4001设置是本次采样诊断契约，不替代正式适应默认4000步预热。

协议和新增日志字段于采集前提交推送`016e012`。逐步日志现在记录critic/actor实际replay写入行数及安全准入mask。没有改变动作、安全过滤或目标值计算；新增mask读取发生在控制计时窗口外的日志阶段。

## 结果与逐行核验

| 地形 | 回合 | 任务终止原因 | 控制步 / 写入replay | 终止保留 | 终止bootstrap | 最大控制耗时(ms) |
|---|---|---|---:|---|---:|---:|
| 101 | 00000 | lateral_violation | 1833 / 1833 | 是 | 0 | 5.975 |
| 101 | 00001 | task_timeout | 2000 / 2000 | 是 | 0 | 4.773 |
| 107 | 00000 | physical_failure | 1404 / 1404 | 是 | 0 | 4.149 |
| 107 | 00001 | physical_failure | 1264 / 1264 | 是 | 0 | 7.614 |

实际 **6501 / 8000 控制步**（101为3833、107为2668），uncertain=0；四回合均完整task terminal，无collector/physical截断，无补抽。所有请求、均值和执行动作均与base一致（atol=1e-6）；replay请求全部为0，仍不提供非零残差动作价值的覆盖。

审计将每步实际写入数与安全准入/paused标志对齐，按被接受的轨迹行重建replay，并按replay存储dtype量化后逐位比较15个字段：当前/下一历史、当前/下一frame、base/next base、请求/执行残差、奖励、终止/截断、bootstrap、discount、safety mask及episode ID。两次checkpoint的累积内容、游标和summary写入计数一致。task terminal的下一历史清零后保留真实末态frame；没有把reset后的frame作为该终止目标。

actor、critic及alpha优化器状态均为空，更新attempt为0；跨回合actor/critic/target critic及温度张量逐位不变。101/107累计warmup采样分别3833/2668，均未完成4001门槛，critic warmup更新数为0。

两次attempt后，通过既有账本API锁定evaluation阶段，阻止继续适应。再次调用带阶段检查的锁定操作新增0步；没有执行evaluation或retention。剩余168/1333适应预算保持未使用，不当作已完成预热或正式学习session，也不把诊断数据作为后续正式适应的免费样本。

## 对旧覆盖问题的判断

[G1.5](lite3-g1-sac-soft-diagnostic-2026-09-29.md)的旧replay没有终止行，原始轨迹中却有安全失败终止；该批采样在控制时延修复之前。本次保留了lateral_violation、task_timeout和physical_failure三类自然任务终止，说明旧覆盖缺口不能推广成“实现始终丢弃terminal”。现有安全mask保持不变。

本组没有completion或physical truncation实例，也没有非零残差请求；四个开发回合不证明所有边界情况或新策略价值都已覆盖。下一步应先登记新的学习/动作覆盖诊断，检查修复后SAC在真实非零请求下的数值和收益，再考虑LN、UTD等训练配置；不凭旧timeout样本增加训练或放宽安全规则。

## 复现与测试

采集目录：`artifacts/g1_sac_boundary_20260929/terrain-{101,107}/`；完整审计及输入哈希：其`audit/report.json`。[精简验证JSON](lite3_g1_sac_boundary_validation_20260929.json)保留完整报告SHA256、成本、每回合指标和源版本。

对账入口为`docs/analysis/audit_sac_boundaries.py`，新增测试验证terminal下一历史清零但保留末态、truncation仍保留未reset历史。全量回归 **1630 passed, 7 skipped**。首次真实对账即通过，未改写原始轨迹、checkpoint或失败记录。
