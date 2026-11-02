# 八环境短时运动/reset 检查：限定范围内通过

状态：`passed_within_registered_short_probe`，2026-11-02。

经用户单独同意，从干净、远端一致的提交
`c61d455e54c5317b73de269cb19dcf25448bc102` 执行唯一一次八环境检查。
父进程验收和从归档副本独立重放均通过；**没有重试、扩大规模或 PPO 更新**。

## 实际完成

| 项目 | 结果 |
| --- | --- |
| 场景 | 8 环境、1 行 shared bank、六类/15 个 L1 cell 保留，cuda:0 |
| 控制与物理步 | 96 全局控制步、384 个 5ms 子步、每环境1.92s，共768诊断transitions |
| 原始采集 | 96条控制记录，384次接触采样、384次运动采样、96次双scanner检查 |
| 接触拓扑 | 136个机器人刚体 ×160个过滤目标，含全部24个ground colliders |
| 详细接触 | 12186条pair明细，缓冲峰值55/65536，未饱和；跨环境接触0 |
| 接触正对照 | env0–7均取得ground points，合计6338个采样接触点（不是独立事件数） |
| reset | 三批注入超时，共8个回合结束；无普通失败回合，无compound termination |
| 恢复首步 | 三批均记录，pending恢复IDs为空 |
| 动作 | 原始动作裁剪127/768（16.5365%）；额外SafetyFilter投影0，pause 0 |
| 结果与清理 | status=passed、父进程accepted=true；环境close返回，应用close进入关闭后进程退出 |
| 学习 | 未构造PPO/critic/optimizer，权重未更新，Source PPO更新0 |

八环境ground接触点计数依次为 `762,788,780,817,809,823,778,781`。
详细力与法向重建pair矩阵的最大绝对残差为 `1.52587890625e-05 N`，
pair矩阵合计与net力残差为0；均满足原有 `atol=rtol=1e-4` 契约。
原始count、pair/net张量与详细接触点已从归档逐子步重建，未靠net力单独判断隔离。

## 三批超时与状态不变性

| 控制步 | 注入env IDs | 捕获terminal steps | 结束后episode counters |
| --- | --- | --- | --- |
| 24 | `[0]` | `[2000]` | `[0,24,24,24,24,24,24,24]` |
| 48 | `[1,4,7]` | `[2000,2000,2000]` | `[24,0,48,48,0,48,48,0]` |
| 72 | `[0,2,5,6]` | `[2000,2000,2000,2000]` | `[0,24,0,72,24,0,0,24]` |

这8个回合的唯一活动项都是 `source_training_timeout`，未触发安全、非法接触、
姿态、tile-bound或completion终止。注入通过预置counter=1999完成，
**不表示实际走了2000步或40秒**。本轮没有在GPU上实际触发普通失败路径；
普通失败继续采样的证据来自上一阶段明确标注的CPU故障注入。

每批reset前后快照独立核对未reset环境的root/joint/action/command/history/sensor
缓存与filter/lifecycle记忆不变，选中环境清理、origin及default-root spawn一致。
独立allocator回放验证了票据与reset/transition计数。恢复首物理步分别为97、193、289，
对应注入的三个ID集合，运动历史按reset后状态重建。
最终episode counters为 `[24,48,24,96,48,24,24,48]`。

## 运动与扫描证据的实际边界

所有采样的17刚体状态、扫描值及保存动作均有限；两套scanner命中同一共享ground，
命中属于分配tile，基座scan时间及XY位置匹配当前root。每个substep重放了link-origin
包络与预测谓词，观测到的当前包络均在所属tile内。

离散样本极值（描述性统计，不是新的阈值）：

- 最大5ms link位移 `0.03606547 m`，最大link线速度 `7.11471175 m/s`；
- 最大有限差分link线性加速度 `1233.92046 m/s²`；
- 当前包络最小边界余量 `0.79432535 m`，最小基座地形相对净空 `0.26957610 m`；
- 最大reset相对root前进量仅 `0.18125296 m`。

这是短时、带std=1探索噪声的完整性探针，主要仍在入口区域；
不据此宣称旧策略已学会地形、通过9m完成条件，或提供六类技能成功率。
5ms采样和有限差分不是连续时间界限，不能用本轮极值替代完整G3的动态范围审查。

## 实际分配台账

| Cell label | Episode assignments | Transitions |
| --- | ---: | ---: |
| flat | 4 | 192 |
| multi_scale_rough | 7 | 360 |
| slope_downhill | 1 | 24 |
| slope_uphill | 1 | 72 |
| slope_rough_uphill | 1 | 48 |
| slope_rough_cross_slope_right | 1 | 24 |
| stairs_descending | 1 | 48 |
| 其余8个cell | 0 | 0 |

合计16次episode分配、768个transitions，全部归属逐步重建一致。
16张票不足一个120-ticket平衡块，不评价长期配比；标签也不证明已经接触完整目标区域。
未观察的cell和friction家族仍保留在配置里，没有因为本轮未分配而删除。

## 资源、时限与清理

启动前217项生产/资产输入及35项安装依赖哈希通过，运行后再核对均一致。
checkpoint和模型revision未变。几何/采样/噪声seed分别30000/131001/132001，
PPO根seed130001；reset标识40000仍是bookkeeping。

监督器计时 `162.042283 s`，关闭前报告计时 `158.618573 s`，未触发120s启动或240s总超时。
地形生成0.281526s、scene创建1.114744s、simulation start 1.231922s。
运行含逐步GPU读回、数值检查和fsync大体量记录，不能换算成训练吞吐或G4结论。

实际启动门读取GPU0可用22053MiB，满足12288MiB下限；32次约5秒间隔的卡级采样
占用峰值7926MiB、最小可用16199MiB。GPU0已有/新增其他用户任务，运行允许共享；
退出后卡级占用5101MiB，不能写成测试进程占用，也不能宣称显存恢复到18MiB。
父进程和独立进程组检查确认本次group 2121305无残留，未操作其他用户进程。
清理记录persistence_errors为空；application=`closing`表示已请求关闭后进程退出，
不声称Python层close返回。

日志保留32条足部visual USD引用警告和Kit IOMMU/P2P提示；未据此授予视觉资产资格。
无Python traceback、采集/监控错误或超时记录。

## 原始归档与复核

`raw/`保留13个原始运行文件（events以无损gzip保存）及Kit/Isaac Lab完整日志。
events原始 `214893213` bytes，gzip为 `16305381` bytes；
`provenance.json`核对原始解压字节SHA、压缩文件SHA和其余逐字节副本。
`review.py`仅CPU读取这些副本，调用独立流式验收器并补充统计，结果为`review.json`。
摘要中的环境ID按JSON字符串key规范比较；没有改写原始记录或参数。

当前结论为指定代码/seed/八环境/96步的短时运动/reset完整性通过。
**G3/G4整体、120环境、live critic参与PPO更新及正式训练均未资格化或执行。**
本阶段生产代码没有修改，因此没有为归档重复上一阶段410项CPU回归。

## 下一阶段提案：Source PPO入口与数值试训准备，仅CPU，待同意

为继续向六类Source学习推进，下一步建议把已有actor-only初始化、SourcePPO
终止观测bootstrap、执行安全桥和TransitionBudget接到专用训练入口，完成CPU验证：

1. actor45 / critic393、三个登记seed各自fresh critic/optimizer/计数；保留全部六类15cell。
2. 每次rollout前限制预算，记录已采集交互与更新数；保持候选120env×24step的
   91/364/728节点，不能从本轮八环境结果擅自改预算或自动放大。
3. 加入loss/gradient/模型有限性停止、全新输出目录、明确拒绝未实现的自动resume，
   完整记录reset和transition曝光。用合成CPU数据验证更新/拒绝路径。
4. 明确数值试训前仍待满足的选定规模运行/吞吐资格、source-only验证与checkpoint
   选择锁定条件，不把本轮短探针改写为正式训练lock。

该提案不启动Isaac Sim、GPU或真实Source PPO，不读取开发/封存评估反馈。
实现、测试、提交汇报后再次暂停，实际数值试训须单独批准。
见`next_stage.json`。当前阶段已停止，等待用户决定。
