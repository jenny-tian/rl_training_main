# G1.1 目标域预热、恢复与学习隔离

本轮完成4000步目标域预热的工程实现与真实执行验证。LoRA预热后才更新actor；SAC完成1000次critic-only更新，日志问题已修复并通过独立短session验证。**这不是G1路线晋级结论**：控制超时仍影响SAC，完整回合留出价值诊断尚未完成。

## 执行与预算

沿用共同base SHA `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`，development rough_4cm seed101、learner seed7、perturbed-v1、速度0.3、任务deadline2000。初态证书仍有效。所有运行均预登记于[实验登记表](lite3_experiment_registry.md)，最多同时使用两张GPU。

| 运行 | 适应控制步 | 地形评估步 | 平地步 | 状态 |
|---|---:|---:|---:|---|
| LoRA主验证 |4512|64|320|全部完成；重复评估新增0步|
| SAC主验证 |4087|0|0|最后回合日志失败，保留aborted与全部成本|
| SAC日志修复回归 |512|2|251|保存与恢复通过；评估被控制超时提前结束|
| 合计 |9111|66|571|**9748步，uncertain=0**|

SAC修复回归是独立工程session，warmup缩短至256步、min_replay=128，仍执行1000次critic-only更新；它不是新的效果候选。修复回归的评估次数均只用一个预定起点，重跑evaluation/retention命令未补抽起点或新增transition。

## 实现与检查

- `run_adaptation_session.py`默认target warmup=4000；底层worker默认0保持旧入口兼容，启用时须有managed finite-horizon session。预热状态与参数写入checkpoint，拒绝缺失状态或配置变化的续接。
- LoRA执行冻结base均值，只优化critic；使用lambda=1块回报，未结束的块有bootstrap。预热边界强制清空旧样本，随后actor仅用新on-policy rollout。critic-only阶段不更新actor优化器或策略版本，拟合结果保留到稳定状态及检查点。
- SAC采集阶段不更新critic/actor/alpha，也不累计后续更新额度。收集结束且replay至少有`max(batch_size,min_replay_size)`条有效记录后，执行最多1000次critic-only minibatch；拒绝的预热更新阻止actor进入学习。保留n=1、当前网络和novelty/安全门控。
- 启用新预热协议时，SAC联动actor的更新也扣除其critic更新额度。诊断仍使用原critic/actor额度0.5/0.005，未声称已完成UTD=1或actor每两次critic更新的后续比较。
- 无探针的session允许selection预算为0，仍禁止超过预算的探针。未计算的actor/alpha loss复用既有日志语义写成JSON `null`。

真实轨迹核验：LoRA和SAC主验证各4000个warmup控制步中，requested/mean/executed action与base的最大差异均为**0**。base文件SHA未变，终止bootstrap与奖励分项重算通过。PPO暖机样本没有留在后续actor buffer中。

## 已读到的学习信号

LoRA在两个2000步物理trial中完成16次critic-only块拟合，共128次minibatch更新，actor更新为0；接下来的512步才发生2次actor更新、16次critic minibatch更新。16个预热块中14个训练内RMSE下降，目标方差范围约88.36–1660.75；这些是同批数据上的诊断，不是完整回合验证集结果。冻结地形/平地阶段actor和critic更新均为0。

SAC主验证checkpoint记录4000步采样和1000次预热，actor/alpha优化器状态始终为空，alpha保持0.01。预热critic loss首末约4.817→0.039，mean TD error约1.591→0.114，mean target Q约1.591→8.253。此后87步产生43次普通critic更新，checkpoint的critic Adam步数为1043。SAC本轮没有实际actor更新；通过预热门槛只表示允许后续novelty/安全门控决定是否更新。

上述Q诊断是训练内、含熵的soft Bellman目标残差，target critic在移动，不能当作固定策略的独立Q真值或冻结任务收益。alpha=0.01沿用G0低温度工程配置，本轮没有做alpha优越性比较。

## 保留的失败与修复验证

SAC主验证共11个物理trial触发20 ms控制计算超时。暂停LoRA并让SAC单独运行后仍有超时，因此不能仅归因于本轮两worker并发，也没有放宽安全阈值。

最后一个SAC trial已保存checkpoint和完整206步trace、1000条预热更新记录，但新日志路径直接序列化未计算loss的NaN占位，严格summary JSON失败。账本保留该trial为aborted，不追认完成、不静默续接该checkpoint；其余证据经离线核验保存为派生报告，原始文件未改写。

修复复用了`record_update_result`的null处理。新session首trial完成256步预热及1000次critic更新并成功保存；第二trial恢复后预热更新为0，普通critic更新继续。冻结地形评估和retention均为零更新、零replay写入，重复调用新增0步。它们分别在2步、251步因控制超时结束，**SAC平地保持comparison_eligible=false**，不能用其仅1个稳态样本作保持能力结论。LoRA平地320步提供70个稳态样本，XY速度MAE约0.02455 m/s；仍未实施正式的配对保持验收。

## 可复核产物

- [验证结果](lite3_g1_warmup_validation_20260929.json)：阶段步数、更新次数、源码哈希、warmup动作等价、SAC中止证据及诊断尺度。
- 原始运行、完整轨迹、更新日志、检查点及验证程序：`artifacts/g1_target_warmup_20260929/`。验证程序为该目录下`validate.py`；原始SAC失败证据另见`aborted_sac_validation.json`。
- [日志修复补丁](lite3_g1_warmup_logging_fix_20260929.patch)：用`git apply --unidiff-zero --reverse`对本次最终源码反向应用可重建两条主验证所用源码；重建后的全体worker/组件SHA已与原manifest核对一致。修复回归使用最终源码。不要把旧source-bound session换成新源码继续运行。
- 测试：**1592 passed, 7 skipped**；覆盖actor/alpha冻结、PPO样本隔离和critic保留、预热恢复、失败梯度回滚、联合更新计费、入口参数、零探针预算及严格JSON日志。

## 下一步

1. 定位SAC控制时延尖峰，区分首次内核/内存整理、同步和系统调度；维持20 ms阈值，保留失败成本，先获得可解释的完整采样窗口。
2. 按G1-A在两个调试地形各收集6个完整冻结策略回合，预先固定4/2训练验证划分，比较常数、时间特征与学习critic。首轮不从本次块内RMSE推断留出EV。
3. 在固定策略/alpha条件下做SAC独立soft-target残差、偏置和尺度诊断；检查控制超时导致的无效replay行及其边界影响，再决定UTD/LN/n-step候选。
4. 候选晋升、相对平地保持阈值和最终封存协议仍未锁定；本轮没有生成sealed种子或宣称任何路线晋级。
