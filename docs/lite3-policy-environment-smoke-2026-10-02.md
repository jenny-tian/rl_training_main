# 状态相关神经策略环境工程实测

新环境工程检查通过，尚未开始策略训练或证明步态收益。8环境×(64正常步＋1人工截止步)=520控制转移，charged=confirmed=520、uncertain=0。正常窗口终止0次；最后明确设置截止计数的工程步8环境均终止，timeout bootstrap标志不存在或全零。人工截止案例不当作自然回合或训练数据。

动作网络输入45维，顺序为角速度、重力投影、速度指令、关节位置、关节速度、上一动作；尺度分别.25、1、1、1、.05、1。critic为48维，额外读取基座线速度；无高度扫描或动作相位输入。关节顺序、动作尺度和clip已保存到验证JSON。原actor参数成功复制，critic和optimizer未导入；65次确定性执行后actor/critic张量未变、优化器状态为空、base哈希不变。

场景有8个固定列，2列平地、3列seed101崎岖、3列seed107崎岖；环境terrain_types为0到7。使用generator的固定列模式，difficulty固定1且无等级更新；这不是动态地形课程，也不保证与旧随机difficulty赛道mesh逐位相同。新训练奖励采用原平地物理项、横向偏离代价和+5/-2终止事件，区别于最终原赛道评测奖励。姿态和非足接触终止沿用原崎岖配置，动作仍使用Lite3物理尺度和clip。

attempt01在记录观察配置时因API不存在而失败，控制步0；仅修复记录接口后新建attempt02完成，没有覆盖失败。该环境的进程退出码不能单独代表成功，必须读取持久report/ledger状态。两个attempt合计仍520步。

[协议](lite3_policy_environment_smoke_protocol_20261002.json)、[验证结果](lite3_policy_environment_smoke_validation_20261002.json)。原始日志和65×8数组保存在`artifacts/policy_environment_smoke_20261002/`。数组的returned_root_pos是step返回后的状态，可能已自动reset，不能冒充终止前状态。相关CPU组件测试12 passed。

下一步实现先计费后执行的PPO微调入口并登记首阶段预算；新训练使用新文件保存权重，保留原基础模型。冻结同起点任务对照、完整平地保持、多独立训练与步态证据仍待完成，不根据本次初始化成功宣称已经学会。
