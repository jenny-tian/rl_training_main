# 多地形与抬脚奖励的官方实现对照

2026-11-16。本阶段完成官方资料检索、四个代码库共 26 个固定版本文件的核查，以及既有训练日志和 480 个上阶回合的离线计算。新增训练更新 0，新增仿真回合 0，模型、历史奖励、参考项目和部署代码均未修改。

结论：提高抬脚目标值得做一个独立对照试验，但公开代码没有支持只调高一个数值就一定能解决上阶。当前模型已经能完成大多数 5 cm 测试；剩余问题还涉及后脚推进、航向/横移和起步恢复。与公开实现相比，台阶训练份额和策略可用信息也有明显差异，这些目前是待验证因素，不能直接认定为故障原因。

## 研究范围与来源

只采用作者或项目组织的仓库、论文摘要和官方项目说明。没有执行下载的代码，没有将非官方 DreamWaQ 复现当作作者实现。代码中的默认值、机器人专用覆盖、训练脚本覆盖和实际奖励函数分别核对；不把仓库里存在某个地形生成函数当作训练已经启用了该地形。

| 实现与对应研究 | 本次固定版本 | 用途 |
|---|---|---|
| legged_gym / ANYmal C，Rudin 等，Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning | `8fa29acc6fd1910c3d9659eef6310bdd301cde0a` | 分块混合地形、按行进距离升降级 |
| HIMLoco / Go1，Long 等，Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response | `ef289acaa62795009363b7b819c9186690630441` | 本体感知历史、速度及隐状态估计、摆脚奖励 |
| Isaac Lab / Go1 rough | v2.3.0，`3c6e67bb5c7ada942a6d1884ab69338f57596f77` | 与本项目接近的地形和训练框架 |
| Walk These Ways / Go1，Margolis、Agrawal，Tuning Robot Control for Generalization with Multiplicity of Behavior | `0e7236bdc81ce855cbe3d70345a7899452bdeb1c` | 显式抬脚高度命令与步态相位；默认脚本不作为混合台阶训练示例 |

论文入口：[Learning to Walk in Minutes](https://proceedings.mlr.press/v164/rudin22a.html)、[HIM](https://arxiv.org/abs/2312.11460)、[Walk These Ways 官方论文与项目页](https://gmargo11.github.io/walk-these-ways/)。前者研究大规模并行训练和地形课程；HIM 以本体感知估计速度及系统响应；WTW 训练可由命令选择的多种步态，以便在新任务中调整行为。本次具体数值均以固定代码版本为依据，不把这些提交等同于原论文实验版本，也不宣称是最新版本。论文展示的硬件能力、训练耗时不能直接换算成 Lite3 的保证。

全部源码路径、固定提交链接和 SHA256 见 [upstream_manifest.json](upstream_manifest.json)。快照位于临时目录，可由 [采集脚本](../../scripts/tools/reference_multiterrain_research_collect.py) 从固定版本重新下载。长期归档的是来源清单与分析，不复制整个外部仓库。

## 地形份额、布局和难度

| 实现 | 配置中的地形份额 | 布局 | 台阶设置 |
|---|---|---|---|
| legged_gym ANYmal C | 光滑坡 10%、粗糙坡 10%、两种台阶 35%+25%、离散障碍 20% | 8×8 m，10 行×20 列 | 阶高 `0.05+0.18×difficulty` m，踏面 31 cm，中心平台 3 m |
| HIMLoco Go1 | 光滑坡 10%、粗糙坡 20%、两种台阶 30%+30%、离散障碍 10% | 8×8 m，10×20 | 同一阶高公式，踏面 30 cm，中心平台 3 m |
| Isaac Lab Go1 rough | 两种台阶各 20%、箱体 20%、随机粗糙 20%、两种坡各 10% | 8×8 m，10×20 | 配置阶高范围 5–23 cm，踏面 30 cm，中心平台 3 m |
| 当前 Lite3 zero10100 | 平地 20%、粗糙 20%、坡 20%、台阶 20%、摩擦变化 10%、粗糙坡 10% | 8×8 m，10×20，六类十子型 | 当前低台阶分支为 2–5 cm，踏面 30 cm，中心平台 3 m |

来源：[LG 配置][lg-config]与[地形实现][lg-terrain]，[HIM 配置][him-config]与[地形实现][him-terrain]，[Isaac 地形配置][il-terrain]与[Go1 覆盖][il-go1]；本地来源为 [reference_six_terrain_cfg.py](../../scripts/tools/reference_six_terrain_cfg.py)、[reference_low_stairs_cfg.py](../../scripts/tools/reference_low_stairs_cfg.py) 及运行时 `configuration.json`。

有三个容易误读的细节：

- LG 和 HIM 的课程行难度是 `row/10`，最大名义难度为 0.9，因而这份十行配置的名义最大阶高为 21.2 cm，随后还受高度场量化影响。公式在 difficulty=1 时得到 23 cm，不意味着生成的最高一行恰为 23 cm。Isaac 采用 `(row+随机数)/10`，每行内部有难度扰动，同样不能把配置端点当作实测最大高度。
- 从中心平台向外走时，凹形台阶对应上阶，凸形台阶对应下阶；本地名称按机器狗向外移动定义。每块地形也能双向走，分配到上阶列不代表所有动作都在上阶。
- Go1 专用覆盖把 Isaac 粗糙高度幅度范围设为 1–6 cm、离散高度步长 1 cm，箱体范围改为 2.5–10 cm。当前六类地形已复用参考项目的这类粗糙设置。噪声参数不是每个接触点的实测峰谷差；实际表面还受采样、下采样、插值影响。

地形生成器会把各子地形 `size` 统一设为总配置的 8×8 m；配置序列化里子类默认 10×10 m 不能单独据此判为尺寸错误。见 [Isaac 生成器][il-generator] 的初始化与课程生成代码，本机 `/home/tjx/IsaacLab/source/isaaclab/isaaclab/terrains/terrain_generator.py` 同样存在该覆盖。

WTW 的 `train.py` 先调用 `config_go1`：其第九类地形权重为 1，噪声幅度为 0，地形课程关闭；脚本的 tile 高度范围也为 0。因此，这份默认脚本主要训练平地上的多种步态。它有台阶泛化展示，但不能引用它作为默认用了大量台阶训练的证据。见 [WTW Go1 配置][wtw-go1]、[训练脚本][wtw-train]、[地形生成器][wtw-terrain]。

## 策略看到的信息差异更大

| 实现 | 推理时 actor 的输入路径 | 对本项目的含义 |
|---|---|---|
| legged_gym ANYmal C | 48 维状态/命令/上次动作 + 187 个高度采样，共 235 维 | 直接包含地形和机身线速度，不能只复制其奖励后假定信息条件相同 |
| Isaac Lab Go1 rough | policy 组直接含 `base_lin_vel` 和 `height_scan` | 地形高度并非仅供 critic 使用 |
| HIMLoco Go1 | 45 维本体观测×6 帧 → 估计速度 3 维和隐状态 16 维；actor 使用当前 45+3+16 维 | 推理不用地形扫描，但需要历史缓冲、估计器及对应训练目标 |
| 当前 Lite3 | 单帧 45 维：角速度、重力方向、命令、关节位置/速度、上次动作 | actor 无高度扫描、无机身线速度、无堆叠历史或递归状态；393 维 critic 的特权信息不会在部署时直接传给 actor |
| WTW 默认训练 | 70 维单帧、30 帧历史配置，含步态命令和时钟等条件 | 抬脚高度属于策略可见的命令，改变接口需同步训练和部署 |

来源：[LG 观测实现][lg-robot]、[Isaac policy 组][il-env]、[HIM 观测缓冲][him-robot]、[actor][him-actor]和[估计器][him-estimator]、[WTW 训练覆盖][wtw-train]；本地活动观测项及历史长度见 [local_audit.json](local_audit.json)。本地维度另可追溯到已有模型/评估配置。上次动作已含有限的历史信息，不能把当前 actor 说成完全没有过去信息。

这说明 HIM 的思路值得作为较大改造方向，但不证明当前单帧模型不可能上阶。既有成功轨迹已经排除了这种绝对判断。加入六帧历史和估计器也不是替换几项配置即可兼容当前 checkpoint，需要新的网络、损失、导出和部署缓存，不能混入本轮单项奖励实验。

## 抬脚奖励究竟怎样算

| 实现 | 实际生效的主要做法 | 能否直接复制数值 |
|---|---|---|
| legged_gym ANYmal C | 着地时奖励腾空时间减 0.5 s，四足求和，仅平移命令有效时生效；权重 1 | 没有同类固定净空高度目标；较长腾空不等同于保证足底过台阶边缘 |
| Isaac Lab Go1 rough | 腾空阈值 0.5 s，Go1 将权重改为 0.01；该配置没有专门净空项 | 应读取 Go1 覆盖，不能误用父类 0.125；也不能据此认定净空奖励无用 |
| HIMLoco Go1 | 机身坐标下足端 z 到 −0.20 m 的平方误差，乘机身相对足端水平速度，再四足求和；权重 −0.01 | −0.20 m 是相对机身高度，绝不是脚在地面下 20 cm；依赖机身高度、腿长、速度尺度 |
| WTW 默认训练 | 足端世界 z 跟踪 `抬脚高度命令×摆动相位+0.02 m`，按期望摆动状态加权、四足求和；权重 −30 | 未减去局部地面高度，不能原样用于累计升高的楼梯；需与接触、落点、步态条件配套 |
| 当前 Lite3 | `d=max(障碍顶面+0.025−足底,0)`，代价 `d²/(0.05²+d²)`，持续状态门控，四足均值，权重 −1，再乘控制周期 | 有界且带门控，与上面的平方误差/四足求和并非同一量纲和累计方式 |

来源：[LG airtime][lg-robot]、[Isaac 奖励定义][il-env]及[Go1 覆盖][il-go1]、[HIM Go1 奖励配置][him-go1]和[函数][him-robot]、[WTW 奖励函数][wtw-rewards]及[训练脚本][wtw-train]。当前实现见 [reference_low_stairs_clearance.py](../../scripts/tools/reference_low_stairs_clearance.py) 与 [reference_low_stairs_clearance_cfg.py](../../scripts/tools/reference_low_stairs_clearance_cfg.py)。

HIM 的实现按相对水平运动速度加权，没有此函数内部的显式摆动相位门控。WTW 设置 3–35 cm 的抬脚命令范围、2–4 Hz 频率范围，同时启用接触力/接触速度塑形、落点和滑动约束。这是配套设计的例子，不是建议 Lite3 统一抬到 35 cm。WTW 项目展示可变抬脚高度与台阶通过，支持研究这个方向，但无法证明该单项对我们的四个剩余失败有效。

当前持续净空规则在合法支撑时免罚，连续支撑 40 ms 后清除目标，墙面代理接触不清除。现有落脚正奖励要求峰值高于新阶面 15 mm，并满足足端/机身前进、双足支撑等条件及连续支撑 60 ms。提高净空目标与降低落脚资格门槛是两个不同实验；本轮建议先检验前者，15→5 mm 落脚候选暂缓，历史候选代码保留未启用。

## 课程、命令及 PPO 的可复用部分

LG、HIM、Isaac rough 的地形课程都按从原点走出的水平距离升降级：超过半块长度升级；低于命令速度乘回合时长的一半且未升级时降级。它们不是按足端完成某一级落脚来升级。当前参考课程已经采用同样的距离规则，并同步本地步态课程；全局平均等级高不能证明 5 cm 直线上阶评测通过。来源：[LG][lg-robot]、[HIM][him-robot]、[Isaac curriculum][il-curriculum]，本地 `/home/tjx/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`。

LG 和 HIM 都有航向误差转偏航速度的控制：增益 0.5，LG 限幅 ±1 rad/s、HIM 限幅 ±2 rad/s；HIM 的原始 yaw 采样范围不能直接当作航向模式最终命令。当前训练同样使用航向命令，增益 0.5、限幅 ±0.8 rad/s、10 s 重采样；已有评测中的世界 +x 航向反馈与此思想相符，但评测目标和命令过程仍需单列，不能把反馈成绩冒充固定零偏航输入成绩。

当前命令范围为 vx ±1.5、vy ±0.8 m/s、wz ±0.8 rad/s，参考配置站立比例 0.3，另有各 0.07 的零命令/纯横向/纯前后/纯转向抽样。特殊抽样会覆盖部分原生标记，不能简单把 0.3 与 0.07 相加当作实际站立份额。持续转向修改只作用于被选中的纯航向子集，其中一半保留航向、各四分之一改为固定左/右转，不是全体环境各有 25% 固定转向。当前命令课程关闭；本轮不调整命令分布。

| 参数 | 当前 Lite3 | legged_gym ANYmal C | HIMLoco Go1 | Isaac Lab Go1 rough |
|---|---:|---:|---:|---:|
| 并行环境 | 4096 | 4096 | 4096 | 4096 |
| 每次 rollout 控制步 | 24 | 24 | 100 | 24 |
| PPO epochs / mini-batches | 5 / 4 | 5 / 4 | 5 / 4 | 5 / 4 |
| 配置初始学习率 | 0.001 | 0.001 | 0.001 | 0.001 |
| gamma / lambda | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 0.95 |
| clip / entropy / desired KL | 0.2 / 0.01 / 0.01 | 相同 | 相同 | 相同 |

来源：[LG PPO][lg-config]、[HIM PPO][him-config]、[Isaac PPO][il-ppo]、本地 [local_audit.json](local_audit.json)。HIM 还有估计器训练，表中并未将算法视为完全相同。当前续训恢复的实际学习率为约 0.00057665，100 次更新后约 0.00038443，不能因为配置写 0.001 就重置优化器。ANYmal 使用 actuator network，HIM Go1 的 PD 为 40/1，当前 Lite3 的机器人参数也不能照搬。迭代上限不是各论文已执行的训练次数，rollout 长度不同也不能直接比较迭代数。

## 现有训练实际覆盖量

本次读取 zero10100 的最后 100 次更新报告：4096 环境×24 控制步×100 次，共 9,830,400 个环境控制帧。下表由 `coverage_by_column_level` 求和，未启动训练。

| 地形家族 | 环境数 | 该段环境帧 | 约占比 |
|---|---:|---:|---:|
| 平地 | 820 | 1,968,000 | 20% |
| 粗糙 | 819 | 1,965,600 | 20% |
| 坡 | 819 | 1,965,600 | 20% |
| 台阶 | 819，其中上 410、下 409 | 1,965,600 | 20% |
| 摩擦变化 | 410 | 984,000 | 10% |
| 粗糙坡 | 409 | 981,600 | 10% |

上阶列 984,000 帧中，205,499 帧至少一足出现净空不足代价，占该列约 20.88%；下阶列为 51,679/981,600，约 5.26%。这是惩罚生效覆盖，不能当作摆脚总数、跨阶次数或成功率。它也表明提高净空裕量仍可能影响下阶列中的局部向上运动，需要下阶回归。

近零命令帧为 2,937,135，占全部约 29.88%，判据为平移命令模长与偏航命令绝对值均小于 0.1，不能等同于严格零命令或实际站稳时间。现有计数没有按地形分解命令方向和真实过阶事件，因此不能据此算出正面上阶的有效样本率。

台阶份额较公开示例低是可检查因素。若后续单项实验效果不足，可另提保留六类的 30% 台阶方案，例如平地 15%、粗糙 15%、坡 20%、台阶 30%（上/下各 15%）、摩擦 10%、粗糙坡 10%，在 20 列上可精确分配。此方案尚未实施，不与净空实验混合，也不承诺增大份额就提高成功率。

## 提高净空裕量的离线敏感性

候选取台阶顶面以上 40 mm，原值 25 mm。40 mm 是为本地受控试验预先选定的候选，不是论文给出的 Lite3 最优值。以相邻高 5 cm 的台阶为例，足底目标相对下一级阶面由 7.5 cm 变成 9 cm；不能将它称为所有脚始终抬高 9 cm，实际奖励仅在持续净空任务有效且未形成合法支撑时计入。

复用先前带航向反馈的名义/随机化各 240 条轨迹，只重算代价函数；门控、目标建立/清除、动作、轨迹和结果都固定。原函数重算与记录逐元素最大误差为 0。第一回合共 254,137 个有效环境帧，报告将全体和失败分开统计。

| 条件 | 原净空累计回报 | 候选累计回报 | 新增被罚足帧 |
|---|---:|---:|---:|
| 名义全部 240 回合 | −157.990 | −216.250 | 6,885 |
| 随机化全部 240 回合 | −151.589 | −211.826 | 5,976 |
| 名义失败 196，5 cm / 0.32 m/s | −2.719 | −3.346 | 40 |
| 名义失败 199，5 cm / 0.32 m/s | −2.708 | −3.376 | 68 |
| 随机化失败 121，3 cm / 0.32 m/s | −1.532 | −2.092 | 20 |
| 随机化失败 231，5 cm / 0.6 m/s | −0.624 | −0.855 | 49 |

表中失败行是全体的子集，不再相加。新增被罚足帧合计 12,861，其中原成功轨迹占 12,684，四个失败占 177。已有正代价的帧还可能增加惩罚幅度。因此这会改变成功步态的学习压力，并非只纠正失败状态；失败与成功的存活时长不同，也不能直接用总回报比较质量。

此计算没有改变实际抬脚高度或通过率，也没有评估完整训练后的策略梯度和动作。提高目标还可能改变摆动时间、能耗和支撑时序，需以训练后的行为验证。当前有界代价的梯度并非在任何高度都随目标提升而增大，不能将本表说成更强奖励必然使脚抬得更高。

## 下一阶段建议和交付

建议优先做净空裕量 25→40 mm 的单项对照：从同一 zero10100 分出 A 原设置与 B 候选，各最多 100 次 PPO 更新；六类分布、2–5 cm 台阶、落脚 15 mm 门槛、其余奖励和网络维持相同。旧的落脚资格 15→5 mm 提案暂缓，恢复门控与历史观测改造分别留待后续判断。

主要指标应包含实际过边缘时的足底高度、碰阶代理、后脚/机身推进、稳定落脚、完整通过率及上/下阶和中高速运动是否退化。单纯奖励变高或摆脚峰值变高不算解决。详细预算、时间预期、停止条件及验收在 [NEXT_STAGE_PROPOSAL.md](NEXT_STAGE_PROPOSAL.md)。下一阶段需用户同意后执行。

研究与计算的可复现入口为 [reference_multiterrain_research_audit.py](../../scripts/tools/reference_multiterrain_research_audit.py)。[local_audit.json](local_audit.json) 包含 26 份源码校验、模型/输入文件哈希、逐地形覆盖量、单项代价重算和四个失败明细。模型 SHA256 保持 `290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497`。没有新训练结果，也没有部署变更。

[lg-config]: https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot_config.py
[lg-terrain]: https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/utils/terrain.py
[lg-robot]: https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot.py
[him-config]: https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot_config.py
[him-terrain]: https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/utils/terrain.py
[him-go1]: https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/go1/go1_config.py
[him-robot]: https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot.py
[him-actor]: https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/rsl_rl/rsl_rl/modules/him_actor_critic.py
[him-estimator]: https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/rsl_rl/rsl_rl/modules/him_estimator.py
[il-terrain]: https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/terrains/config/rough.py
[il-generator]: https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/terrains/terrain_generator.py
[il-go1]: https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py
[il-env]: https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
[il-curriculum]: https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py
[il-ppo]: https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/agents/rsl_rl_ppo_cfg.py
[wtw-go1]: https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/go1/go1_config.py
[wtw-train]: https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/scripts/train.py
[wtw-rewards]: https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/rewards/corl_rewards.py
[wtw-terrain]: https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/utils/terrain.py
