# Lite3：FlowDAgger / USR 与跨领域在线适应方法复核

日期：2026-09-28。任务：重新阅读用户提供的两篇论文，联网扩展四足、双足/人形、VLA 和机械臂论文及项目，给当前方案提供可追溯的取舍依据。

本文是**研究建议及文献复核，不是已实现、已验证的新算法**。执行计划仍为 [v2.1](lite3_online_adaptation_plan_v2_2026-09-24.md)，执行时同时遵守 [9/28 审核中的证据修正](lite3-plan-audit-2026-09-28.md)。不能因新论文名称相似而重启此前失败的教师路线或跳过 G0。

## 1. 结论和阅读优先级

现有冻结 Gaussian PPO 不具备 FlowDAgger / USR 所依赖的多模态生成先验。值得迁移的是**纠正经验的聚合、结构化的适配接口、先验内引导与额外残差的分工、在线价值学习的稳定化**，不是把普通残差重新命名为 noise steering。

建议按以下顺序推进，SAC 与 LoRA 都保留：

1. 先完成 G0 配对基线和动作/终止语义核对。目前基线仍存在 runner 差异。
2. M0 做可靠对照：LoRA-PPO 参考 SLowRL / VLA-RL 的 actor–critic 协同适配；Residual SAC 参考 WSRL / ResFiT 的目标域数据、价值学习和更新调度。
3. M2 先检验是否存在可复现的低维、时间连续行为修正；比较固定行为基、随机基和时间相关探索，不能直接宣称已有“行为 latent”。
4. M1 只在成功修正能冻结复现后做自模仿；模仿后的闭环表现才是验收条件。
5. M4 只在 M2 确实有效后研究额外残差。M3 快照分支是单独的仿真特权实验臂。

最优先细读：**FlowDAgger、USR、SLowRL、Walk These Ways、WSRL、ResFiT、ZPRL、SteerGenPO**。新增 PARTS 和 OSRAM 可帮助判断“应该在动作、行为还是指令层适配”。暂不为了套用生成策略论文而重训大型 diffusion / VLA 基础模型。

## 2. 本轮检索与证据等级

- 重新提取两份用户 PDF，阅读方法、算法及相关实验/附录段落。不是仅阅读摘要。
- 在线核对 arXiv 标题、摘要、日期；对 WSRL、ResFiT、SLowRL、VLA-RL、DSRL、SteerGenPO、PARTS、OSRAM、ZPRL 阅读 HTML 的相关方法段落。其他条目若标“摘要”，不代表完成全文审稿。
- 从论文/作者项目页追踪代码链接，读取 GitHub 仓库信息和 README；FlowDAgger 另核对源码树。**未安装或运行这些外部项目，仓库存在不等于论文已可完整复现。**
- 扩展查询包括 arXiv 的 `robot online adaptation residual`，GitHub 的 `quadruped adaptation`、`VLA residual reinforcement learning` 和论文名称检索。通用搜索中 Google 仅返回跳转页，DuckDuckGo 返回验证页，Bing 结果相关度较低；未将这些页面当作论文证据。
- USR 的 OpenReview 正常页面跳到验证页面、API 返回 403。本轮会议信息以用户 PDF 首页为依据；不依据这次受阻请求确认其旧投稿结果。未检索到可确认的官方代码，**这不是“代码一定不存在”**。
- 查询发现的新论文均另外核对首发日期；OSRAM 为 9/24，VLaRL 为 9/25，均早于本文日期。它们仍是新近预印本，尚无本项目复现实验。
- 访问 URL、HTTP 状态、页面 SHA256、标题/日期和仓库 README 标识保存在 [检索证据清单](lite3-literature-evidence-2026-09-28.json)。HTTP 200 的挑战页也明确记为未核验。

旧调研仍可作为线索，但下列论断不能沿用：EV<0 就证明 critic 从未学会；λ=1 的截断 rollout 就是完整 MC；残差 R² 高就证明任务有收益；“组合这些组件尚无人做”就构成创新。详见 9/28 审核。

## 3. 两篇指定论文到底能借鉴什么

### 3.1 FlowDAgger

来源：[论文](https://arxiv.org/abs/2607.08877)、[作者项目](https://microsoft.github.io/FlowDAgger/)、[官方代码](https://github.com/microsoft/FlowDAgger)。用户 PDF 首页为 2607.08877v1，2026-07-09。官方 README 自述 CoRL 2026；本文不把 README 自述升级为独立核验的会议录事实。

设冻结生成器为 G₀(o,w)。人或脚本专家在当前策略访问的状态给出纠正动作块 a*，论文反演出 w* 使 G₀(o,w*)≈a*，再监督训练小网络 w=fθ(o)。对 flow 的每个 Euler 步，采用局部 fixed-point 迭代反演；这依赖局部收缩条件，不能表述为任意网络上的无条件精确逆。

论文 §4.3 的训练 batch 等比例抽取：

- 专家干预的反演噪声目标；
- 成功自主 rollout 中实际生成动作的噪声。

这不是“一半纠正、一半平地零残差”，也不是“无需纠正源即可训练”。官方示例用 scripted expert，不等于算法不需要额外专家信息。

**成本与结构边界：**附录 Table 5 中，π0.5 + MetaWorld 默认反演在 RTX 5090 上的中位耗时为 **456 ms/动作块**；这不是 Lite3 的 20 ms 控制周期内可直接完成的实时步骤。WAM 的噪声维度很高，附录采用 PCA 64 维预测形式；不能把所有 noise steering 统称为低维。冻结 G₀ 只保证权重不变，不保证适配后闭环能力不退化。

对当前 Gaussian PPO，若 a=μ₀(o)+σ⊙ε，且没有 clipping / tanh / 安全投影，则 ε*=(a*−μ₀(o))/σ。它本质上是缩放后的动作残差，未增加行为先验。若用执行动作反演，安全投影一般不可逆，上式只能定义一个回归目标，不能恢复当时的请求噪声。

**可取：**在实际访问状态聚合纠正数据、保留成功经验、把探索发现的行为蒸馏成确定性闭环控制。**不应做：**把安全模块输出当成任务专家，把普通 Gaussian 残差回归命名为“FlowDAgger 复现”。

### 3.2 Unified Latent Steering and Residual Refinement（USR）

来源：[OpenReview 地址](https://openreview.net/forum?id=9sQVoVMeSD)及用户本地 PDF。用户文件名写“2025”，但 PDF 首页标注 **ICML 2026 DEMO Workshop**，不是 ICML 主会。

其动作组合为：

```text
(w, r) ~ πθ(· | o)
a = G₀(o, w) + α r
```

其中 w 选择先验中的行为，r 修正先验难以覆盖的误差。论文 §4.2 有**不同输入空间**的两个 critic：

- Q_A(o,a)：在最终动作空间用环境转移学习；
- Q_C(o,[w,r])：蒸馏 Q_A(o,G₀(o,w)+αr)，给轻量 actor 提供梯度。

这不是 SAC 的普通 twin-Q。§4.1 的 progressive exploration 是残差启用概率 ε 从 0 增长到 1 的 Bernoulli 机制；它不是已经验证过的风险连续门控。选择层与修正层也不是必须完全分阶段训练。

§5.5 在 **Agibot G1 双臂平台的开盖任务**报告 base 8/20、DSRL 15/20、USR 18/20；在线训练 100 回合，由人提供终局奖励和必要重置。不能将其说成 Unitree G1 双足行走实验，也不能据此承诺 Lite3 20 回合达到 90%。

**可取：**先确认已有行为覆盖范围，再用额外残差补足表达能力。**限制：**对于当前小 MLP / 线性行为解码器，绕开大型生成器反传的蒸馏 critic 未必值得；动作残差和 latent 修正也可能重复参数化同一自由度。

## 4. 跨领域论文：机制、额外条件和迁移判断

### 4.1 四足和低维适应

| 工作与来源 | 本轮证据 | 可借鉴机制 | 额外条件 / 对 Lite3 的限制 |
|---|---|---|---|
| [SLowRL, 2603.17092](https://arxiv.org/abs/2603.17092) | 方法/实验段落 | actor 与 critic 的低秩适配；recovery policy 保护训练；rank 消融 | Go2 跳跃/小跑 sim-to-real，已有预训练能力和 recovery；不能从 rank-1 结果推出 Lite3 固定最优 rank 或直接把学习率增大 10 倍 |
| [Walk These Ways, 2212.03238](https://arxiv.org/abs/2212.03238) | 摘要、项目、README | 把步态、抬脚、姿态等组织成可调行为族 | 原策略经过行为条件化训练；现有 Lite3 PPO 没有这些输入，不能直接把参数塞进网络，也不能宣称自动调参本身是空白 |
| [PMTG, 1910.02812](https://arxiv.org/abs/1910.02812) | 摘要 | 可调轨迹发生器提供相位和周期先验，策略进行调制 | 需要定义有用的步态结构；低维不保证能越过当前障碍 |
| [RMA, 2107.04034](https://arxiv.org/abs/2107.04034) | 摘要、项目、README | 从本体历史推断环境/动力学 latent | 大量仿真预训练；部署适应主要是推断，不等于在目标地形在线更新权重；当前冻结 PPO 没有现成 latent 接口 |
| [A Walk in the Park, 2208.07860](https://arxiv.org/abs/2208.07860) | 摘要、README | 四足实机上的高样本效率离策略学习；公开 UTD 配置 | A1 行走与约束动作设计；“20 分钟学走路”不等于 Lite3 20 回合复杂地形适应，UTD=20 不是默认答案 |
| [TWiRL, 2304.09834](https://arxiv.org/abs/2304.09834) | 摘要 | 迁移既有控制器产生的经验来引导新技能探索 | 需要有用源经验；直接搬平地数据未必覆盖障碍接触 |
| [Fast Adaptation with BFMs, 2504.07896](https://arxiv.org/abs/2504.07896) | 摘要 | 在预训练任务 embedding 上做 actor-only / actor–critic 快速搜索 | 前提是 BFM 已包含有用技能；当前 PPO 不是 BFM，不能假定只缺搜索器 |
| [Robots that can adapt like animals / ITE, 1407.3501](https://arxiv.org/abs/1407.3501) | 摘要 | 离线行为库＋在线试错，六足和机械臂均有实验 | 行为库本身有离线成本；小样本结果不能算成“无预训练成本” |

### 4.2 双足、人形与物理角色控制

| 工作与来源 | 本轮证据 | 可借鉴机制 | 额外条件 / 对 Lite3 的限制 |
|---|---|---|---|
| [Robot Trains Robot, 2508.12252](https://arxiv.org/abs/2508.12252) | 摘要、项目、README | 优化动力学 latent，机械臂支持人形训练并自动重置 | 机械臂教师提供保护、奖励、扰动和重置；不能视为无额外支持的在线 RL |
| [ASAP, 2502.01143](https://arxiv.org/abs/2502.01143) | 摘要、项目链接、README | 收集真实轨迹，学 delta action 对齐仿真，再微调技能 | 残差用于补偿动力学差距，主问题不同于未知地形越障；需要真实数据、动作跟踪技能及额外训练 |
| [SteerGenPO, 2606.16572](https://arxiv.org/abs/2606.16572) | 方法/实验段落 | 先训练生成式 RL 策略，再学习状态条件 latent actor，实现确定性部署 | Isaac Lab 与 Unitree G1 行走证据；必须有生成式 base，是本项目若改做 generative steering 的直接近邻 |
| [NaP-Control, 2605.20209](https://arxiv.org/abs/2605.20209) | 摘要 | 对扩散运动先验的噪声做 RL steering | 物理角色动画领域；本文未据此确认真实双足部署，不得与实机 locomotion 混称 |
| [ADAPT：Agile Diffusion Action Priors, 2609.00677](https://arxiv.org/abs/2609.00677) | 摘要、项目 | 文本条件人形动作先验＋冻结先验上的残差 RL | 需要文本标注的人形轨迹；online text-driven control 不自动等于部署时更新权重 |
| [ADAPT：Analytical Disturbance-Aware Policy Training, 2606.16542](https://arxiv.org/abs/2606.16542) | 摘要 | 解析扰动观测器给 G1 策略提供外力估计 | 与上一条同缩写但完全不同；依赖动力学模型和重新训练的观测接口，可借鉴诊断，不能直接接到原 PPO 输入 |
| [OSRAM, 2609.28878](https://arxiv.org/abs/2609.28878) | 方法段落 | 学“控制器＋机器人”的闭环指令响应，在线改参考指令而非底层权重 | 双足跟踪/loco-manipulation；闭环模型需随机动力学元训练。值得做低成本指令层对照，但指令调节未必解决抬脚不足 |

### 4.3 VLA、机械臂与生成策略后训练

| 工作与来源 | 本轮证据 | 可借鉴机制 | 额外条件 / 对 Lite3 的限制 |
|---|---|---|---|
| [DSRL, 2506.15799](https://arxiv.org/abs/2506.15799) | 方法段落、项目、仓库元信息 | 冻结 diffusion，在噪声空间用 RL 适应；价值蒸馏 | 需要生成先验；不能等同于 Gaussian PPO 噪声偏移 |
| [Policy Decorator, 2412.13630](https://arxiv.org/abs/2412.13630) | 摘要、README、USR 对其机制描述 | 冻结大策略上的受限残差和渐进探索 | 已覆盖“冻结 base＋逐步放开 residual”，不能把这项组合当新贡献 |
| [ResFiT, 2509.19301](https://arxiv.org/abs/2509.19301) | 方法/消融、README | n-step、延迟 actor、Q ensemble、LayerNorm、适度 UTD、保留演示 | 操作及人形灵巧手任务，不是双足行走；公开入口为 residual TD3。n=3、UTD≈4 仅是候选；SAC 的熵、target 和 off-policy n-step 偏差需单独处理 |
| [WSRL, 2412.07762](https://arxiv.org/abs/2412.07762) | 方法/消融、README | 用预训练策略在目标任务采初始 replay，缓解价值分布失配 | 原方法从离线 RL 的 actor/Q 初始化；Lite3 随机 residual Q 只能借鉴，不能称等价复现。warmup 首先是数据采集，不等同于所有步骤都“只训练 critic” |
| [VLA-RL, 2505.18719](https://arxiv.org/abs/2505.18719) | 方法/消融、README | 更新 actor 前先用初始策略轨迹训练 critic | 大模型、过程奖励、多任务系统；借鉴预热机制，不能照搬 token PPO 或 LIBERO 效果 |
| [RIPT-VLA, 2505.17016](https://arxiv.org/abs/2505.17016) | 摘要 | 动态采样、leave-one-out 优势，减少价值网络依赖 | 组内 rollout 也消耗预算；全成功/全失败无比较信号；项目链接本轮返回 404 |
| [HIL-SERL, 2410.21845](https://arxiv.org/abs/2410.21845) | 摘要、README | 演示＋人工纠正＋高效离策略学习 | 明确需要人类信息和硬件系统；Lite3 限幅器不具备同等信息 |
| [UniSteer, 2605.10821](https://arxiv.org/abs/2605.10821) | 摘要 | 将人工动作反演成 latent 标签，同时监督和 RL 更新 noise actor | “FlowDAgger 式监督＋latent RL”已有近邻；需要可反演生成器及纠正源 |
| [ZPRL, 2605.19919](https://arxiv.org/abs/2605.19919) | 方法段落、README | 离线训练 VIB 瓶颈，在线只扰动紧凑 latent，获得结构化探索 | 不是从任意旧策略免费提取瓶颈；需要离线插入/训练接口，和本项目冻结原权重约束有成本冲突 |
| [PARTS, 2609.21788](https://arxiv.org/abs/2609.21788) | 方法段落、项目 | 在失败瓶颈子任务激活残差，以局部成功学习，但最终评测完整任务 | 人指定瓶颈，coding agents/VLM 构建 selector/verifier，另需重置；不是自动发现所有失败原因，也不能用局部奖励替代完整越障验收 |
| [ORPA, 2608.17323](https://arxiv.org/abs/2608.17323) | 摘要 | ALOHA / ACT 上通过反馈做关节残差修正 | 人类反馈条件的修正不等于自主发现改进方向 |
| [BORA, 2605.30226](https://arxiv.org/abs/2605.30226) | 摘要 | 离线动作块 critic＋在线人工干预残差 | 额外离线价值学习和人类干预；动作块在接触步态上应更短、可中断 |
| [VLaRL, 2609.30868](https://arxiv.org/abs/2609.30868) | 摘要 | 在仿真训练 frozen VLA 的 latent 条件残差，并对齐跨域 latent | **不同于 2025 VLA-RL**；本工作明确不依赖实机在线 RL，不能用作有限目标域在线学习的直接证据 |

### 4.4 与“执行感知探索”最接近的已有研究

- [Clipped Action Policy Gradient, 1802.07564](https://arxiv.org/abs/1802.07564)：利用 clipping 结构降低策略梯度估计方差。本文仅核验摘要；它针对动作界限，不可直接推广成状态化限速器的推导。
- [Safe RL using Action Projection: Safeguard the Policy or the Environment?, 2509.12833](https://arxiv.org/abs/2509.12833)：明确讨论 action aliasing、把投影放进策略或环境的差别。本文仅核验摘要；“认识到安全投影造成别名”本身已非新发现。

因此，若以执行层约束为创新方向，必须与这些文献以及简单时间相关噪声、参数空间探索做明确区别和实验比较。

## 5. 开源项目可用性：哪些能真正拿来参考

以下均有论文/作者页面指向和实际网络读取记录。除特别注明外，本轮读到 README，未运行代码。

| 项目 | 已确认提供的内容 | 对接成本 / 核验限制 |
|---|---|---|
| [microsoft/FlowDAgger](https://github.com/microsoft/FlowDAgger) | pi0.5 + MetaWorld assembly / GR00T N1.7 + LIBERO-90 的最小示例；树中有 inverter、buffer、DAgger loop、scripted experts | JAX 与 PyTorch 分环境；MIT；不是用户论文所有 WAM/实机实验的完整包 |
| [Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) | Go1 行为条件化训练、评测、部署说明及预训练路径 | Isaac Gym Preview 4；机器人和当前 Isaac Lab 不同；许可应查看仓库分组件文件，API 为 NOASSERTION |
| [ikostrikov/walk_in_the_park](https://github.com/ikostrikov/walk_in_the_park) | A1 仿真/实机训练入口、DroQ/UTD 示例 | JAX/MuJoCo/Unitree SDK；MIT；参考学习器而非替换整个环境 |
| [zhouzypaul/wsrl](https://github.com/zhouzypaul/wsrl) | JAX/Flax 离线到在线算法、warmup 和 benchmark 运行说明 | README 有 MIT badge，API 未识别许可；复制前核对具体文件。源模型为离线 RL 初始化 |
| [amazon-far/residual-offpolicy-rl](https://github.com/amazon-far/residual-offpolicy-rl) | BC 训练及 `train_residual_td3.py`、任务配置 | Python 3.10、Robosuite/LeRobot 数据和模型依赖；API 许可 NOASSERTION；最适合核对 residual 学习配方 |
| [ajwagen/dsrl](https://github.com/ajwagen/dsrl) | 作者项目指向的 DSRL 仓库，元信息可读取 | README 两次读取 SSL 失败，故不声称核对运行入口；π0 另有作者链接的 `nakamotoo/dsrl_pi0` |
| [tongzhoumu/policy_decorator](https://github.com/tongzhoumu/policy_decorator) | ManiSkill/Adroit 示例，预训练 base 下载说明 | 适合借鉴残差限幅/启用调度；API 许可 NOASSERTION |
| [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl) | SAC、演示/干预及实机采集学习系统 | JAX 和操作硬件；Apache-2.0；需明确人类数据成本 |
| [GuanxingLu/vlarl](https://github.com/GuanxingLu/vlarl) | Ray/vLLM/LoRA/FSDP 训练和评测入口 | README 自述仍在开发、实机 RL 未完成；README 标 Apache-2.0；Lite3 无需引入此大型依赖栈 |
| [hukz18/Robot-Trains-Robot](https://github.com/hukz18/Robot-Trains-Robot) | 人形/机械臂训练系统安装与训练说明 | MIT；硬件形态不同；README clone 路径与公开仓库地址不一致，需对接前核查 |
| [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) | motion tracking、delta action、数据、sim2sim/real 流程说明 | MIT；IsaacGym/IsaacSim/Genesis 多栈；动作数据和重训成本较高 |
| [ManUtdMoon/ZPRL](https://github.com/ManUtdMoon/ZPRL) | 瓶颈 latent 方法说明、依赖/数据/训练配置 | 依赖特定 Robosuite/Robomimic 版本和离线数据；本轮未核定独立许可文件 |
| [antonilo/rl_locomotion](https://github.com/antonilo/rl_locomotion) | RMA 项目链接的 Cross-Modal Supervision / RMA 训练实现 | Raisim、GPL-3.0；应称基于 RMA 的 CMS 实现，不应笼统称原论文一键完整复现 |

SLowRL、SteerGenPO、USR：本轮未找到可确认的官方实现，不等于不存在。PARTS 项目页展示系统及部分上游实现链接，未确认完整官方训练包。OSRAM 请求成功但页面是通用实验室模板，未将其视为代码发布证据。ADAPT diffusion 项目页可读，未确认完整训练代码。

## 6. 推荐给 Lite3 的最小算法候选

### 6.1 研究问题

建议聚焦：**在冻结、非生成式的平地控制器外，如何用有限交互获得能穿过实际安全/限速环节、并在关闭探索后仍有效的地形修正？**

它比“在四足上套 LoRA/USR”具体，但仍是待检验问题。这里“穿过安全环节”指请求修正能产生可控、可重复的执行变化，不是绕过或削弱安全约束。

### 6.2 适配接口

先测试如下最小形式，不立即加入所有模块：

```text
h_t：本体历史、前一执行动作及限速/门控所需内部状态
z_k：在一个短窗口内保持或缓慢变化的低维修正参数
δ_t = B(h_t) z_k
request_t = μ₀(o_t) + δ_t
executed_t = P_safe(request_t; h_t)
```

B 首先是明确的固定行为基或从开发数据构造的固定基。若采用抬腿/相位相关基，必须验证触地/相位估计；仅按回合时钟造正弦波不等于正确步态。PCA 的方差方向也不自动等于任务有效方向。没有良好 B，就不能称之为已学到的可控行为先验。

先做少量固定候选的 headroom 检验，比较独立初始扰动下的完成、相对前进距离和失败类型。候选筛选和复测的步数都计费。若只是某次噪声帮忙越过脆弱停滞点，冻结重测无收益，则不扩大算法。

### 6.3 对 SAC 和 LoRA 的具体建议

**Residual SAC（继续保留）**：

- 用当前目标地形上的冻结 base 收集 replay，记录探索/更新版本。平地 anchor 是能力保持数据，不替代目标域预热。
- 先核对 requested action、执行动作、终止、SAC soft target 和熵的坐标语义。若滤波器带历史，critic 输入应具有足够历史；不能凭“把投影视为环境”就假定有限观测已满足 Markov 条件。
- 把 LayerNorm、延迟 actor、UTD、n-step 作为逐项消融，优先小规模配置。不能同时打开十个改动再归因。ResFiT 是 TD3 风格，移植到 SAC 时保留 SAC 自洽的熵项和 target；off-policy n-step 需说明偏差处理。
- 结构化接口有 headroom 后，可比较动作残差 SAC 与低维 z-SAC。若 z 跨步保持，需用一致的宏步转移/折扣或状态增强定义，不能把每个内部步当作重新独立采样的 z。

**LoRA-PPO（继续保留）**：

- 先固定策略，使用正确终止/截断语义的数据诊断价值拟合。独立完整回合的回报预测与训练内 bootstrap EV 分开；不设未经验证的 EV>0.3 硬门槛。
- 检查 base critic 的观测特征和奖励含义；平地 critic 不可原样当作地形任务真值。
- actor/critic 的适配范围、rank、学习率各自消融。SLowRL 提供动机，不提供 Lite3 最优超参数证明。
- 若试验时间相关噪声，必须实现其条件采样概率和缓存历史，不能沿用独立 Gaussian 的 log-prob/ratio。也可先用独立的回合参数搜索避免这一实现混淆。

### 6.4 如何借鉴 FlowDAgger 的学习闭环

1. 从有效修正的成功轨迹聚合数据，严格按回合划分训练与复测。
2. 保留 requested / executed action、base action 和安全状态。对已知 z 保存原始 z；不要把安全执行动作直接当成请求动作重复过滤。
3. 对不具备可靠 inverse 的安全映射，可用执行一致性目标验证适配器：

   `L_exec = ||P_safe(μ₀(o)+B(h)fθ(h); h) − a_exec*||²`。

   这是候选监督目标；需要正确复原过滤状态，非光滑投影还需处理优化问题。它也不证明 a_exec* 在新状态下仍最优。
4. 用完整闭环的无探索复测判断自模仿是否学到了任务收益。R²/MSE 仅是诊断，不能代替成功率。
5. 数据只来自自主成功轨迹时称“成功轨迹自模仿/蒸馏”；有人或可信专家在访问状态提供纠正时，才具有 DAgger 式标签条件。快照分支另列特权成本。

### 6.5 如何借鉴 USR 而不制造冗余

只有结构化 δ=Bz 无法覆盖剩余误差时，才加入 `+g(h)rθ(h)`。需比较：仅 Bz、仅动作残差、二者组合、相同参数量/动作幅度的较大 B。否则收益可能只是容量和探索预算增加。

g 的依据可以是可观测失败/不确定性，但风险评分本身要验证；先不宣称形式安全保证。现有小解码器不必增加 USR 的蒸馏 critic；若要增加，必须证明解决了实际反传或别名问题。

### 6.6 本轮新论文带来的两项备选

- **PARTS 启发：**只在停滞、打滑或非足接触等可识别瓶颈段集中学习，同时检查完整赛道结果。触发和局部奖励若用仿真真值必须注明特权；不能按已看过的固定世界坐标写死触发点并声称泛化。
- **OSRAM 启发：**若 base 已接收速度/转向指令，可先比较受限指令调节；它可能比关节残差更稳定。必须固定最终任务要求，不能降低目标速度、延长 deadline 或绕过障碍来获得虚假改善。没有对应输入的抬脚/机身高度命令不能凭空增加。

## 7. 创新性与可证伪实验

以下**不能单独当算法创新**：冻结 base、小网络、LoRA、latent＋residual、渐进残差启用、成功轨迹重放、critic warmup、低维搜索、安全投影。已有近邻已覆盖多数单项及若干组合；跨领域迁移不等于自动获得创新性。

可以优先检验的一项研究假设是：

> 在状态化安全投影和接触动力学下，适配应围绕实际可执行、具有时间结构的行为方向进行，而非仅缩小参数量或调节独立动作噪声。

可能的贡献必须落在**新机制及证据**：如何估计有效方向、如何保持跨时刻的一致性、如何把发现的修正转成确定性控制，且比简单 OU/低通噪声、随机低维基、固定较大基和普通 residual RL 更有效。目前本文没有证明这一点，也没有排除所有相关先行研究。

最小比较顺序：

| 问题 | 必要比较 | 能支持 / 否定什么 |
|---|---|---|
| 是否真有学习收益 | 同 runner 的 frozen base；修复后的 LoRA-PPO；修复后的 Residual SAC | 排除执行配置差异和弱基线 |
| 是否只是噪声平滑 | 独立噪声 vs 时间相关噪声，匹配执行动作量级和交互成本 | 平滑就能解决时，不夸大复杂模块贡献 |
| 低维基是否有用 | 行为基 vs 随机基 vs 固定更高维基，独立扰动复测 | 区分维数、结构和候选选择运气 |
| 蒸馏是否有用 | 搜索最佳固定 z vs 学到 fθ(h)，都关闭探索 | 排除成功样本只来自采样运气或多种走法被平均 |
| 额外 residual 是否必要 | steering only / residual only / combined，同等成本 | 判断 USR 式分工是否真的成立 |
| 瓶颈集中学习是否有用 | 全程适配 vs 状态触发适配，仍测完整赛道 | 区分奖励/预算变化和触发机制效果 |

统一记录：目标地形全部交互步数（预热、搜索、候选选优探针、分支均计入）、更新计算、外部演示/预训练成本、失败和安全事件；独立最终评估成本另列。报告学习曲线和多次独立适应会话的冻结结果，不能把同一会话多个回合视为独立训练种子。失败池选取与 sealed test 严格分开；泛化和平地保持仍是次要终点。

## 8. 当前工程事实与下一步

本轮读取的 `artifacts/g0_base_pair_20260928/manifest.json` 已完成 5 个冻结 LoRA runner 基线：

| terrain seed | 本轮完成 | max_x（世界坐标，m） | 与历史 SAC runner 完成判定 |
|---|---|---|---|
| 101 | 是 | 4.5024 | 不同：历史失败 |
| 103 | 是 | 4.5043 | 相同 |
| 107 | 否 | 1.6078 | 相同 |
| 109 | 否 | 2.2832 | 不同：历史成功 |
| 113 | 是 | 4.5006 | 相同 |

当前是 **3/5**，但与历史相同总数掩盖了逐种子结果的变化。状态是 `complete_with_baseline_mismatch`，不代表算法提升。原“101 稳定失败、停在 0.65 m”的前提不能直接用于新实验。上表 max_x 也不能直接拿来计算相对进展比例。

下一步优先级仍是：核对两个 runner 的 base 输入、历史/上一动作、reset、动作缩放和安全控制；补齐 SAC 逐步记录与公平协议。然后做小规模 M0/M2 验证，再决定是否进入 M1/M4。不要现在凭文献直接启动长训练、重训 diffusion base 或把任何一条候选确定为论文主算法。

本轮工作仅更新文献与设计依据，没有变更学习器代码，也没有启动新训练。
