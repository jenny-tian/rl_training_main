# Lite3 在线适应：跨领域可迁移方法调研

> 2026-09-28 更新：请优先结合[重新联网核验的文献与算法评估](lite3-literature-and-algorithm-review-2026-09-28.md)及[计划审核](lite3-plan-audit-2026-09-28.md)阅读。本文保留历史调研；旧代码发布状态、价值学习推断和新颖性判断不应未经复核直接沿用。

日期：2026-09-25。本文为计划 v2.1（`lite3_online_adaptation_plan_v2_2026-09-24.md`）中 D3、D8 和 G1 方法轨道提供依据。涉及的领域有四足、双足/人形、机械臂和 VLA，入选标准是"功能相似，可迁移"。本文与 `lite3-generative-policy-adaptation-review-2026-09-15.md` 互补：那份覆盖 USR、FlowDAgger、DSRL 等生成策略方法；本文补上新近工作，并按我们的实际瓶颈重新归类。

## 0. 核验说明

- 本轮检索时 WebSearch/WebFetch 不可用。条目通过 arXiv API、OpenReview API、Semantic Scholar 引用和 PDF 全文（pdftotext）核验。"全文"指读过方法和实验段落，"摘要"指只核对了摘要。
- 另外对其中 12 个 arXiv ID 的标题和首发日期用 arXiv API 做了复核，全部一致：2607.08877、2606.16572、2607.21867、2603.17092、2503.01491、2505.18719、2605.10821、2504.07896、2403.03950、2412.07762、2511.14759、2509.19301。
- USR 没有 arXiv 版本，也没有公开代码。它在 OpenReview 上是 ICML 2026 DEMO workshop poster（`9sQVoVMeSD`），此前投 ICLR 2026（`DbBD2aT1OG`）被拒。
- 文中的超参数均摘自原文，没有在 Lite3 上复现过。

## 1. 与本项目相关的已核实事实

以下事实决定了哪些方法能迁移过来。它们都出自仓库代码或 `artifacts/semester_2026/09/` 下的日志。

| 事实 | 出处 | 含义 |
|---|---|---|
| base 是单峰 Gaussian MLP（45→512→256→128→12） | base checkpoint | FlowDAgger 的"反演到噪声空间"退化为 ε\* = (a − μ(o))/σ，即残差回归；DSRL/USR 所谓"留在先验流形上"的优势不会自动出现，必须先构造出带行为结构的低维 latent |
| 训练期安全投影比例在 33%–99.8% 之间，随探索 std 从 0.03 降到 0.01 单调下降（三组 09-22/23 消融结果一致） | `lora_ppo_multiepisode/ablation_*/episodes/*/summary.json`，字段 `safety_projected_fraction` | 残差变化率上限 0.03 小于独立噪声相邻两步差的尺度（σ√2），所以实际执行的探索是一条被截断的随机游走，不是白噪声。需要注意：①训练期成功和执行噪声的这种形态有关；②自模仿时，监督目标必须用执行动作；③时间相关噪声或参数空间噪声能更完整地通过限速器 |
| PPO 配置：γ=0.99，GAE λ=0.95，每次更新 256 步，完成奖励 +305，critic 观测 48 维（含已用时间） | episode `config.json` | 0.99^200≈0.13，完成奖励只影响最后约 200–300 步的价值；256 步的 batch 里只偶尔含有终点，这与 critic EV<0 的现象吻合 |
| base checkpoint 带有 critic（48→512→256→128→1） | `model_1998.pt` 中的 `critic_state_dict` | 可以作为在线 critic 的初始化候选（SLowRL 的做法）。但它的奖励是平地速度跟踪，与在线奖励不一致，需要另加 ΔV 头或只复用特征层 |
| 残差 SAC：双 Q，n_step=1，没有 LayerNorm 和 ensemble；critic 预热用的是平地 replay，不是目标地形上的 base rollout | `online_adaptation/agent.py`、`sac.py` | 和 WSRL/ResFiT 的配置不同：WSRL 明确说预热数据必须来自目标任务上的 base |
| 没有保存逐步轨迹，每个 episode 目录只有 config、summary、updates.jsonl、latest.pt 和日志 | `episodes/episode-*/` | 自模仿预检、RECAP 离线训练、回报方差分解都没法直接用旧日志，必须先加逐步记录再重跑 |
| 重置是确定性的（`deterministic_root_reset_params`） | 代码 | 利于公共随机数比较和确定性候选评估；风险是模仿时会背下一条路线 |

## 2. 文献按功能分组

相关度满分 5，指对 Lite3 当前瓶颈的直接程度。

### 2.1 价值函数拟合失败的修复（对应 critic EV<0）

| 工作 | 领域 | 机制 | 迁移做法 | 核验 | 相关 |
|---|---|---|---|---|---|
| VC-PPO / VAPO（2503.01491 / 2504.05118） | LLM PPO | 冻结策略，用 λ=1（MC）目标预训练价值；Decoupled-GAE：λ_critic=1.0、λ_actor=0.95。VAPO 去掉 value pretrain 后 AIME 从 60 降到 11 | 前几个回合只训练 critic，留出回合 EV>0.3 后才更新 actor。预训练过久会过拟合 | 全文 | 5 |
| VLA-RL（2505.18719） | VLA PPO | 先用模仿策略采数据预热 critic；去掉预热后 LIBERO-Spatial 成功率从 90.2 降到 80.0 | 同上，是机器人领域的直接证据 | 全文 | 5 |
| π_RL（2510.25889） | flow VLA PPO | 评测初期的下跌与 critic 预热期同步；LoRA 需要更保守的更新次数 | 把 EV 当作解冻 actor 的门槛 | 全文 | 3 |
| WSRL（2412.07762） | offline→online | 用预训练策略在目标任务上采 5k 步重新校准 Q；随机动作预热无效；10-Q ensemble、LayerNorm、UTD 4 | 残差 SAC 的预热数据应改为目标地形上的 base rollout | 全文 | 4 |
| Stop Regressing / HL-Gauss（2403.03950） | 通用 | 用分类交叉熵代替 MSE 学价值，对噪声目标鲁棒 | 51-bin HL-Gauss critic；先把回报缩放到有界区间 | 全文 | 4 |
| π\*0.6 RECAP（2511.14759） | 实机 VLA | 用 MC 回报训练 201-bin 分布式价值，归一化到 (−1,0)；奖励为每步 −1 加失败惩罚，价值即"剩余步数" | 把价值改成"剩余距离/剩余步数"式的有界目标 | 全文 | 5 |
| GR-RL（2512.01801） | 实机 VLA | 在稀疏奖励下，有界分布式 critic 比回归 critic 稳；上线前先单独重训 critic 头 | 同上 | 全文 | 4 |
| SLowRL（2603.17092） | 四足 LoRA-PPO | LoRA 加在 actor 和 critic 的所有层，rank-1 最好，LoRA lr 1e-2；只调 actor 时 critic loss 不收敛 | 从 base critic 初始化并加 LoRA；提高 actor lr | 全文 | 5 |
| LoRA Without Regret（Thinking Machines 博客，2025） | LLM | RL 需要的容量很低；LoRA 最优 lr 约为全量微调的 10 倍 | 解释当前 KL 远低于目标的现象（lr 5e-5 可能偏低） | 全文 | 4 |
| RL-100（2510.14830） | 实机扩散策略 | 离策略评估预测会改进时才接受新策略 | 更新门控：训练地形上的确定性评测不变差才接受 | 全文 | 4 |

### 2.2 不依赖 critic 的低维搜索（绕开 critic，直接用 MC 回报）

| 工作 | 领域 | 机制 | 迁移做法 | 核验 | 相关 |
|---|---|---|---|---|---|
| Walk These Ways（2212.03238） | 四足 | 只在平地训练，策略以 8 维行为参数为条件（抬腿高度、步频、机身高度等），可泛化到台阶和湿滑地面 | 与"平地 base→未见地形"的设定同构；原文靠人手动调参，自动调参是空白 | 全文 | 5 |
| PMTG（1910.02812） | 四足 | 策略调制步态发生器并输出残差 | 残差用按相位参数化的基函数表示 | 摘要 | 4 |
| ITE（1407.3501，Nature 2015） | 六足 | 离线用 MAP-Elites 建行为-性能图，在线用 GP-BO 试错；损伤后约 8 次试验恢复；在原始 36 维参数上直接 BO 失败 | 低维（≤6）行为空间用 BO | 全文 | 5 |
| SO-CMA / MSO（1810.05751 / 1909.12995） | 四足、双足 | 策略族 π(s,μ)，目标域用 CMA-ES 或 BO 搜 μ；MSO 在实机上 15 个 episode 完成 | 每个候选用确定性回合评估 | 全文 | 5 |
| ES-MAML 批量爬山（2003.01239） | 四足实机 | 噪声大时用批量爬山代替策略梯度，50 个 episode | 候选先复测再接受 | 全文 | 4 |
| Peng et al. 2020（2004.00784） | 四足实机 | 在 latent z 的搜索分布上做 episode 级 AWR，约 50 次试验 | 参数空间 RWR | 全文 | 5 |
| PoWER / PI²（NeurIPS 2008 / JMLR 2010） | 机械臂、仿真四足 | 参数空间结构化探索，用 MC cost-to-go 加权；PoWER 指出每步独立白噪声会冲刷掉动作效果 | 更新规则；支持改用结构化噪声 | 全文 | 5 |
| Fast Adaptation with BFMs（2504.07896） | 运动、导航 | 在低维任务 embedding 上搜索；不用 critic 的 LoLA 是唯一单调提升的方法 | 做"无 critic 与带 critic"的对照 | 全文 | 5 |
| When Is a Learned Command Adapter Worth It?（2607.21867） | Go2/H1 | 审计冻结 locomotion 策略的 adapter：理论 headroom 5.2%，实际只学到 0.55%，结论 NO-GO | 做任何适应之前，先量化可回收的 headroom | 摘要 | 4 |
| RTR（2508.12252） | 人形 | 冻结 actor，只用 PPO 调动力学 latent；直接微调 π 会崩 | 只调低维 latent 加单独 critic | 全文 | 4 |

### 2.3 steering + 残差（USR、FlowDAgger 一族）

| 工作 | 领域 | 机制 | 迁移做法 | 核验 | 相关 |
|---|---|---|---|---|---|
| USR（OpenReview 9sQVoVMeSD） | 操作/VLA | actor 同时输出 steering 噪声 w 和残差 r；Q_A 蒸馏到 Q_C；残差按 ε 逐步放开；γ=0.97。实机开盖任务：base 40%、DSRL 75%、USR 90% | 先 steering 后残差的门控顺序。审稿意见：主 benchmark 为自建，在 Adroit 上并不优于 DSRL | 摘要＋审稿/rebuttal | 5 |
| FlowDAgger（2607.08877） | 操作/VLA | 把纠正动作反演成噪声 w\*，回归一个小 noise policy；干预数据与成功自主 rollout 各占 50% | 纠正来源由人换成"自动纠正"（快照分支或成功回合）；Gaussian base 下反演即残差回归 | HTML 正文 | 5 |
| UniSteer（2605.10821） | VLA | 纠正反演到噪声空间，同一个 noise actor 同时做监督学习和 RL；实机 66 分钟从 20% 提到 90% | 反演得到的目标可用作 actor 的 BC 正则 | 摘要 | 4 |
| FRS（2606.13675） | VLA | 把粗糙动作反推回噪声、映射到附近模式后再蒸馏 | 需要 flow 先验，目前不适用 | 摘要 | 3 |
| SteerGenPO（2606.16572） | Isaac Lab，G1 行走 | PPO 训练生成策略，再学确定性的状态条件 latent actor | 最近的腿足近邻工作（新颖性威胁） | HTML 正文 | 5 |
| NaP-Control（2605.20209，ECCV 2026） | 角色控制 | 在扩散先验的噪声上做 PPO，用于先验里没见过的起伏地形 | 近邻工作；需要高度图和大 batch | HTML 正文 | 4 |
| ZPRL（2605.19919）/ PULSE（2310.04582） | 操作 / 角色 | 在瓶颈 latent 上做残差，而不是直接做动作残差；PULSE 的消融显示，latent 动作写成"相对先验均值的残差"很关键 | 说明动作层残差的探索质量差 | 摘要 / 全文 | 4 |
| RFS（ICLR 2026 Poster） | 操作 | 组合两种策略 | "steering 加残差"这一组合本身已经有人做过 | 审稿页 | 3 |

### 2.4 自模仿、DAgger 与自动纠正

| 工作 | 领域 | 机制 | 迁移做法 | 核验 | 相关 |
|---|---|---|---|---|---|
| SIL / AWR（1806.05635 / 1910.00177） | 通用 | 只模仿 (R−V)+ 大于 0 的样本；按 exp(A/β) 加权回归 | 自模仿更新规则。SIL 的下界只对 Q\* 成立，对确定性均值策略不成立 | 全文 / 摘要 | 4 |
| ESPER（2205.15967） | 离线 RL | 单条轨迹的回报可能来自运气，改用"聚类平均回报"作条件 | 用来诊断成功是否只是噪声运气 | 全文 | 4 |
| RIPT-VLA（2505.17016） | VLA | 同一初始条件跑 K 条，用留一均值作基线；结果全同的组丢弃；不用 critic | 在同一地形、同一扰动起点内做组内比较 | 全文 | 4 |
| PLD（2511.00091） | VLA | 残差 SAC 探测失败区域 → 混合 rollout → SFT 蒸馏；残差限幅为 ξ | 最后把搜索到的解蒸馏回 LoRA | 全文 | 4 |
| Sirius（2211.08416）/ RLIF（2311.12996） | 实机操作 | 干预前的样本降权；把干预本身当负奖励 | 把安全投影、回滚、跌倒前的窗口降权，并作为扣分项 | 全文 | 3 |
| CR-DAgger（2506.16685） | 接触操作 | 在冻结 base 上学增量残差，50–100 个纠正回合就优于全量微调 | 支持"冻结 base＋残差/LoRA"的形式 | 全文 | 3 |
| LbC / CTS（1912.12294 / 2405.10830） | 驾驶 / 足式 | 特权教师在学生实际访问的状态上在线标注 | 快照分支自动纠正采用同样的标注方式 | 全文 | 3 |
| VinePPO（2410.01679） | LLM | 从中间状态做 MC 分支估计优势，不用 critic | 在停滞点保存仿真快照后分支 | 相关文献中提及 | 3 |

### 2.5 离策略残差（核对残差 SAC 配置）

| 工作 | 领域 | 要点 | 核验 |
|---|---|---|---|
| A Walk in the Park（2208.07860） | 四足实机 | 主要收益来自 LayerNorm 和 UTD 20；动作范围收窄 | 全文 |
| TWiRL（2304.09834） | 四足 | 目标环境上 base 的数据固定占 50% 批次；迁移数据比只迁移权重更关键 | 全文 |
| ResFiT（2509.19301） | 实机人形操作 | LayerNorm critic、ensemble、UTD≈4、n-step=3、delayed actor；残差 PPO 约需 200 倍样本 | 全文 |

### 2.6 反证与限制

- **What Can RL Bring to VLA Generalization（2505.19789）** 和 **Learning Without Critics?（2511.03527）**：在长时域任务上，无 critic 的组相对方法（GRPO）不如 PPO。因此在我们的方案里，组内相对优势只用于诊断或低维搜索，不直接替代全参数的 actor 更新。
- **RL's Razor（2509.04259）**：遗忘程度可由平地状态上的前向 KL(π0‖π) 预测，可作为 S2 平地保持的代理监控指标。

## 3. 由此得到的候选方法

以下 M0–M4 只是候选。能否进入 G3，由 G1 的门槛决定（见计划 v2.1 第 5 节）。

| 编号 | 方法 | 来源 | 是否用 critic | 最便宜的证伪 | 主要风险 |
|---|---|---|---|---|---|
| M0 | LoRA-PPO＋价值修复：MC 目标预训练 critic；Decoupled-GAE；有界回报或 HL-Gauss；整回合 batch；LoRA lr 提高 10 倍；可选从 base critic 初始化 | VC-PPO、VLA-RL、RECAP、SLowRL | 是 | 在 101、107 上，critic 预训练后留出回合的 EV 是否大于 0.3（每个地形约 4 回合） | 修好 critic 后，actor 可能仍然只学到噪声平均 |
| M1 | 自监督 FlowDAgger：拟合 ε\* = (a_exec − μ(o))/σ；用 1 秒窗口的 MC 优势做 AWR 加权；batch 一半来自平地，目标为 ε=0 | FlowDAgger、SIL/AWR、RECAP、RIPT | 否（一维按位置平均的基线） | 预检：按回合交叉验证拟合 o_t→δ_t 的 R²；若 R²≈0，说明成功只是噪声运气，直接放弃 M1 | 因果混淆；多种走法被平均掉；背下路线 |
| M2 | 低维行为 latent 搜索：latent 为按相位参数化的残差基（抬腿、机身高度、髋前摆，前后腿分开，4–8 维）或执行残差的 PCA 基；用 CMA-ES、BO 或 PI² 优化；每个候选跑确定性回合 | WTW、PMTG、ITE、MSO、PoWER/PI²、BFM-LoLA | 否 | headroom 网格：在 101、107 上对"抬腿幅度×机身高度"做 5×5 网格（每个地形约 25 回合、不到 1 GPU 小时）；若没有任何格点让前进距离比 base 提高 30% 以上，就换基或放弃 | 冻结策略会把残差当扰动抵消；状态无关的残差对台阶反应不足 |
| M3 | 快照分支自动纠正 DAgger：在停滞或高风险状态保存仿真快照，分出 K 条短分支，按短程 MC 回报取最优分支作为纠正；训练 batch 为 50% 纠正样本＋50% 平地/自主样本 | FlowDAgger、LbC/DAgger、VinePPO、PI² | 否 | 快照恢复并重放同一动作序列，75 步内偏差小于 1e-3 m；在 101 号 0.65 m 停滞点，最优分支能否越过 | PhysX 接触缓存导致恢复不是逐位一致；属于仿真特权，分支步数必须计入预算 |
| M4 | USR-lite：M2 的 latent steering 加上按 ε 门控的残差（M0 或 SAC） | USR | 是 | 只在 M2 已找到有效 latent 时才做 | 退化为残差 SAC，critic 问题照旧 |

另有两项作为附加改动，可以叠加到任一方法上：
- **结构化探索噪声**：时间相关噪声（如 gSDE、低通滤波噪声）或参数空间噪声，让探索扰动能通过 0.03 的限速器，被执行的扰动也属于策略类本身。
- **更新门控**：每次更新后在训练地形上跑一次确定性评测，不变差才接受（RL-100、批量爬山）。

## 4. 新颖性定位

最近的工作有：SteerGenPO（Isaac Lab、腿足，latent actor 引导生成策略）、NaP-Control（噪声 steering 用于未见起伏地形）、USR/RFS（steering 加残差）、WTW/Peng 2020/SO-CMA/MSO（四足上的 latent 或行为参数搜索）、BFM 快速适应（几十个回合、有无 critic 对比）。

目前还没有一篇工作同时满足以下条件：
- 单个环境，20–100 个回合；
- 仅本体感知；
- 从单峰 Gaussian 平地 base 出发做四足地形适应；
- 在同一协议下比较 critic 修复后的梯度方法、无 critic 的低维搜索和自动纠正 DAgger，并以冻结后的稳定完成为终点。

这是论文可以主张的空白。如果 G3 结果是负面的，就按 F2 转向机制分析。
