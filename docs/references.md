# 项目参考文献表

更新时间：2026-09-07

这份清单把项目计划、设计文档和在线学习原理中的文献统一起来。年份优先采用正式发表年份；只有预印本时采用 arXiv 首次公开年份，并保留 arXiv 链接。“状态”列用来区分“论文中可直接作为方法依据”和“仅作启发/待核验”，避免把项目没有实现的算法写成已复现结果。对应的 BibTeX 文件为 [`references.bib`](references.bib)。

## A. 基础强化学习与控制

| Key | 文献 | 年份 | 项目中的用途 | 状态 |
|---|---|---:|---|---|
| schulman2017ppo | Schulman et al., *Proximal Policy Optimization Algorithms* | 2017 | 冻结的 Lite3 基础 PPO、GAE 和 clipped objective | 直接方法依据 |
| haarnoja2018sacapp | Haarnoja et al., *Soft Actor-Critic Algorithms and Applications* | 2018 | 最大熵 SAC、双 Critic、目标网络、重参数化和温度调节 | 直接方法依据 |
| johannink2019residual | Johannink et al., *Residual Reinforcement Learning for Robot Control* | 2019（arXiv 2018） | “基础控制器 + 有界学习残差”结构 | 直接结构依据 |
| rudin2022minutes | Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* | 2022（CoRL 2021） | `legged_gym`/大规模并行四足 PPO 的工程背景 | 训练框架依据 |

## B. 本体感知与四足地形适应

| Key | 文献 | 年份 | 项目中的用途 | 状态 |
|---|---|---:|---|---|
| kumar2021rma | Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* | 2021 | 用本体感知历史推断环境变化；用于与 residual frozen 对照 | 结构启发，不是完整复现 |
| nair2020awac | Nair et al., *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets* | 2020 | 离线数据暖启动和降低在线探索成本的对照思想 | 概念对照，未采用 AWAC 更新 |
| wu2022daydreamer | Wu et al., *DayDreamer: World Models for Physical Robot Learning* | 2022 | 真实机器人在线学习和样本效率的参考 | 概念对照，未实现世界模型 |
| nahrendra2023dreamwaq | Nahrendra et al., *DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning* | 2023 | 本体感知历史、隐式地形表示和四足鲁棒运动 | 结构启发，不是完整复现 |
| zhang2025motionpriors | Zhang et al., *Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility* | 2025 | 平地技能迁移、运动先验和 Lite3 mocap 试验的背景 | 部分工程验证，主线未依赖 |

## C. 离线到在线强化学习与价值校准

| Key | 文献 | 年份 | 项目中的用途 | 状态 |
|---|---|---:|---|---|
| ball2023rlpd | Ball et al., *Efficient Online Reinforcement Learning with Offline Data* | 2023 | 源 replay 与在线 replay 混合、在线样本效率 | 设计参考；混合比例仍需实验确认 |
| nakamoto2023calql | Nakamoto et al., *Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning* | 2023 | Q 值尺度失配、保守/校准 Critic 的诊断依据 | 候选扩展，未声称已实现 Cal-QL |
| kumar2020cql | Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning* | 2020 | Q 失配明显时的保守价值学习候选 | 可选消融，非主线 |

## D. 持续适应、安全和残差更新

| Key | 文献 | 年份 | 项目中的用途 | 状态 |
|---|---|---:|---|---|
| schmidgall2023synaptic | Schmidgall and Hays, *Synaptic Motor Adaptation: A Three-Factor Learning Rule for Adaptive Robotic Control in Spiking Neural Networks* | 2023 | 运行中改变控制参数的实时适应案例 | 相关工作，不采用脉冲网络 |
| smith2023growlimits | Smith et al., *Grow Your Limits: Continuous Improvement with Real-World RL for Robotic Locomotion* | 2023 | APRL 风格的策略正则化、行为漂移和持续学习 | 安全/正则化启发 |
| wu2025loopsr | Wu et al., *LoopSR: Looping Sim-and-Real for Lifelong Policy Adaptation of Legged Robots* | 2025（arXiv 2024） | 真机数据回流仿真、长期适应和部署边界 | 后续真机路线参考 |
| zhong2025bas | Zhong et al., *Bridging Adaptivity and Safety: Learning Agile Collision-Free Locomotion Across Varied Physics* | 2025 | 适应与安全切换、恢复策略和 reach-avoid 思想 | 安全扩展参考 |
| nai2025powersaving | Nai et al., *Fine-Tuning Hard-to-Simulate Objectives for Quadruped Locomotion: A Case Study on Total Power Saving* | 2025 | 真实数据辅助的功耗/难仿真目标微调 | 可选扩展 |
| jayasinghe2026residual | Jayasinghe et al., *Residual Control for Fast Recovery from Dynamics Shifts* | 2026 | 残差幅度、方向一致性和性能条件门控 | 预印本，待进一步核验 |

## E. 近期腿式 SAC 与相关预印本

| Key | 文献 | 年份 | 项目中的用途 | 状态 |
|---|---|---:|---|---|
| sabatini2026leggedsac | Sabatini et al., *Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged Locomotion* | 2026 | SAC 初始化、timeout-aware target、multi-step return | 预印本；须独立消融 |
| qin2025dmloco | Qin et al., *Integrating Diffusion-based Multi-task Learning with Online Reinforcement Learning for Robust Quadruped Robot Control* (DMLoco) | 2025 | 仿真 PPO 微调与在线 RL 的近期对照 | 预印本；不能等同真机在线更新 |
| wang2026latentmimic | Wang et al., *LatentMimic: Terrain-Adaptive Locomotion via Latent Space Imitation* | 2026 | 潜空间模仿和地形适应的近期观察 | 预印本，仅基于摘要 |
| kordos2026locomposition | Kordos et al., *LoComposition: Terrain-Adaptive Energy-Efficient Quadruped Locomotion without Gait Priors* | 2026 | 无步态先验、能耗和地形适应的近期观察 | 预印本，仅基于摘要 |

## 链接与核验说明

- 每条 arXiv 链接和 BibTeX 条目见 [`references.bib`](references.bib)。
- `APRL`、`LoopSR`、`BAS` 在早期计划中曾只出现缩写；本表根据后续 worktree 原理文档补充了可识别题名。正式提交论文前仍应下载全文，核对作者顺序、正式会议版本和页码。
- 2025--2026 条目可能在后续产生会议版本；若论文引用格式要求“正式发表优先”，应在定稿前更新 `booktitle`、`pages` 和 DOI。
- Isaac Lab、RSL-RL、DeepRobotics `rl_training` 和 AI4Animation 是项目所用软件/数据资源，不与上述学术论文混在一起；应在论文的“软件与数据来源”小节单独引用。
