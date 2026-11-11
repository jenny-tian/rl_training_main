# 原项目上下台阶代码复用审查

2026-11-11 用户提出参考 `/home/tjx/rl_training`。本次只读对照该目录与当前工作树，没有修改正在运行的训练配置。原目录 HEAD 为 `127b4ff`，未跟踪的 `README_LITE3_TRAINING.md` 保持原样；原目录未发现 `logs/`，因此没有依据声称其 Lite3 已通过上下台阶训练验证。

## 关键事实

1. 原项目通用场景引用 Isaac Lab 的 `ROUGH_TERRAINS_CFG`，确实包含 `pyramid_stairs` 和 `pyramid_stairs_inv`。安装的配置位于 `/home/tjx/IsaacLab/source/isaaclab/isaaclab/terrains/config/rough.py:22`，通用台阶参数为 0.05–0.23 m 高、0.3 m 踏面、3 m 中心平台。
2. 原项目 Lite3 子配置在 `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py:105` 覆盖采样比例：粗糙 0.4、两类坡各 0.3、箱体和两类台阶均 0.0。生成器存在不代表当前 Lite3 配置实际训练这些台阶。
3. 原 Rough 在同文件第 62 行关闭 actor 高度扫描，但没有关闭 critic 高度扫描。critic 项定义在原项目 `velocity_env_cfg.py:241`，大范围扫描器位于第 75 行。当前 Flat 派生课程的 actor 45/critic 48 维均不包含地形扫描；9 射线底部扫描仅服务奖励与安全检查。
4. 原 Rough 在第 178 行启用 `phase_foot_trajectory_exp`，权重 2.0；同时启用步态和镜像奖励。当前 Flat 派生任务只启用自己的奖励白名单，因此这些项关闭。该函数使用预设周期、相位、机体系摆腿轨迹，不是根据台阶边缘实时计算落脚点。
5. 原 Rough 第 205 行将 `illegal_contact` 设为 None。当前任务保留非足接触终止和统一安全限制，已有失败诊断指向 SHANK 接触；关闭终止不能作为通过原资格评估的修复。
6. 对双方 `mdp/rewards.py` 的函数/类做 AST 比较，共有 51 项中 48 项完全一致（忽略行号和注释）。`phase_foot_trajectory_exp`、`GaitReward`、`joint_pos_penalty_except_turn_side_cmd` 均相同。当前代码已有这些可复用实现；主要差别在启用方式。`base_height_l2` 则包含本项目已修复的每环境有效性处理，不应回退覆盖。
7. 原项目 `velocity_env_cfg.py:56` 设置 `max_init_terrain_level=5`，`:881` 启用地形课程；`mdp/curriculums.py:27` 根据行走距离升降级。安装的 importer 在初始 0–5 行间随机分配，不是全部固定在第 5 行，也不能等同于本项目定义的 L2。原通用布局为 10×20 个 8×8 m 地块。

## 可复用方向及限制

| 参考项 | 可以借鉴什么 | 当前实验中为何不直接替换 |
| --- | --- | --- |
| critic 高度扫描 | 训练时向价值网络提供地形信息，actor 可继续保持本体观测 | critic 输入维度改变，需要事先确定权重/Adam 迁移方式及独立训练预算 |
| 摆腿轨迹、步态奖励 | 检查已有摆腿目标与小腿碰撞之间的关系 | 启用会改变当前无相位先验的训练目标，尚无本任务收益证据 |
| 通用上下台阶生成器 | 作为几何和碰撞实现的独立参考 | 原几何是中心平台/金字塔布局，当前为直线、固定步数与踏面；直接换地形会改变试验条件 |
| 关节姿态惩罚地形门控 | 允许特定地形上关节更偏离默认姿态 | 原函数检测相对环境起点的正向高度超过 0.05 m，对向下地形不直接成立；当前对应惩罚项也未启用 |
| 非足接触终止关闭 | 解释原任务和当前验收的差异 | 改变安全标准，不能用来宣称解决小腿接触 |

当前定向试验继续按已批准预算完成。后续若需要调整训练目标，优先提交基于本次逐行计数和独立评估的具体方案，再由用户确认；不在运行中途改动地形、观测、奖励或安全标准。

用户随后提出固定 L2 和分块训练，并选择保留现有六类、复用分块布局与台阶，箱体继续留给阶段二。已整理 [下一阶段具体提案](block-training-proposal.md)。这是范围选择，不表示已启动新任务或新正式训练预算。
