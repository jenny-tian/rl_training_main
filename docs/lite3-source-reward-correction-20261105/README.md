# Source 奖励与探索初始化修正：配置阶段汇报

本阶段已完成配置代码、原模型初始化接口、针对性 CPU 检查和真实 checkpoint 离线核对。新增配置名为 `DeeproboticsLite3SourceAdaptationEnvCfg`，奖励配置版本为 `lite3_source_lowspeed_v1`。实际仿真步数、训练交互和 PPO 更新均为 0；还没有产生新的训练模型，也没有验证重新训练后的行走表现。

用户在退化诊断后要求每阶段详细汇报，并同意继续完成奖励与初始化配置修正。本阶段按此范围执行。参考项目 `/home/tjx/rl_training` 的主要权重已被旧 Source 沿用，本次以能够行走的原模型保存配置为基准，恢复其 17 项非零奖励。保存配置 SHA256 为 `860a7cb7f8c0949cc431b1f8738960b1849a5970aa2b906aac6d953de9902a31`。

**已完成的修改。**

| 项目 | 旧 Source 90 轮设置 | 新配置 | 修改依据 |
|---|---|---|---|
| 平面速度奖励宽度 | 0.7071 | 0.25 | 降低有命令却静止时的得分 |
| 转向速度奖励宽度 | 0.7071 | 0.5 | 恢复原模型设置 |
| 膝关节、外展关节姿态惩罚 | -2、-0.4 | 移除这两个新增项 | 避免新增姿态约束压过低速行走收益 |
| 奖励系数 | 依赖全局变量，原采样实测为 0 | 环境内显式 `reward_gait_level=1.0` | 避免地形课程关闭后奖励长期归零 |
| 机身高度 | 目标 0.55 m，权重 -50，实际被门控关闭 | 目标 0.375 m，权重 -25，使用局部地面扫描 | 恢复原高度目标，同时适配坡面和台阶 |
| 垂直速度、水平姿态惩罚 | -20、-20 | -2、-10 | 恢复原模型保存权重 |
| 动作变化惩罚 | -0.1，实际被门控关闭 | -0.05，系数为 1 | 恢复原来的平滑约束 |
| 接触力、落脚冲击 | -0.1、-2，实际被门控关闭 | -0.01、-0.5，系数为 1 | 恢复原模型的物理正则强度 |
| 腾空时间与时长方差 | 使用 Rough 的直行/转向项，方差项关闭 | 原腾空时间 +2、时长方差 -5 | 恢复原模型的步态奖励组合 |
| 相位轨迹、步态接触模板等 | 继承 Rough 的新增项 | 从本版本奖励集合移除 | 保持与原训练目标的明确对应 |
| 动作探索标准差 | 全部重置为 1.0 | 完整保留 checkpoint 的 12 维值，平均 0.34315 | 避免已验证的探索噪声突增 |

完整的 17 项定义见 [reward_profile.json](deliverables/reward_profile.json)，与旧 Source 的逐项差异见 [reward_changes.csv](deliverables/reward_changes.csv)。原保存配置中所有非零项的函数、权重、参数已逐项比对；唯一地形适配差异是 `base_height_l2.sensor_cfg` 从无地面参考改为 `height_scanner_base`，使用机器人下方的小范围扫描计算相对高度。

奖励系数为 1 表示相关项不再被全局门控关闭，不代表每一步所有项都应非零。例如没有零速度命令时，`stand_still` 正常为零；没有落脚事件时，落脚相关项也可以为零。真实分项是否符合预期仍要在下一阶段检查。

**代码如何接入。**

- [adaptation_profile.py](../../source/rl_training/rl_training/source_scan/adaptation_profile.py)定义独立奖励集合、配置核对和保留探索标准差的 actor 初始化接口。每个环境构建独立对象，修改一个环境不会改变其他环境。
- [source_adaptation_env_cfg.py](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_adaptation_env_cfg.py)在原 Source Training 环境构造完成后装入新奖励。它处理了父类删除零权重奖励，以及 `configclass.replace()` 再次构造时的配置类型问题。
- [rewards.py](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py)的系数读取优先使用环境内配置；没有该字段的旧环境继续使用原地形课程变量。
- [source_training.py](../../source/rl_training/rl_training/source_scan/source_training.py)增加 `action_std=None`，表示保留 checkpoint 标准差，并在写入 actor 之前检查标准差是否可表示为有限正数。旧的显式常数初始化仍可复现。
- [source_adaptation_preflight.py](../../scripts/tools/source_adaptation_preflight.py)对奖励定义和真实 checkpoint 做 CPU 核对，不导入 Isaac，不构建仿真场景，也不更新参数。

下阶段运行入口应显式配对使用新配置和 `initialize_source_adaptation_actor(actor, checkpoint, cfg)`。旧的 `source_training_entrypoint.py` 仍执行旧实验合同，不能把旧授权文件换个名字用于新版训练。本阶段没有更改旧的 90 轮实验配置、授权记录或预算规则；新增配置尚未连接到新的受限执行入口，该入口属于下一阶段工作。

新配置继承原 Source L1 场景：120 个并行环境、15 个固定条件、向前 0.3 m/s、40 秒回合上限，以及原观测、动作缩放、动作裁剪、地形采样和安全终止。变更中的倾斜保护奖励移除属于奖励集合调整，既有安全阈值和终止逻辑保持原实现。

**本阶段检查结果。**

36 项针对性 CPU 检查通过，耗时 1.44 秒。检查覆盖独立保存配置对照、奖励参数漂移检测、环境间系数隔离、局部地面高度平移不变性、父类被删除奖励恢复、原 actor 初始化兼容性、非均匀探索标准差保留，以及非法标准差在写入前拒绝。按本阶段范围未运行完整 PPO 更新测试，也未扩大到全仓库测试。[检查日志](raw/targeted_checks.log)、[JUnit 结果](raw/targeted_checks.junit.xml)已保存。

第一次 pytest 启动被环境中自动加载的 typeguard 插件与 typing_extensions 不兼容阻止，尚未执行测试；保存了[启动失败原件](raw/pytest_plugin_startup_failure.log)。后续关闭无关插件的自动加载，仍使用用户指定的 `/home/tjx/anaconda3/envs/env_isaaclab/bin/python` 完成检查，没有修改 Conda 环境。

真实 checkpoint 核对使用原 `model_1998.pt`，并读取前一阶段原模型和候选轨迹中已保存的 960 个真实观测。新接口加载后的全部 actor 张量与 checkpoint 完全相等；与原冻结 actor 在相同 CPU 观测上的动作最大绝对差为 0。12 维标准差保持原值，平均 0.3431523；随机数状态不变，CUDA 未初始化。完整结果见 [cpu_preflight.json](raw/cpu_preflight.json)。

原训练台账前后 SHA256 均为 `f171d9fdbe380a20c827453c52bb30ac30c8f6ed431f45c01f796625be303435`，累计计费交互仍为 259584 / 262144，剩余 2560。原 checkpoint 内容不变。

这些是 CPU 配置和真实 actor 核对结果。SDK 场景构造、真实奖励管理器分项和新训练表现尚未验证。旧轨迹没有保存恢复全部奖励所需的力矩、落脚历史等原始量，因此不能把单项速度重新计分当成新版完整奖励已经获胜。

**下一阶段准备做什么。**

下一阶段首先实现并运行一次受限的新版配置只读验证。使用 GPU0、相同 120 环境和 15 个条件，对原模型与 90 轮候选各运行 400 个控制步，总上限 96000 条诊断交互，训练更新为 0；总运行期限 300 秒，场景启动期限 120 秒，关闭期限 15 秒。它将验证新配置真实启动、17 项奖励的实际注册与分项、系数为 1、原模型继续行走，以及新版奖励是否降低静止候选的相对得分。记录动作裁剪、回合结束和配对初态；遇到非有限数据、配置不符或超时立即终止并保存现有结果。

该阶段结束后单独汇报实际结果，再决定短训练参数。初步短训练规模为从原模型出发进行 2 轮 PPO，即 5760 条训练交互，并在第 1、2 轮保存 checkpoint；不自动扩展长训练。原数值预算剩余 2560，不能容纳这两轮。若后续执行，需要明确单独的增量预算记录，保留历史计费，累计计费将达到 265344。critic 和学习率在本配置阶段没有新增改动，不能把这次修正当成已经验证了它们的影响。

下一阶段只读验证和后续两轮训练均尚未执行。具体上限、检查项与待定事项见 [next_stage_plan.json](deliverables/next_stage_plan.json)。
