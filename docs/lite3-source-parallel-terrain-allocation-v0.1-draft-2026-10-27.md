# Lite3 Source PPO 并行地形分配设计与离线验证

版本：v0.1 draft，2026-10-27

状态：`draft_not_for_run`

执行授权：`false`

本步承接单环境 reset sampler，完成兼容 120 个 Source PPO 环境的地形位置分配原型与离线测试。120 仍是基础技能训练的吞吐候选；C1-a 在线适应当前使用单环境、完整回合交互，两阶段分别配置。本文件不改变运行配置，也不授权仿真或 PPO 更新。

## 1. 已发现的约束

当前地图为 1 行 × 120 列，每个位置只能分配给一个活动环境。若 120 个环境占满地图，而只有一个环境 reset，它只有原位置空闲；不能保证预登记的下一种地形可用。

随机重抽直到找到空位会改变 reset 配比；把仍在运行的机器人搬走会破坏完整回合；强制全部环境同步 reset 则改变回合长度、终止与训练数据语义。因此，单环境的列洗牌器不能直接推广到满占用的 120 个环境。

## 2. 候选：每个环境保留完整地形组

原型为每个环境预留一行，每行包含全部 15 个家族/方向子类，每个子类一个 tile。环境 `i` 永远只用第 `i` 行；地形子类决定列。任何时刻两个不同环境的行号不同，所以不会获得相同的 `(row, column)`。

| 项目 | 120 环境候选 |
|---|---:|
| 行数 / 行的含义 | 120 / 环境所有权 |
| 每行列数 | 15 个子类，各一列 |
| 地形总数 | 1,800 |
| 相对当前 120-tile 地图 | tile 数量为 15 倍 |
| 沿用 12 m × 4 m tile 的未加隔离间隙尺寸 | 1,440 m × 60 m |
| 活动机器人 | 120；其余位置预留 |

这是用地图规模换取简单、可验证的异步分配保证。1,800 不是所有可行设计的最小 tile 数，120 也不是已批准的环境数；如内存或建图耗时不合适，应在锁定预算前减少并行数或另审分配设计。不能因资源不足偷偷改为占用相关的重抽策略。

调度核心：[parallel_terrain_allocation.py](../source/rl_training/rl_training/source_scan/parallel_terrain_allocation.py) 中的 `ReservedRowTerrainAllocator` 只用 Python 标准库，不调用 Isaac、Torch、地形生成器或训练器。相同模块另提供可选的运行时适配函数，负责验证真实地形 importer buffers 并在 curriculum reset hook 写入新分配；离线假环境测试覆盖该写入路径。

## 3. 地图容量与回合配比分开

物理地图每行需要 15 个子类各一列，因此候选生成器使用 **15 个相等的地图权重**。回合配比由独立的整数票据表控制，不能把原来的比例直接套到 15 列上；稀有方向可能没有列。15 列等权符合当前 Isaac Lab 累计比例分列公式。

| 子类 | 每 120 次分配数量 |
|---|---:|
| Flat | 12 |
| Multi-scale Rough | 24 |
| Slope uphill / downhill / cross-left / cross-right | 各 6 |
| Stairs ascending / descending | 各 15 |
| Slope + Rough uphill / downhill | 各 6 |
| Slope + Rough cross-left / cross-right | 各 3 |
| Friction left-right / front-back / alternating | 各 4 |

每块 120 张票据按显式 seed 洗牌，每次分配消费一张；初始的 120 个 episode 也计入第一块。之后按照 reset 事件批次、批次内部递增的 env ID 消费，跨批次延续同一队列，不跳过、不因失败重抽。返回记录包含 env ID、行列、子类、全局分配编号、每环境 episode 编号以及块内位置，可用于后续审计日志。

配比保证是**全局每 120 次新 episode 分配**，不保证每个环境、每次 PPO 更新或同一时刻的活动环境比例。回合长短不同，transition 配比仍可能不同。后续 runner 必须同时记录 reset 和 transition 计数，未完成的尾块按实际数量报告。

可复现需要相同的 seed、子类顺序以及 reset 批次序列；仅 seed 相同而回合结束顺序不同，不能保证每个环境得到相同轨迹。构造器复制输入配比，调用方之后修改字典不会改变已建立的调度。

## 4. 生命周期与错误行为

1. 首次非空分配必须覆盖全部环境，避免部分环境仍使用默认 importer 的共享位置。空批次不消费票据。
2. 每次先校验全部 reset ID：重复、越界、布尔值、非整数或无效序列会在修改 RNG/占用状态前被拒绝。
3. 合法批次按 env ID 排序，只替换这些环境的分配；未 reset 环境的行列、episode 编号和分配记录保持不变。
4. 分配不依赖其他环境是否结束，故“1 个环境反复结束、119 个环境持续运行”不会发生等待或换位。

原型 reset 适配已实现为默认关闭的可选配置：Importer 启动时先为每个环境绑定独立 row 和第一列 origin；ManagerBasedRLEnv 的 reset curriculum hook 随后在 `scene.reset` 与根状态 reset 事件前，只更新本次 reset 环境的列、row 和 origin。启用必须同时显式给出并行环境数与 sampler seed。每个地形行代表环境所有权，而非 difficulty level。

Reset 调度器在写环境 buffers 前先验证新一批分配，并对后续分配使用候选调度副本；若写入失败，回滚目标 buffers 且不推进存活调度器。原型还没有 checkpoint 调度状态序列化。将来恢复 PPO 训练时必须恢复洗牌队列、游标、RNG、活动分配和 episode 计数，或从完整 reset 事件记录重建；仅重建同 seed 分配器会从第一块重新开始，不能声称无缝续训。

## 5. 离线证据与适用边界

测试：[test_parallel_terrain_allocation.py](../tests/source_scan/test_parallel_terrain_allocation.py)。覆盖首次全环境分配、119 个环境持续占用时的 1,200 次单环境 reset、四个调度 seed 下各 300 个异步 reset 批次、每个完整块的家族/方向计数、复现、顺序与分批语义、非法批次原子拒绝，以及 1/7/240 环境与 120 回合块的独立性。测试还从真实 Source 配置提取纯配置工厂，核对 15 个实际名称和权重，不启动 Isaac Sim。

验证结果：并行分配器/importer-reset 辅助测试 **49 passed**。测试模拟 importer 初始 origin 布局、reset hook 更新、假 root-reset 消费新 origin、异步环境不动，以及无效地图拒绝。Python 语法与 `git diff --check` 通过。另完成一次 2 环境 Isaac Sim importer/reset smoke，运行记录见[结果 JSON](lite3-source-parallel-terrain-allocation-runtime-smoke-gpu0-2026-10-27.json)。

离线断言验证的是**唯一地形索引和调度不变量**。2 环境 smoke 生成了 `(2, 15, 3)` 地图，并检查启动 origin/root、一次零动作物理步、两个 height scanner 的有限输出及单环境异步 reset。尚未生成 120×15 地图，也没有测量该规模下的碰撞隔离、GPU 显存或吞吐，不能据此称 120 环境训练已资格化。

验证命令（环境已有的 typeguard pytest 插件与 typing_extensions 不兼容，关闭自动加载第三方插件；保留 pytest 内置插件）：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_scan/test_parallel_terrain_allocation.py
```

## 6. 下一步：扩大到独立的并行物理隔离与资源资格检查

Source 场景默认仍为 `scene.num_envs=1`，并行 sampler 默认关闭；并行适配只有在显式启用并给出环境数和 seed 后才会构建多行地图。以下运行资格仍未完成：

- **启动时物理检查：** 2 环境 smoke 已在 GPU0 确认真实 root state 与各自 origin 对齐，首个零动作物理步及两类高度扫描输出有限。该结果只覆盖 2×15 地图，不证明 120×15 布局或高负载场景。
- **地形和 origins 契约：** 验证实际 `(120, 15, 3)` origin grid、15 个等权列及固定 L1 几何，设置 `terrain_levels[env_id]=env_id`、`terrain_types[env_id]=column`、`env_origins`。关闭自动行难度课程。当前 Source adapter 只用 difficulty 派生几何 seed，多行会改变该派生分布，必须记录和审计；行号不代表晋级 L2。
- **reset 与日志：** 2 环境 smoke 已确认仅 reset 环境 0 会消费下一张全局票并更新其行列/origin/root，同时环境 1 的分配、origin 和 root 保持不变。仍须在独立 reset 生命周期检查传感器缓存、命令、接触、foot histories、新回合标签和失败后的完整审计日志。
- **物理隔离：** 唯一 origin 不足以保证运动中不碰撞。须验证机器人活动范围、跨 tile 越界终止、reset 偏移、邻接几何、collision filtering 和所有高度扫描 mesh，包含首个 physics step 与异步 reset。当前 Source 配置禁用了 terrain-out-of-bounds termination，不能直接运行本布局。
- **资源与预算：** 测量新地图生成和 ray-caster mesh 的内存、耗时、并行吞吐；只在实测后锁定环境数和 `num_envs × rollout_length` 预算。本设计没有启动 262,080-transition Pilot。
- **继续训练：** 补齐调度状态保存/恢复、初态 seed 消费与完整审计日志；继续保留 Source 资格、安全桥和训练授权门。

对应[执行锁定清单](lite3-stairs-l1-source-training-lock-checklist-v0.1-draft-2026-10-27.md)仍为草案。[并行 importer/reset smoke 方案](lite3-source-parallel-terrain-importer-smoke-plan-v0.1-draft-2026-10-27.md)已完成 2 环境诊断；下一步是单独准备并行物理隔离、地图资源和吞吐资格检查，不能直接扩大为 120 环境训练。
