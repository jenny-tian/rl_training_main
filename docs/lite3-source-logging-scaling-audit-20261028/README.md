# Source 日志修复与环境规模诊断（2026-10-28）

状态：`offline_logging_repaired_scaling_audited_runtime_failure_unresolved`。

本阶段接续用户同意的离线诊断，并回答“之前能训练 4096 环境，为什么现在 60 环境失败”。完成日志修复、101 项离线回归及代表性地形统计；没有启动 Isaac Sim、重新执行 60 环境检查或运行 PPO。最大已通过的 Source 启动检查规模仍为 32，这不是硬件的并行环境上限。

## 4096 与当前 Source 环境的区别

仓库既有的 Lite3 Flat 训练命令使用 `--num_envs 4096`，例如[长期平地训练方案](../superpowers/plans/2026-08-01-lite3-long-horizon-flat-stability.md)。对应 [Flat 配置](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_env_cfg.py) 使用 `terrain_type="plane"`、`terrain_generator=None`，并关闭两个高度扫描器。常规 Rough 配置引用本机 Isaac Lab 的固定 10×20 地形网格，地图行数不等于机器人数量。这些是配置证据；本轮没有重新核验用户所指的某次历史 4096 训练的完整运行日志或吞吐。

当前 [Source 配置](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py) 则设置 `num_rows=num_envs`、`num_cols=15`：每个环境独占一行完整地形组。原因见[原分配设计](../lite3-source-parallel-terrain-allocation-v0.1-draft-2026-10-27.md)：任意环境异步 reset 时都能切换到预登记的下一个地形类型，不必等待、挪动其他机器人或因占用而重抽。这种保证用了大量预留地图，每行同时只占用一个 tile。

[离线统计脚本](../../scripts/tools/source_terrain_scaling_audit.py)从真实 L1 配置工厂读取参数，以纯 NumPy/trimesh 生成各子类的一个代表实例。每行普通三角形为 **469,844**，分区摩擦三角形为 **144**。

| 环境数 | 地形块数 | 普通三角形，不含外围边界 | 地图尺寸 |
|---:|---:|---:|---:|
| 32 | 480 | 15,035,008 | 384×60 m |
| 60 | 900 | 28,190,640 | 720×60 m |
| 120 | 1,800 | 56,381,280 | 1,440×60 m |
| 4096 | 61,440 | 1,924,481,024 | 49,152×60 m |

32 与 60 的历史 ordinary 计数均比此 tile 总和多 8 个三角形，符合生成器还添加外围边界的路径。4096 行的全部 tile 三角形外推为 **1,925,070,848（约 19.25 亿）**；这是结构性外推，没有生成或运行该规模地图，也不是内存/吞吐预测。

每个 rough 或 slope+rough tile 有 **78,080** 个三角形，普通 slope 为 **19,840**；stairs 每块只有 **36**。粗糙地形按最小空间尺度 0.2 m 的四分之一取 0.05 m 网格，普通坡面为 0.1 m。高度场棱柱把同样密度的三角形铺在顶面和底面；每行底面共有 **230,430** 个三角形，约占全部三角形 **49.0%**。减少底面细分是候选优化方向，不能直接删除底面或放粗接触表面而假定物理语义不变。

## 失败原因能确认到哪一步

[60 环境复测](../lite3-source-g1-60env-recheck-20261028/README.md)明确在 Ujitso/PhysX 碰撞网格 cooking 阶段失败，没有完成 reset 或训练。GPU 每秒采样峰值 2,786 MiB，没有日志证据表明这次是 GPU 显存耗尽；这也不排除 CPU、cooking 进程、缓存或其他资源问题。

两个普通分块各 14,095,324 个三角形，仍然失败。因此，32 环境成功过的 15,035,016 个三角形不能作为其他网格保证成功的“安全阈值”。现有分块只按面序号切割；本机生成器先遍历列，再遍历全部行，分块不保证空间局部性。它也没有减少总三角形数量、最初完整地图的拼接内存或高度扫描器合并地图的开销。

15 个代表 tile 在投射到 60 行地图末行的位置并转为 float32 后，坐标均有限、没有零面积三角形；原 tile 均闭合且绕序一致。这仅是抽样排除明显数值问题，没有重建失败地图的全部随机实例、最终 USD、两个实际 cooking 输入或运行缓存，**尚未确定 PhysX 内部失败的唯一根因**。详细配置、各类计数与边界见 [scaling_audit.json](scaling_audit.json)。

## 已完成的日志修复

[监督器](../../scripts/tools/source_parallel_terrain_startup_supervisor.py)新增以下行为：

- 子进程使用 Python `-u` 和 `PYTHONUNBUFFERED=1`，通过 AppLauncher 的 `--kit_args` 设置 Carbonite `/log/flushStandardStreamOutput=true`。本机 `StandardLogger2.h` 与 `LoggingSettingsUtils.h` 说明该刷新选项默认关闭；这支持缓冲假设，但不是历史日志延迟的实测根因。
- 进程组清理完成后重新扫描完整控制台，包含 SIGTERM 处理或退出时才刷出的错误。
- `stop_reason` / `failure_marker` 保留真实停止触发原因；新增 `collision_cooking_failure` 独立记录碰撞错误、发现阶段和监督器相对时间，`stop_requested_elapsed_s` 记录请求停止的时刻。
- 发现阶段分为 `monitor`、`process_exit`、`post_termination`。最后一种不能被解释成实时 fail-fast 成功。即使子进程返回 0、smoke 报告通过且没有主动停止原因，任何收尾碰撞错误也会使最终判定失败。
- manifest 使用 v2 并记录刷新配置。历史失败产物保持原样，本阶段只做解析复核。

离线回归 **101 passed in 4.59s**。新增覆盖缓冲日志在启动超时、总超时、资源采样失败后刷新，保留原停止原因，大量退出输出中的错误，以及“smoke 通过但收尾碰撞失败”的最终判定。既有地形分块、importer、分配器与 reset 回归同时通过。结果与文件哈希见 [offline_validation.json](offline_validation.json)，逐项结果见 [JUnit](offline_tests.junit.xml)。这些测试没有验证 Carbonite 在真实仿真中的实时输出延迟。

## 后续建议

应先调整地形组织与几何成本，再决定下一次运行检查；单纯提高 env 数或继续沿用失败过的面数阈值，缺少依据。可审查的下一步为：

1. 优先设计保持顶面坐标、三角形和摩擦区域不变的底面简化，以及按 tile/空间组织的碰撞分块；比较三角数、闭合性、边界与法向，之后再做有界 cooking 检查。
2. 面向上千环境，评估可复用地形资产或独立碰撞场景，解耦机器人数量与整套地形复制数量。地形池不能只按平均配比配满：必须证明异步 reset 的最坏占用、碰撞过滤、扫描器目标、边界约束和固定票据顺序都能保持，才能替换 reserved-row 方案。
3. 若局部网格仍失败，单独捕获更细的 Ujitso/cooking 诊断并固定输入对比缓存/构建路径。当前证据不支持直接禁用 Ujitso、延长启动超时或认定某一资源耗尽。

这里给出设计方向，没有改变现有训练分配语义，也没有开始新的仿真阶段；后续实现和运行按原对话的逐步确认约定推进。
