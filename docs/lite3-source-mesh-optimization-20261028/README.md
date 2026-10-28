# Source 地形底面、碰撞分块与复用原型（2026-10-28）

状态：`offline_geometry_optimized_shared_bank_prototype_runtime_pending`。

本阶段依据用户对[扩展诊断](../lite3-source-logging-scaling-audit-20261028/README.md)后续优化的同意，完成底面精简、按 tile 导入碰撞网格、共享地形分配器离线原型和相应验证。没有启动 Isaac Sim、PhysX、GPU 训练或新一轮 G1 检查。历史 Source 启动检查最大通过规模仍为 32；新拓扑和新 collider 布局尚未运行验证。

## 已接入 Source 配置的几何优化

高度场棱柱的顶面网格保持原样；底面用保留全部边界顶点的扇形三角网格封闭，替代复制整张顶面网格。侧壁继续使用原来的三角形，底面与侧壁共用完全一致的边界顶点，不引入未连接的边。粗糙地形和坡面细节没有降采样，平地、台阶和摩擦区域生成方式未改。

| 每个 L1 tile | 旧三角形 | 新三角形 | 旧底面三角形 | 新底面三角形 |
|---|---:|---:|---:|---:|
| Multi-scale Rough / Slope+Rough | 78,080 | 40,320 | 38,400 | 640 |
| 普通 Slope | 19,840 | 10,560 | 9,600 | 320 |
| Flat | 12 | 12 | 2 | 2 |
| Stairs | 36 | 36 | 2 | 2 |
| Friction，每种布局 | 48 | 48 | 8 | 8 |

[SourceTerrainGenerator](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_generator.py)沿用本机 Isaac Lab 的生成顺序、随机数、平移与 origins 计算，仅记录每个 tile 在合并网格中的面范围，并跳过零宽外围边界。离线执行原始 border 代码确认，当前零宽配置最后遗留的 8 个面全部面积为零；没有删除有效外围表面。

[Importer](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_importer.py)按“材质 → tile”选择原始三角形；完整 tile 的闭合面不再随巨型全图的面序号被截断。摩擦区域仍各自绑定对应材质。每块 100,000 个三角形是代码中的构建预算，不是宣称 PhysX 的安全上限；当前 L1 最大 tile 为 40,320，未触发 tile 内的面数回退切分。缺失或不连续的并行 tile 面范围会在创建 prim 前被拒绝。

提取小块时，只对该块引用的顶点重新编号，避免为每个小块创建全图大小的顶点映射。全图仍会先被生成器拼接，高度扫描仍合并 `/World/ground` 下的地形；本阶段不宣称消除了这些全图开销。

## 完整 60 行地图的离线验证

使用实际 L1 配置、地形 seed 10001 和本机 Isaac Lab 的生成循环，在 CPU 上生成 **60×15=900 个 tile** 并经过真实 Source importer 的 Python 分块方法。只替换了缓存哈希、配置记录包装和 USD/物理材质写入；没有启动模拟器。所读取依赖的 SHA-256 记录在 [mesh_audit_60env.json](mesh_audit_60env.json)。

| 项目 | 旧布局 | 新布局的 CPU 实测 |
|---|---:|---:|
| 普通三角形 | 28,190,648，旧运行日志 | 14,635,440 |
| 含摩擦区域的全部三角形 | 28,199,288，旧 ordinary 日志加固定摩擦面数 | 14,644,080 |
| 最大普通分块 | 14,095,324，两个失败分块各一块 | 40,320 |
| 全部碰撞块 | 原来按材质全图分组 | 1,440 |

总三角形减少 **48.07%**。新地图的 1,440 个输出块均通过闭合性、绕序、正体积、有限坐标、非零面积、材质标记移除及对应 tile 范围检查；所有输入面都有输出，prim 路径不重复。1,440 个块对应每行 12 个普通 tile，以及 3 种摩擦布局各 4 个区域。

本次 CPU 建图 17.89 秒，含逐块几何审计共 39.24 秒，审计进程峰值 RSS 2,450,204 KiB。它们包含离线检查开销，不能与旧仿真启动时间或 GPU 显存直接比较，也不代表训练吞吐。大量小 collider 对 PhysX 启动和 RayCaster 合并的影响仍需实测。

## 地形复用的离线原型

[SharedBankTerrainAllocator](../../source/rl_training/rl_training/source_scan/shared_terrain_bank.py)复用原 `ReservedRowTerrainAllocator` 的票据队列，仅把每个环境映射到 `env_id % bank_rows` 行。相同 seed、环境数和 reset 批次下，地形类别、全局分配编号、每环境回合编号和票据顺序与原方案一致，未 reset 的环境保持不动。测试覆盖 1、60、120 和 **4096** 环境；这是调度测试，没有构建 4096 个物理机器人。

| 地形库行数 | tile 数 | 优化后的 L1 三角形 | 每类几何变体数 |
|---:|---:|---:|---:|
| 1 | 15 | 244,068 | 1 |
| 4 | 60 | 976,272 | 4 |
| 8 | 120 | 1,952,544 | 8 |
| 16 | 240 | 3,905,088 | 16 |

地图大小可与机器人数量解耦，但不同机器人会占用同一物理 tile/出生点。**原型未接入运行配置**。启用前必须验证跨环境机器人碰撞过滤、共享静态地形碰撞、传感器目标及异步 reset 隔离；原有“不同环境必须有不同 row/origin”的检查也需要以经过验证的新隔离契约替换，不能直接删除断言后运行。

几何多样性会从“每个环境一套”改为有限变体库，这是显式分布选择，不等价于原地图；本轮未批准或锁定具体库大小。原型没有提供训练 checkpoint 的调度状态恢复接口，不是可直接续训的 runner。

另外，针对全部 15 种子类构造了合法的异步 reset 序列：只让非目标类回合结束，保留目标类回合，最终所有 60 个环境都能同时占用同一类。这说明独占 tile 池若只按平均配比缩小，无法保证任意回合时长下的可用性。共享 bank 原型通过复用物理地形解决容量问题，所需的机器人隔离保证必须另行验证。

## 回归与证据

主回归 **280 passed in 14.64s**，包含几何、历史表面指纹、分块、生成器、共享 bank、原分配器/reset 与监督器。随后补强原始零宽边界验证，生成器的两个检查再通过：**2 passed in 3.51s**。命令、覆盖数量和文件哈希见 [offline_validation.json](offline_validation.json)。[主 JUnit](offline_tests.junit.xml)、[边界补查 JUnit](border_tests.junit.xml)、[新规模统计](scaling_audit.json)。

45 个固定对照覆盖 15 子类 × L1/L2/L3。参考来自提交 `368bb0421593cef02e5851183650d3e2ed5ee39a` 的原始生成器，基准源 SHA 与配置保存在[历史表面指纹](../../tests/source_scan/fixtures/source_mesh_surfaces_368bb04.json)。对照逐字节比较所有非底面三角形的坐标、顺序和绕序，且检查总面积、体积、出生点、几何元数据与摩擦区域。它证明这些候选实例的表面不变；不证明新碰撞布局的运行接触行为已经验证。

## 准备好的下一步

建议先运行一次 **G1 60 环境启动复测**，验证底面精简与完整 tile 碰撞块能否解决 cooking 失败。继续使用 reserved-row 分配方式与原始 120 秒启动、180 秒总时限；遇到碰撞构建失败或超时即停止，保留收尾诊断，不自动扩大规模或进入 PPO。共享 bank 的运行接入是另外一阶段。

从本 worktree 的干净提交执行，运行目录必须不存在：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 180 \
  --run-dir /tmp/lite3_g1_60env_tilemesh_20261028
```

本命令尚未执行；依照原对话的逐步确认约定，下一次真实仿真检查需用户确认。
