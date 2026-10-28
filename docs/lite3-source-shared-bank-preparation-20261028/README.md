# Source 共享地形接入离线准备（2026-10-28）

状态：`shared_bank_offline_prepared_runtime_pending`。

用户恢复会话 `01a18161-9d0c-7a52-8c3e-1f26767efe0c` 并同意上次提出的“共享地形接入离线准备”。本阶段在原工作树、基线提交 `076db47` 上完成配置、适配器、两环境诊断与离线检查。**没有启动 Isaac Sim、物理积分、冻结策略或 PPO。** 下一次两环境真实运行仍按原会话逐步确认约定等待批准。

## 实现与取舍

- `source_shared_bank_rows=None` 为默认值，保留单环境与 reserved-row 配置。共享模式必须同时显式提供 parallel sampler、环境数、seed 与 `1..num_envs` 的 bank 行数。环境 `i` 使用 `i % bank_rows` 行；列继续由原有 120-ticket 配比决定。
- Importer 在启动时建立独立的 `(num_envs,3)` origin buffer，允许数值相同，但不会把环境 buffer 与地形 bank 或另一环境共享存储。reset 继续通过既有 curriculum hook，在根状态 reset 前事务式更新选中环境的 row/column/origin；写入失败回滚且不消费票据。
- 候选使用 `replicate_physics=False`、`clone_in_fabric=False`、`filter_collisions=True`，触发已安装 Isaac Lab 的显式 USD collision-group 路径。机器人为环境内 `collision_group=0`；地形 `/World/ground` 为公共 `collision_group=-1`。
- 每次非空共享 reset 在分配前读取真实 USD：全局 inverted filter 必须启用；`group0..groupN` 的 includes、allowed groups、expansion rule 必须符合 Cloner 结构；每组只允许自身和地形，公共组不得混入机器人。缺失、额外或错误的组、跨环境允许关系、排除机器人等均报错。此审计是结构检查，不能替代 PhysX 接触实测。
- 两套 `MultiMeshRayCaster` 只合并静态 `/World/ground` 下的地形 mesh，启用 hit mesh index。诊断进一步核对运行时 Warp mesh ID、索引与命中高度，避免仅以“输出有限”判定扫描隔离。

共享 bank 的几何多样性是每类 `bank_rows` 个版本，不再是一环境一份几何。行数会影响原有 difficulty 派生几何 seed；L1 几何参数保持原值。该候选没有通过训练分布、性能或恢复协议资格检查。显式 USD 组和逐次审计也可能增加开销，不能将它称为高吞吐最终实现。

## 本地依赖审计

查看了本机真实实现而非只查看分配器索引：

| 文件 / 机制 | 核对结果 |
|---|---|
| Isaac Lab `InteractiveScene` | 非 replication 且启用过滤时，构建完场景调用 `filter_collisions`；`collision_group=-1` 的资产进入 global paths |
| Isaac Sim `Cloner.filter_collisions` | 创建 `/World/collisions/global_group`、`group0..groupN`；使用 `physxScene:invertCollisionGroupFilter`，关系表示允许接触的组 |
| `MultiMeshRayCaster` | 单一目标根视为全局 mesh；合并地形子网格并为每环境引用；hit mesh index 是本地目标索引 |
| `activate_contact_sensors` | 每个刚体应用 `PhysxContactReportAPI`，默认阈值为零 |
| PhysX contact-report 示例 | 支持 `subscribe_contact_report_events` 与 `PhysicsSchemaTools.intToSdfPath` 读取 collider 归属 |

[dependency_audit.json](dependency_audit.json) 固定 10 个本地依赖文件的路径与 SHA-256。共享监督器在启动前验证这些哈希，写入运行 manifest，并在运行后再次核对；源码变化需要重新审阅。它们是当前安装版本的离线证据，不是动态能力保证。

## 离线验证

**246 项通过**，见 [regression.log](regression.log)。覆盖共享与原有分配器、事务适配器、G2 reset 生命周期、碰撞分块、tile guard、启动监督器及新增共享诊断。

10 个变更 Python 文件语法编译、`git diff --check` 与 10 个依赖文件哈希复核通过；监督器 `--help` 可正常解析共享参数。检查摘要与源码哈希见 [validation.json](validation.json)。

新增 53 项检查包括：

- 真实配置的场景构建部分保持默认关闭，拒绝非法/隐式启用；真实 importer origin 方法正确映射较小 bank。
- 多规模异步分配与原调度器逐票一致；共享 origin 不串写；部分写入后异常仍回滚并保留下一张票。
- 错误过滤配置和伪造 USD 关系在消费票据前被拒绝，包括另一机器人混入全局组或扫描目标。
- 扫描真实目标 ID、缺失命中、非有限坐标与错误平地高度的故障检测；接触 collider 路径归属分类。
- 完整两环境 reset 诊断使用真实事务适配器、合成 managers 与 USD 接口，连续经历五次 allocator 替换，核对八张票；错误根状态、未 reset 历史修改与旧 allocator 引用均被捕获，比较记录保留。
- 完成报告缺少必要覆盖或没有两个机器人的地面接触阳性记录时，不能判为通过。

验证命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_scan/test_shared_terrain_runtime.py \
  tests/source_scan/test_shared_terrain_bank.py \
  tests/source_scan/test_parallel_terrain_allocation.py \
  tests/source_scan/test_source_parallel_reset_lifecycle.py \
  tests/source_scan/test_source_parallel_terrain_startup_supervisor.py \
  tests/source_scan/test_source_terrain_importer_chunking.py \
  tests/source_scan/test_tile_bounds.py \
  tests/source_scan/test_source_mesh_chunking.py
```

边界：配置测试用记录型配置替代 Isaac 类型，仅执行真实场景构建代码；importer 测试替代 simulator 基类；USD/manager 测试使用合成接口。未运行 Kit、真实 USD 场景、PhysX contact callback 或 Warp raycast。针对 callback 的运行覆盖必须由下一阶段取得；若本机 callback 不提供所需数据，按失败停止，不能把没有报告解释为没有碰撞。

## 待批准的一次两环境真实检查

固定候选：`num_envs=2`、`bank_rows=1`、L1、15 等权物理列、12×4 m tile、地形 seed 10001、环境与 sampler seed **130196**。该 seed 在保持原票据表/洗牌逻辑的情况下，前两张恰好均为 flat，所以机器人从同一个 tile、同一个 root pose 起步。离线票据预览见 [schedule_preview.json](schedule_preview.json)。未人为改写票据或重抽分配。

步骤：

1. 核对干净代码提交、机器人子模块/资产与本地依赖哈希；重新检查 GPU0 空闲且可用显存至少 8 GiB。任一条件不满足即停止。
2. 构建 `(1,15,3)` 地形 bank 和两个机器人。初始 reset 后核对分配、origin/root 重叠、全部刚体 tile guard、USD collision groups、每个刚体的零阈值 contact-report API、运行时两套扫描的 ground mesh ID 和平地高度。
3. 不推进物理，在固定机器人 0 位置时将机器人 1 抬高 0.8 m，强制刷新扫描。机器人 0 的地面命中必须不变；随后恢复机器人 1 的 root pose。
4. 订阅 PhysX 接触事件，执行 **20 个零动作控制步**。按 collider 路径识别跨环境接触，任何一例立即失败；两个机器人都必须出现带接触点的地面报告，作为检测管线的阳性对照。每步检查状态、观测、ray hit 与终止标记。
5. 执行五批直接 reset：`[0]`、`[1]`、`[1,0]`、`[0]`、`[1]`。比较 `_reset_idx` 紧邻边界，检查未 reset 环境的分配/物理/命令/动作/episode/sensor/history buffers 不变，选中状态清理、新 root/origin 一致；只 forward 刷新扫描，不继续物理积分。连同初始化共 8 张票、6 次直接 reset。
6. 保存逐步 JSONL、结果、控制台、trace、源文件与依赖哈希、每秒 GPU 采样；关闭环境后复核日志、GPU 与残留进程，再归档。

拟执行命令（**本阶段未执行**）：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_20261028
```

监督器只允许共享诊断使用上述两环境和 seed，且与 `--reset-lifecycle` 互斥。一次运行，启动/执行时限 120/240 秒，沿用最多 10 秒进程组清理宽限；目录必须全新。失败、超时、cooking 错误、报告缺失、异常终止、跨环境接触或无地面阳性对照，均保留证据并停止，不自动重试、改卡、延长预算或扩大环境数。

## 判定边界

即便下一步通过，也只支持该版本与配置下的“两环境 shared-bank 短时重叠/扫描/直接 reset smoke”。它不覆盖所有地形上的机器人重叠、自动终止 reset、非空策略历史的真实运行、单边动作激励/冻结策略运动隔离、G3 完整动态越界检查、G4 吞吐预算、120 环境、checkpoint sampler 恢复或 PPO。

既有 60 环境 reserved-row G2 通过记录继续保留；本阶段不把它扩展解释为共享 bank 已通过。下一阶段真实运行须经用户确认。

**后续运行结果（2026-10-28）：** 用户批准后，从干净提交 `87b5c89` 执行了一次上述两环境检查。在观测管理器构建阶段，启用 mesh hit ID 的本机 `MultiMeshRayCaster` 出现 `(2,345)` 到 `(2,345,1)` 的写入维度错误，未进入初始 reset 或共享诊断。异常关闭未及时结束，经核对进程身份后发送 SIGTERM 收尾，监督器总耗时 114.317 秒、显存采样峰值 2,818 MiB；退出后 GPU0 18 MiB、无残留测试进程。没有自动重试或修复运行代码。详见 [失败记录与下一阶段离线修复建议](../lite3-source-shared-bank-2env-20261028/README.md)。

**后续离线修复（2026-10-28）：** 经用户同意，共享配置已改用项目内静态地面 ray caster 适配 mesh-ID 维度，监督器已增加致命标记自动停止。293 项离线回归通过，尚未真实复测。依赖清单在原 10 项上扩充为 14 项，原安装文件未改动。修复与新目录复测命令见 [离线修复记录](../lite3-source-shared-bank-offline-repair-20261028/README.md)。
