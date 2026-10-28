# 共享地形扫描 ID 与异常停止离线修复（2026-10-28）

状态：`shared_bank_offline_repair_passed_runtime_pending`。

用户同意后，在基线提交 `b670e7f` 上完成了两项离线修复。**293 项针对性回归通过；没有启动 Isaac Sim、GPU raycast、策略或训练。** 上次两环境真实检查仍为初始化失败，修复版本等待一次新的运行授权。

## 扫描 ID 维度修复

上次失败发生在环境构建期间：本机 `MultiMeshRayCaster` 用 `(N,B,1)` 缓存 ID，但 `raycast_dynamic_meshes` 返回 `(N,B)`，直接赋值触发 shape mismatch。该路径之前只在共享配置启用 hit ID 时进入。

新增项目内 [SourceGroundRayCaster](../../source/rl_training/rl_training/source_scan/source_ground_ray_caster.py)，仅由共享 bank 配置中的两套高度扫描启用：

- 沿用上游 mesh/ray 初始化和机器人射线姿态计算，公开 `ray_mesh_ids` 仍为 `(N,B,1)`。
- 静态地面更新在写缓存前把 rank-2 ID 加上末维；已符合契约的 rank-3 ID 原样使用。保留 `int16` 与 `-1` 未命中值，不将无效命中改成 ground。
- 先校验选中环境数、射线数、ID dtype 和缓存维度，再写入 hit positions 与 IDs；形状错误不会先覆盖其中一个缓存。
- 对局部或乱序环境更新，仅写选中环境，保留缓存对象与其他环境的数据。
- 此类只支持一个合并的静态 `/World/ground` mesh。初始化时检查每环境引用同一 ground mesh ID、没有动态 mesh view。由于每行是同一个 ID，Warp 的批内行号也适用于重排/部分环境；其他目标或动态 mesh 配置会拒绝。
- shared 诊断新增公开 ray/ID 缓存维度断言，继续检查真实 ground mesh、命中索引和坐标有限性。

该适配器为共享静态地面提供明确的兼容边界，不是通用动态多 mesh 传感器实现。单环境和 reserved-row 配置继续使用原 `MultiMeshRayCaster`。已安装的 Isaac Lab 文件没有修改，也没有修改其模块级 raycast 函数。

## 致命异常与收尾

[监督器](../../scripts/tools/source_parallel_terrain_startup_supervisor.py) 每次轮询同时检查本次 console 和 trace 中的完整标记行：

```text
[parallel-terrain-smoke] fatal-exception-recorded
[parallel-terrain-smoke] simulation-app-close-exception
```

收到任一标记后立即记录 `stop_reason=fatal_exception` 与来源、检测阶段和时间，再按现有机制向本次进程组发送 SIGTERM，清理宽限结束仍存活则发送 SIGKILL。生产轮询间隔仍为 0.1 秒，清理宽限仍为 10 秒，启动/执行预算仍为 120/240 秒。

分段写出的标记、只有 trace 有标记、进程快速零退出和大量尾部日志均有覆盖。若标记直到超时/资源错误后的清理阶段才可见，保留原停止原因，另记 `fatal_exception.detected_phase=post_termination`；即使存在成功结果 JSON，最终判定也必须失败。manifest schema 更新为 v3。

只匹配脚本明确写出的致命标记，不把任意日志里的单词或正常 `simulation-app-close-start` 当成异常。没有标记的非零退出仍按失败处理；没有标记且挂起仍由既定超时保护。这次没有改变脚本的成功判定覆盖要求或扩大运行规模。

## 离线验证与边界

[regression.log](regression.log)：**293 passed in 13.42s**，无跳过项。相对上一阶段 246 项回归，新增/扩展 47 项；运行命令如下。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_scan/test_source_ground_ray_caster.py \
  tests/source_scan/test_shared_terrain_runtime.py \
  tests/source_scan/test_shared_terrain_bank.py \
  tests/source_scan/test_parallel_terrain_allocation.py \
  tests/source_scan/test_source_parallel_reset_lifecycle.py \
  tests/source_scan/test_source_parallel_terrain_startup_supervisor.py \
  tests/source_scan/test_source_terrain_importer_chunking.py \
  tests/source_scan/test_tile_bounds.py \
  tests/source_scan/test_source_mesh_chunking.py
```

关键证据：

| 检查 | 结果 |
|---|---|
| 固定本机上游初始化与更新方法体，替代 mesh/Warp 工作 | 原方法在 345/9 rays、整批/单环境四种组合仍复现 shape mismatch |
| 同一实际上游初始化 + 项目适配器更新 | 345/9 rays、整批/单环境/乱序、rank-2/rank-3 返回值全部通过 |
| 缓存与错误输入 | 保留缓存对象、未更新环境、-1 miss；错误维度/dtype、非 ground 或动态目标均被拒绝 |
| 真实 Source 场景构建代码的离线配置检查 | 只有 shared 模式绑定新传感器，两套扫描均绑定正确 |
| 受控 Python 子进程 | console/trace 致命标记均立即停止；SIGTERM 无效时可强制清理；同组顽固子进程也能清理 |
| 异常边界 | 零退出、大量尾日志、分段标记、清理后才出现的异常均不能被误判成功 |
| 上次真实失败日志的 CPU 子进程重放 | 自动触发 fatal_exception，未等待启动超时 |
| 原分配/reset、tile guard、地形分块与监督器回归 | 通过 |

子进程测试使用缩短到 0.1 秒的清理宽限，断言总耗时小于 3 秒、停止请求后的清理小于 1 秒；生产参数仍是 10 秒。它们验证操作系统进程组的停止机制，没有运行 Kit 的真实关闭流程。后者须由下一阶段复测取得证据。

传感器测试执行了固定本机依赖中的真实初始化方法、原更新方法，以及项目中新更新方法；仍用 CPU tensor 和占位 raycast 替代 Warp/PhysX。测试需要本地固定源码，在源码不存在的机器上会明确跳过；本次环境没有跳过。静态目标约束、cache 维度与局部更新已覆盖，但真实 raycast、碰撞报告和物理隔离仍未通过验证。

[dependency_audit.json](../lite3-source-shared-bank-preparation-20261028/dependency_audit.json) 在原 10 项基础上加入传感器 data、RayCaster、SensorBase 和 Warp ops，共 14 项；核对了公开缓存维度、rank-2 返回值、int16 dtype 和 -1 miss 契约。原有 10 个依赖哈希保持一致，所有依赖均未修改。[validation.json](validation.json) 记录检查与当前修复源码哈希。

## 待批准的一次两环境复测

配置与检查范围沿用 [准备方案](../lite3-source-shared-bank-preparation-20261028/README.md)：GPU0；2 环境共用 1 行地形，15 列、L1、12×4 m tile；seed 130196 的初始两张票均为 flat。固定 pose 重叠、扫描干扰探针、20 个零动作控制步、接触报告阳性对照和五批直接 reset 均须完成。

运行前核对干净修复提交、模型与 14 个依赖哈希，重新确认 GPU0 空闲且可用显存至少 8 GiB。只运行一次；失败、超时、任何致命标记、地形 cooking 错误、意外终止、跨环境接触或缺少地面阳性对照均停止，不自动重试或扩容。

拟执行命令（**本阶段未执行**）：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_repair_20261028
```

启动/执行时限 120/240 秒，另有既定 10 秒进程组清理宽限；使用全新目录，保留 [首次失败证据](../lite3-source-shared-bank-2env-20261028/README.md)。本阶段结束于真实复测前的确认点。共享隔离、完整 G3/G4、扩容和 PPO 仍未批准或资格化。

**后续真实复测（2026-10-28）：** 用户批准后，从干净提交 `27441a8` 执行一次上述两环境复测。环境构建、初始 reset、两套扫描、0.8 m 扫描干扰探针和 20 个零动作控制步完成，扫描 ID 维度错误未再出现。但接触 journal 无任何接触记录，两个地面阳性对照均缺失，五批局部 reset 未进入，整体失败。监督器在第 15.0036 秒识别致命标记并自动收尾，停止请求后 0.8239 秒结束，总耗时 15.8275 秒，GPU 采样峰值 2,820 MiB；退出后 GPU0 18 MiB，无残留测试进程。140 项源码／审计与 14 项依赖哈希一致，未修改运行代码或重试。详见 [复测记录与接触链路排查建议](../lite3-source-shared-bank-2env-repair-20261028/README.md)。
