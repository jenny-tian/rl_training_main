# G1 六十环境失败后的离线修复（2026-10-28）

## 范围

本阶段依据用户同意，仅做源码排查、候选修复和离线测试。没有启动 Isaac Sim/PhysX，没有重试 60 环境，也没有运行 120 环境或 PPO。

## 证据与修复

失败日志记录普通地形 `/World/ground/terrain/ordinary/mesh` 的 PhysX `cooking failure` 和 `Unable to create triangle mesh`。32 环境通过记录测得同一 ordinary collider 含 `15,035,016` 个三角形并成功完成首步；60 环境没有完成初始化，因此没有普通网格三角计数和可用于排除其他原因的运行时数据。现有证据支持把单个 cooking 输入规模作为修复目标，但不能确认失败必然由网格三角数上限导致，也不能排除缓存、有效性或资源相关原因。

`SourceTerrainImporter` 现在将超过 32 环境已实测成功规模的普通或分区摩擦网格按原始面顺序平衡切分成多个静态 PhysX 网格。每个 collider 上限为 `15,035,016` 个三角形；未超过上限的无标记网格仍保留原 prim 路径。多块网格使用父 Xform 下的 `part_###` prim；TerrainImporter 会登记每个碰撞 prim，高度扫描仍从 `/World/ground` 合并其后代网格。

新增 [启动监督器](../../scripts/tools/source_parallel_terrain_startup_supervisor.py)，先要求 worktree 与模型子模块干净、核对机器人资产哈希、确认目标 GPU 无计算进程且至少空闲 8 GiB，再启动检查。它按 100 ms 间隔轮询输出，识别地形路径下的 PhysX cooking failure 或 mesh 创建失败后向启动进程组发送 SIGTERM，必要时用 SIGKILL 收尾；即使主进程先退出，也会清理同组子进程。监控异常或键盘中断同样进入清理路径。GPU 资源监控每秒采样，单次查询限时 1 秒，采样失败或超时会停止本次启动；同步采样期间日志轮询会相应延后。输出记录代码哈希、设备快照、资源采样、stop reason、首条匹配日志、耗时和退出码。原 120 秒启动和 180 秒总时限保持不变。监督器必须显式运行才会启动仿真；本阶段没有调用它的运行模式。

## 验证

最终离线回归 **95 项通过（3.56 秒）**：三组新增测试共 26 项，现有地形分配和 reset sampler 回归共 69 项。覆盖分块覆盖/上限/无空块、三角形坐标与绕序不变、摩擦材质选择、prim 登记、PhysX 失败日志检测、主进程退出后拒绝 SIGTERM 的子进程清理、GPU 查询超时/异常停止、trace 读取异常清理，以及启动/总超时分类。测试中的子进程仅为短暂的 Python 测试进程。

使用 `env_isaaclab` 环境，并设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避开已知的自动加载插件 `typeguard` / `typing_extensions` 版本冲突。命令、各测试文件数量及被测文件 SHA-256 见 [验证记录](offline_validation.json)，逐项结果见 [JUnit XML](offline_tests.junit.xml)。`git diff --check` 通过。

验证没有载入 `pxr` 或启动模拟器，因此不证明多 collider 场景能通过真实 PhysX cooking，不证明 RayCaster 和摩擦材质绑定在分块后运行正确，也不证明新的网格上限足以避免 60 环境失败。60 环境状态继续标记为失败，最大通过规模仍为 32。

## 下一步

**后续更新：** 用户已批准并完成一次 60 环境复测，两个普通地形分块仍报 cooking failure，实时检测未捕获，最终由 120 秒超时停止。本候选修复未通过运行验证；详细结果及后续建议见[复测记录](../lite3-source-g1-60env-recheck-20261028/README.md)。以下为离线阶段结束时的原计划，示例输出目录现已使用。

需另行批准一次带新 fail-fast 监督器的 G1 60 环境启动检查。若再次出现碰撞构建失败，应保留诊断并立即停止，不自动重试或放大环境规模；如果成功，也只能据此更新 G1 启动结果，不能宣称完整并行运行资格或训练资格。

批准后从当前 worktree 的干净提交运行如下命令；输出目录必须尚不存在，GPU0 在执行时仍须通过监督器的资源检查。该命令本阶段未执行。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 180 \
  --run-dir /tmp/lite3_g1_60env_recheck_20261028
```
