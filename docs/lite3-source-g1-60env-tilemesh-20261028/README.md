# G1 六十环境底面精简与 tile 分块后启动通过（2026-10-28）

状态：`g1_60env_tilemesh_startup_passed_120env_pending`。

用户批准的一次 60 环境启动复测已完成。新布局完成环境构建、首次 reset、一个零动作控制步和两类高度扫描；监督器与仿真进程均返回 0，未触发超时或碰撞构建错误。Source 启动检查的最大已通过规模更新为 **60**。本次使用 reserved-row 地图，没有启用共享地形原型、加载策略或运行 PPO。

## 固定输入与执行

- 运行提交：`e15e8aa99bf45f58172c1450818f7763d6c0a020`；代码工作树干净。
- 模型提交：`b452a1f404d37155400242f18e454fed8bcc51b8`；子模块干净，六项机器人资产 SHA-256 与既有审计一致。
- GPU0 / RTX 3090；启动前占用 18 MiB、无计算进程。沙箱内驱动访问失败后，经沙箱外检查确认设备正常，仿真在获准的沙箱外执行。
- 环境数 60；环境/reset sampler seed 130001；地形 seed 10001；L1，60×15 origin grid，每 tile 12×4 m。
- 监督器记录的 136 个源码、启动脚本和资产审计文件在运行前后哈希一致，事后再次核对通过。

执行命令：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 180 \
  --run-dir /tmp/lite3_g1_60env_tilemesh_20261028
```

上述命令已执行一次，输出目录已存在，不可直接重用。本目录归档原始结果与日志，运行目录原件继续保留。

## 检查结果

| 项目 | 实测结果 |
|---|---|
| origin grid / tile | `(60, 15, 3)` / 12×4 m |
| 初始 row、origin 与 robot root | 60 个独占 row，位置对齐，无重复出生位置 |
| tile guard | 覆盖全部 17 个运行时刚体 |
| 首个零动作控制步 | 完成；边界、非法接触、姿态终止标记对所有环境均为 false |
| 状态、观测、回报 | 检查通过，值有限 |
| 两类 height scanner | `(60, 345, 3)` 和 `(60, 9, 3)`，命中坐标有限 |
| USD 地形网格数 | 1,440 |
| 地形三角形总数 / 最大单块 | 14,644,080 / 40,320 |
| 环境构建 | 84.737 秒 |
| 场景创建 / simulation start 日志计时 | 32.424 秒 / 51.931 秒 |
| 监督器总耗时 | 105.061 秒 |
| CPU 峰值 RSS | 10,997,800 KiB，约 10.49 GiB |
| GPU 每秒采样峰值 | 9,225 MiB，约 9.01 GiB；共 100 个样本 |
| 停止原因 / 碰撞失败诊断 | 均为 null |
| 子进程 / 监督器退出码 | 0 / 0 |

实际 USD 的 1,440 个路径及各网格三角形数与[完整地图离线预测](../lite3-source-mesh-optimization-20261028/mesh_audit_60env.json)逐项一致。首步通过且控制台、Kit 原生日志及 IsaacLab 日志均未检出 `[Error]`、`cooking failure for` 或 `Unable to create triangle mesh for`。这表明此前阻塞 60 环境启动的碰撞构建失败，在本固定版本与本次运行中未复现。

既有 Lite3 足端 visual USD 引用警告仍存在；另有本机环境的扩展弃用、IOMMU/peer-to-peer 和配置提醒。未据此确认视觉资产完整性。本次没有触发碰撞失败，因此不能单独验证监督器对真实 PhysX 失败的实时停止延迟。

退出后 GPU0 占用 32 MiB、利用率 0%，没有 GPU0 计算进程；沙箱外进程检查未检出本次监督器或启动仿真脚本。trace 最后记录 `simulation-app-close-start`，随后进程返回 0；收尾结论依据监督器退出状态及 GPU/进程检查，不依赖缺失的后续 trace 标记。

## 结论与边界

**60 环境 G1 初始化与首步检查通过。** 三角形总量约减半、按完整 tile 分块的实现已获得一次真实启动证据；不能据此把任一固定三角形预算解释成 PhysX 的通用成功阈值。

本次没有验证完整地形上的行走表现、各摩擦区域的有效接触系数、异步 reset 的跨环境不变性、运动隔离或稳态吞吐。GPU 峰值是离散采样值，不能按本次占用直接保证 120 或 4096 环境；CPU/初始化时长也不能代替训练吞吐。共享 bank 原型仍未接入运行，120 环境、G2–G4 与 PPO 尚未执行或获本轮授权。

## 证据

[结果 JSON](result.json)、[监督器执行](execution.json)、[最终判定](outcome.json)、[清单与源码哈希](manifest.json)、[trace](trace.log)、[GPU 采样](gpu_samples.jsonl)、[控制台](console.log)、[Kit 原生日志](kit.log)、[IsaacLab 日志](isaaclab.log)、[事后复核及产物哈希](post_run_review.json)、[依赖版本](dependencies.json)、[退出后 GPU](gpu_after.csv)、[退出后计算进程](gpu_compute_after.csv)、[测试进程检查](test_processes_after.json)。
