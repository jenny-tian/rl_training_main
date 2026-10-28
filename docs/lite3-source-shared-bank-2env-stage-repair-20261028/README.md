# 两环境共享地形复测：完整逐子步采集后接触报告仍为空

日期：2026-10-28。状态：`shared_bank_smoke_failed_empty_reports_after_full_substep_capture`。

用户批准的一次 GPU0 两环境复测已执行并自动结束。**stage 生命周期校验通过，20 个零动作控制步和全部 80 次逐物理子步即时采集完成；但报告 header／data 均为空，已安装的接触回调未被调用，地面接触阳性对照缺失，整体仍失败。** 足端 tensor 传感器记录到非零力，不能用它替代带对端身份的接触证据。五批直接 reset 未进入。本次没有重试或修改运行代码。

## 输入与命令

- 运行提交：`5d93b74c090780eb34eeda7d69907b69b16397e5`；工作树干净。
- 模型提交：`b452a1f404d37155400242f18e454fed8bcc51b8`；子模块干净，6 项资产哈希匹配。
- GPU0 / RTX 3090：启动前 18 MiB、0%，无该卡计算进程，可用显存超过 8 GiB。
- 两环境、单行 15 列共享 bank、seed 130196；初始票均为 row 0、column 0、flat，两机器人初始 root state 完全相同。
- 启动时限 120 秒、总时限 240 秒，另有 10 秒进程组清理宽限。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_stage_repair_20261028
```

原目录保留，不能重用。本次基于[stage 离线修复](../lite3-source-shared-bank-stage-offline-repair-20261028/README.md)，没有改变运行预算、环境数、零动作或 reset 方案。

## 实际覆盖

| 检查 | 结果 |
|---|---|
| 环境构建、首次 reset、重叠初始状态 | 完成 |
| USD 碰撞组、34 个刚体报告 API、零报告阈值 | 静态检查通过 |
| 345／9 条地面扫描、抬高机器人 1 的干扰探针 | 完成 |
| stage 身份 | 初始／sim／scene／scene.stage_id／USD context 均为 `9223001` |
| 生命周期检查 | 81 次全部通过，均为普通 USD 模式；物理 view 有效且未替换，timeline 播放，接触处理开启 |
| 显式 attached ID | 始终为 `0`；结合完整生命周期证据正常通过，本次未再发生 stage 误判 |
| 控制步／物理子步 | 20／80；控制 dt 0.02 秒，共 0.4 秒；物理 dt 0.005 秒 |
| 逐子步即时采集 | 连续 sim step 1–80，每控制步 4 次；80 次均为 0 header、0 data |
| 接触回调 | 订阅已安装，调用次数 0；本次可以确认不是“有调用但 header 为空” |
| 终止与数据检查 | 20 步均无 terminated／truncated，状态、观测、奖励及扫描有限 |
| 地面接触阳性对照 | 失败，两个机器人的成对接触证据均缺失 |
| 五批直接 reset | 未进入 |
| 结果 | 无 `result.json`；保留 265 条 journal 记录及 v2 失败 summary |

失败来自 `ContactCapture.finish()`：

```text
AssertionError: contact-report positive control missing: both robots must touch ground
```

stage 修复与采集覆盖已得到这次运行支持；**接触报告恢复、共享物理隔离及完整 G3/G4 仍未通过**。不能以零条跨环境报告证明没有跨环境接触。

## 新获得的足端力证据

| 指标 | 环境 0 | 环境 1 |
|---|---:|---:|
| 首次记录非零足端力的控制步末 | 4 | 4 |
| 第 20 步四足垂直净力之和 | 120.535 N | 121.574 N |
| 全部控制步末的单足力模长峰值 | 55.415 N | 73.994 N |
| 第 20 步足端 body origin 高度 | 约 0.022 m | 约 0.022 m |

这些是每个控制步末的 tensor 传感器输出。它们说明本次确实获得了非零接触力遥测，但没有碰撞对端信息；高度也不是 collider 离地间隙。因此它们不能代替地面阳性报告或物理隔离检查。

## 收尾与复核

监督器在 **15.0122 秒**发现致命标记，总耗时 **15.8765 秒**，停止请求后 **0.8642 秒**结束；没有超时或人工信号。子进程返回 0，监督器返回 1，致命标记正确阻止假成功。trace 包含 environment close complete，不含 simulation app close complete；这里只确认监督器自动收尾，不确认 Kit 自然关闭完成。

15 次 GPU 采样峰值 **2,820 MiB**；退出后 GPU0 为 **18 MiB、0%**，无该卡计算进程或匹配测试脚本的残留进程。尝试记录运行中 PID 时测试已结束，因此未取得本次活跃 PID／进程组 ID；事后检查依据 `/proc` 脚本参数与 GPU 进程列表。未操作其他 GPU 上的任务。

**141 项源码／审计、29 项依赖、6 项资产哈希一致。** 原始运行文件与归档副本逐字节相同，JSON／JSONL 可解析。既有失败归档保留。日志没有地形 cooking failure 或 `[Error]` 行，原有足端 visual USD 引用警告仍在，Python traceback 已明确判失败。本阶段未重复已通过的 362 项离线回归。

## 下一阶段的具体线索

本机 `SimulationManager.set_physics_sim_device()` 在 CUDA 分支设置 `/physics/suppressReadback=True`，启用 GPU dynamics 与 Fabric；绑定还提供运行时 `is_readback_suppressed()`。本次日志仅记录了 `disableContactProcessing=False`，**没有记录 suppressReadback 的实际运行值或其对接触事件的影响**，因此 GPU 回读抑制目前是待验证线索，不是已确认根因。

ContactSensor 通过 `RigidContactView.get_net_contact_forces()` 读取力；同一 tensor API 还提供按对端过滤的接触力矩阵和接触数据。下一阶段应离线核对 GPU 回读模式与事件报告的支持范围，再选择有依据的报告修复，或保留当前 GPU 配置的成对 tensor 接触证据方案。任何替代方案仍须提供地面阳性对照、跨环境对端检查及容量／溢出验证，不能退化为只看净力。

本阶段未切换 GPU 回读开关、未改成 CPU、未新建 tensor contact view、未再次仿真。下一阶段离线修复按用户逐阶段确认的约定等待批准，后续真实运行另行确认。所查源码与哈希见事后记录。

## 证据

[manifest](manifest.json)、[execution](execution.json)、[outcome](outcome.json)、[接触与遥测 journal](result.shared.jsonl)、[trace](trace.log)、[console](console.log)、[Kit 日志](kit.log)、[Isaac Lab 日志](isaaclab.log)、[GPU 采样](gpu_samples.jsonl)、[退出后 GPU](gpu_after.csv)、[退出后计算进程](gpu_compute_after.csv)、[进程检查](test_processes_after.json)、[事后核对与哈希](post_run_review.json)。
