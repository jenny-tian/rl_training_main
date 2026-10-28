# G1 六十环境分块修复后复测失败（2026-10-28）

状态：`g1_60env_recheck_failed_cooking_and_startup_timeout`。用户已批准的一次 60 环境复测结束：两个普通地形分块均出现 PhysX cooking failure，监督器未实时捕获，最终按 120 秒启动超时终止。没有完成环境构建、首次 reset 或首步验证，最大通过规模仍为 32。未自动重试、扩大到 120 环境或运行 PPO。

## 固定输入与执行

运行使用干净提交 `7a6f94b9a227280a7c264d55d129cfc94918ff15`，模型提交 `b452a1f404d37155400242f18e454fed8bcc51b8`，GPU0 / RTX 3090，60 环境、seed 130001。启动前 GPU0 显存占用 18 MiB、无计算进程；代码和模型工作树干净，机器人资产 SHA-256 与审计一致。监督器记录的 134 个源码、诊断及审计文件在运行前后哈希一致。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 180 \
  --run-dir /tmp/lite3_g1_60env_recheck_20261028
```

该命令已执行，输出目录已存在，不可直接再次运行。原始产物保留在 `/tmp/lite3_g1_60env_recheck_20261028`，本目录包含归档副本。

## 结果与证据

- importer 日志记录普通地形共有 **28,190,648** 个三角形，已导入为 **2** 个分块。按固定版本的平衡算法计算，每块为 14,095,324 个三角形，小于候选上限 15,035,016；这不是最终 USD/PhysX 验收得到的网格计数。
- Kit 日志时间约 **98.017 秒**时，`ordinary/part_000/mesh` 报 cooking failure；约 **106.491 秒**时，`ordinary/part_001/mesh` 也失败，两者随后均报告 `Unable to create triangle mesh`。Kit 的相对计时与监督器从启动子进程开始的计时基准不同。
- 监督器实际记录 `stop_reason=startup_timeout`、`failure_marker=null`，完整收尾 **121.384 秒**。仿真子进程响应终止后返回 0，监督器返回 1；原始退出码不构成通过证据。
- trace 最后停在 `environment-construction-start`；没有 `result.json`，没有 origins、guard、传感器或首步检查通过的运行证据。
- GPU 每秒采样共 114 条，峰值 **2,786 MiB（约 2.72 GiB）**，仅反映未完成启动时的占用。结束后 GPU0 回到 **18 MiB**、无计算进程，未发现残留复测进程。未取得完整环境构建时间和 CPU 峰值结果。
- 已知的 Lite3 足端视觉 USD 引用警告仍存在。

原始证据：[执行摘要](execution.json)、[监督器结果](outcome.json)、[运行清单和哈希](manifest.json)、[trace](trace.log)、[GPU 采样](gpu_samples.jsonl)、[完整控制台](console.log)、[IsaacLab 日志](isaaclab.log)、[碰撞失败日志摘录](cooking_failure_excerpt.log)。

## 监督器诊断缺口

运行结束后，用同一提交的 `find_collision_cooking_failure` 解析完整控制台，可以识别上述错误；实时监督器却没有记录 failure marker。因此本次不能认定“碰撞失败立即停止”已通过运行验证。

可能存在原生日志缓冲或输出可见时机问题，但没有逐次字节到达时间，不能仅凭日志内时间戳确认根因。源码还存在明确的收尾缺口：超时分支退出日志读取循环后才终止子进程，没有在终止后重新扫描最终输出。因此保留原始 `execution.json` / `outcome.json` 不作改写，另以[运行后复核](post_run_review.json)记录碰撞构建失败、实际停止原因及 GPU 清理结果。

## 结论与下一步

本次分块没有解决 60 环境碰撞构建失败；32 环境成功过的三角数不能视为其他网格必然成功的充分条件。当前证据仍不能区分网格有效性、cooking 路径、缓存或资源相关原因。

建议下一阶段先离线定位日志读取/刷新缺口，补充终止后诊断和相应回归，再结合两个失败分块复核网格与 cooking 路径。下一阶段需按用户的逐步确认约定批准；不以延长超时或直接重跑替代诊断，不自动扩大环境规模。
