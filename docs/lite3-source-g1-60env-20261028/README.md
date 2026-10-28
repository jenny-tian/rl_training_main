# G1 六十环境启动检查失败（2026-10-28）

状态：`g1_60env_failed_cooking_and_startup_timeout`。本次 60 环境运行已获用户同意并停止，未自动重试，120 环境未运行；当前最大通过规模仍为 32。

干净提交 `fefa63e810768e49e5ec1ef6dae580227c5299a0`，模型提交 `b452a1f404d37155400242f18e454fed8bcc51b8`。全部 130 个 source 文件与 32 环境运行哈希一致，运行前资产哈希通过核对，运行前后代码哈希一致。GPU0 / RTX 3090，环境和 sampler seed 130001，地形 seed 10001。

## 观察到的失败

- 约 Kit 启动 108.2 秒时，日志先记录 `preloadDataAfterSuccess() failure (FAILURE)`，随后 PhysX 报告 `UjitsoMeshCookingContext: cooking failure for /World/ground/terrain/ordinary/mesh` 和 `Unable to create triangle mesh`。
- 监控脚本在启动计时达到 120 秒后发出 SIGTERM，完整收尾 122.19 秒。进程响应终止后返回 0，但 `stop_reason=startup_timeout`，且没有 `result.json`，因此绝非通过。
- trace 只到 `environment-construction-start`；没有环境构建完成、首次 reset 或首步验证。未取得运行时 origin/guard/scanner 通过证据。
- 碰撞构建失败发生在超时终止之前；该日志在运行后复核时发现。现有监控仅检查超时，未实时识别这种 warning 级别的致命碰撞缺失，应补充失败即停止检测。
- GPU0 每秒采样最大显存 2,786 MiB，仅表示未完成启动时的用量，不代表 60 环境完整资源预算。退出后 GPU0 回到 18 MiB，未发现残留测试进程。
- 未加载策略、未运行 PPO；未修改地形/guard，也未延长超时。

## 原因边界与下一步建议

已定位失败对象为普通地形碰撞网格，但底层原因尚未确定，不能仅凭日志判断为网格大小限制、缓存问题或内存问题，也不能把延长超时当作修复。

源码 `SourceTerrainImporter.import_mesh` 将所有非摩擦标记的三角面合并到一个 ordinary collider；32 环境已实测该网格有约 1504 万三角形。60 环境的实际三角数未完成记录。下一步建议离线检查网格规模/有效性及 cooking 路径，制定或实现有证据支持的修复，同时为诊断补充 cooking 失败立即停止检测；先不重新启动仿真。需用户同意后进行下一阶段。

证据：[失败判定](outcome.json)、[执行摘要](execution.json)、[关键 cooking 日志](cooking_failure_excerpt.log)、[完整控制台](console.log)、[IsaacLab 日志](isaaclab.log)、[trace](trace.log)、[运行清单与哈希](manifest.json)。代码快照在 `/tmp/lite3_g1_60env_20261028_094115/code.tar`，也可从固定提交重建。
