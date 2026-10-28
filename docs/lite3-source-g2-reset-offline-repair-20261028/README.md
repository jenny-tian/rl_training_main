# G2 reset 诊断过期引用的离线修复（2026-10-28）

状态：`offline_repair_passed_runtime_retry_pending_user_approval`。

用户同意上一阶段提出的离线修复后，已修复诊断脚本并完成回归。本阶段没有启动 Isaac Sim、调用 GPU、重跑 G2、加载策略或执行 PPO。上次真实 G2 仍为失败，最大 G1 启动通过规模仍为 60。

## 改动与原因

[上一轮真实运行](../lite3-source-g2-60env-reset-20261028/README.md)在首次 `[0]` reset 后失败。生产适配器在事务成功后替换 allocator 对象，诊断闭包仍读取旧对象。

`source_parallel_reset_lifecycle.py` 现在在每批 reset 前读取当前 assignments 作为 prior，reset 返回后再从环境读取新的 assignments。选中环境票据与 oracle 比较、未 reset 环境分配不变性检查都使用这次新快照，不再跨批保存 allocator 引用。生产适配器及地图配置没有改动。

日志在断言前追加并刷新 `assignment_comparison`，包含实际票据、预期票据和全部当前分配；`before_reset` 包含全部 prior 分配。reset 调用或后续断言异常时追加 `reset_failed`（类型、消息、IDs、模式）并重新抛出，外层 finally 恢复原 reset 方法。若 reset 调用本身未返回，就只记录 before/失败事件，不伪造有效的 reset 后结果。

## 验证

**141 项针对性回归通过**，包括新增 15 项完整流程测试：

- 两个 seed 下运行完整 `run_lifecycle()`，含 9 批直接 reset 和一批模拟 `env.step()` 超时 reset。环境、传感器与 manager 为 CPU synthetic fixtures，但每批地形分配均调用真实 `assign_source_terrain_for_parallel_reset()`，实际发生 allocator 对象替换；覆盖 363 张票据、3 个完整块及尾块。
- 在第 1、2、10 批分别注入选中票据错误、未 reset 环境票据污染、未 reset root 状态污染、reset 函数异常，共 12 个故障场景。要求失败即停止、保留票据证据及失败事件、恢复原 reset 方法，且不多跑后续批次。
- 用另一个文件句柄在 transition 检查被调用之前读取 journal，证明比较证据已 flush，而非等函数退出才写入。
- synthetic step 在 reset 边界之前推进所有 root 位置，确认正常移动不会被当作 reset 污染；非空 synthetic observation history 的清理检查也经过完整 hook。

[版本对照脚本](compare_diagnostic_versions.py)在同一 fixture 上比较失败提交 `7918d00` 与当前诊断：旧版第 1 批复现 `runtime assignment differs from deterministic replay`，修复版完成 10 批/363 张票据。见 [对照结果](version_comparison.json)。这验证了新测试能捕捉原错误，而非只检验新实现自己的输出。

回归命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_scan/test_source_parallel_reset_lifecycle.py \
  tests/source_scan/test_parallel_terrain_allocation.py \
  tests/source_scan/test_source_parallel_terrain_startup_supervisor.py \
  tests/source_scan/test_source_terrain_reset_sampler.py
```

[回归日志](regression.log)：141 passed in 4.73s。Python 编译与 `git diff --check` 通过。关闭 pytest 第三方插件自动加载沿用之前的环境设置，避免无关 typeguard/typing_extensions 版本不兼容。本次没有声称整个仓库测试通过；已记录的缺失 runtime_progress 模块及其他测试隔离问题未在本修复中处理。

这些测试不包含 PhysX、真实传感器或 Isaac Lab manager 的运行，不能代替 G2 真实复测。自动 reset 的 fixture 只模拟计时和调用路径；运行资格、自然终止、运动隔离与 checkpoint 恢复均未由此获得。

## 待用户确认的下一步

执行一次 60 环境 G2 复测，GPU0、seed 130001、启动 120 秒/总计 240 秒，维持上一轮预算。启动前重新核对代码/模型干净、模型资产哈希、GPU 空闲及至少 8 GiB 可用显存。失败即停止，不自动重试、增加预算或扩容。

待批准命令（本阶段未执行）：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --reset-lifecycle --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_g2_60env_reset_repair_20261028
```

运行目录必须尚不存在；运行 manifest 记录届时干净提交和代码哈希。完成后检查全部 363 张票据、10 批 journal、未 reset 环境状态、传感器与自动 reset 结果，核对控制台/Kit 日志和退出后的 GPU/进程清理，并先汇报结果再请求下一阶段授权。
