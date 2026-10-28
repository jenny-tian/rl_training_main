# G2 六十环境异步 reset 检查失败记录（2026-10-28）

状态：`g2_failed_diagnostic_stale_allocator_reference`。

用户批准的一次 G2 运行已执行并停止，没有重试。初始化、首次 reset、首个零动作控制步及两类高度扫描通过；第一次直接 reset `[0]` 的票据比较失败。**G2 未通过，G1 最大启动通过规模仍为 60。** 此阶段只归档与定位，没有修改运行脚本或生产 reset 适配器。

## 固定输入与实际执行

- 干净代码提交 `7918d00638f578128436d459182de7519e460983`。
- 干净机器人模型提交 `b452a1f404d37155400242f18e454fed8bcc51b8`，监督器核对机器人资产哈希。
- GPU0 / RTX 3090，运行前 18 MiB、无计算进程。GPU3 的无关进程未受影响。
- 60 环境、reserved-row L1 地图、环境/reset sampler seed 130001、地形 seed 10001。
- 启动时限 120 秒，总时限 240 秒；沙箱外运行以访问 Isaac Sim/GPU 系统资源。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --reset-lifecycle --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_g2_60env_reset_20261028
```

该目录已经存在，不能重用。原始产物继续保留。

## 结果与资源

| 项目 | 结果 |
|---|---|
| 场景创建 / simulation start 日志计时 | 30.516 / 16.109 秒 |
| 监督器总耗时 | 60.939 秒 |
| 初始化、首次 reset、首个控制步 | 通过 |
| 首步终止标记 | 60 环境全部 false，包括诊断 time_out |
| 两类高度扫描 / 刚体 guard | 有限性检查通过 / 覆盖 17 个刚体 |
| 初始票据记录 | 60 张 |
| 异步 reset | 仅尝试 `[0]`，票据比较失败；完成批次数 0 |
| 自动超时 reset / 363 张票据核对 | 未执行 / 未完成 |
| GPU 每秒采样峰值 | 9,220 MiB，约 9.00 GiB |
| 碰撞构建错误 / 超时 | 均无 |
| 仿真子进程 / 监督器退出码 | 0 / 1 |
| 最终 result.json | 未产生，监督器判定失败 |
| 记录源码运行前后哈希 | 137 项全部一致，事后复核一致 |
| 退出后 GPU0 | 18 MiB，利用率 0%，无计算进程 |
| 残留测试进程 | 未检出 |

子进程在异常后的 Kit 关闭路径仍返回 0，但没有成功结果文件，监督器返回 1。原始 outcome 使用通用字符串 `startup_check_failed`，不表示初始化本身失败；真正失败发生在 G2 第一次异步 reset 的诊断断言。原始日志和结果未改写。

控制台、Kit 日志和 IsaacLab 日志未检出 `[Error]` 或碰撞 cooking failure；控制台包含 Python AssertionError。既有足端 visual USD 引用警告仍存在。本次没有资源耗尽证据。

## 失败原因与证据边界

这是新增诊断脚本的对象生命周期错误：

1. `run_lifecycle()` 在进入循环前保存 `allocator = env._source_parallel_terrain_allocator`。
2. 生产 `assign_source_terrain_for_parallel_reset()` 为避免写入失败消耗票据，先 deepcopy 旧 allocator，在候选对象上分配并写入 origins；全部成功后执行 `env._source_parallel_terrain_allocator = candidate_allocator`。
3. 诊断闭包仍从旧 `allocator.assignments` 读取实际票据，第一次 `[0]` reset 后读到的仍是初始 `assignment_index=0`，而重放预期是 `assignment_index=60`，因此必然触发 `runtime assignment differs from deterministic replay`。

[离线复现脚本](reproduce_stale_allocator.py) 使用真实生产 adapter 和已有 synthetic terrain fixture，不启动仿真。它复现对象替换，并确认旧对象票据为 0、新对象与 oracle 都为 60、其他环境分配不变，见 [诊断 JSON](offline_diagnosis.json)。这证明当前诊断读法错误，不代表本次真实运行已完整验证新 allocator。

运行 traceback 表明 reset 调用和紧邻的 buffer transition 检查已返回，随后在票据比较处失败；但脚本只来得及写入 before-reset journal，未保存这次实际新对象票据和成功快照。不能据此补写成功记录，不能宣称所有异步 reset、自动 reset、观测历史或运动隔离通过。

## 下一步建议，尚未执行

下一阶段只做离线修复，待用户确认：

- 在每次 reset 前后分别从 `env._source_parallel_terrain_allocator` 读取当前对象，更新实际票据及未 reset 环境对比，避免捕获过期引用。
- 在失败前记录实际与预期票据，便于再次定位。
- 加入经过完整诊断 reset hook 的测试，用真实 transactional adapter 替换对象，覆盖第一次及后续多批 reset；现有组件测试没有覆盖这一对象替换交互，是本次遗漏。
- 离线验证后提交推送并汇报，再单独请求同规模、同预算的一次真实复测；不在修复阶段自动启动仿真。

共享地形、扩容、策略、PPO 均未执行。

## 证据

[manifest](manifest.json)、[监督器执行](execution.json)、[原始判定](outcome.json)、[trace](trace.log)、[reset journal](result.resets.jsonl)、[控制台](console.log)、[Kit 日志](kit.log)、[IsaacLab 日志](isaaclab.log)、[GPU 采样](gpu_samples.jsonl)、[退出后 GPU](gpu_after.csv)、[退出后计算进程](gpu_compute_after.csv)、[测试进程检查](test_processes_after.json)、[事后源码核对与产物哈希](post_run_review.json)。
