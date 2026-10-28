# G2 六十环境 reset 生命周期复测通过（2026-10-28）

状态：`g2_reserved_row_reset_lifecycle_passed`。

用户批准的一次修复后复测已完成，初始化、首步、9 批直接 reset 和一批自动超时 reset 全部通过。**本次 60 环境 reserved-row reset 生命周期诊断通过**；不是共享地形隔离、运动能力、吞吐或完整并行运行资格通过。没有加载策略或执行 PPO。

## 固定输入与执行

- 运行提交：`91e948f45096fc8f6752efbc5d0d9cc55f6a640c`，工作树干净。
- 模型提交：`b452a1f404d37155400242f18e454fed8bcc51b8`，子模块干净、资产哈希通过监督器检查。
- GPU0 / RTX 3090；运行前 18 MiB、无计算进程。沙箱内驱动查询失败后，经沙箱外复核确认空闲，在沙箱外运行。
- 60 环境，L1 reserved-row 地图，origin grid `(60,15,3)`，12×4 m tile；环境/reset sampler seed 130001，地形 seed 10001。
- 启动/总时限 120/240 秒；一次运行，无超时、重试或扩容。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --reset-lifecycle --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_g2_60env_reset_repair_20261028
```

该目录已存在，不能直接重用；原件继续保留。上次失败与离线修复分别见[首次失败记录](../lite3-source-g2-60env-reset-20261028/README.md)和[离线修复记录](../lite3-source-g2-reset-offline-repair-20261028/README.md)。

## 检查结果

| 项目 | 结果 |
|---|---|
| 初始化、首次 reset、首个控制步 | 通过，首步所有终止项均 false |
| 全刚体 tile guard | 覆盖 17 个运行时刚体 |
| 直接 reset | 9 批、300 次；单个、不连续乱序、轮转 45 个及 24 个环境 |
| 自动 reset | `env.step()` 中恰好重置 `[2,23,58]`，无其他终止 |
| 总分配次数 | 初始 60 + 直接 300 + 自动 3 = 363 |
| 票据与配比 | 编号 0–362 连续，逐项重放一致；3 个完整 120-ticket 块计数正确 |
| 尾块 | 3 张：flat、friction_alternating_regions、slope_cross_slope_right 各 1 |
| 未 reset 环境 | reset 边界前后的 assignment、row/type、origin、root、关节状态、命令、动作、episode counter、sensor 缓存及时间戳检查通过 |
| 被 reset 环境 | 新 origin/root 对齐、有限状态、动作与 episode 清零、接触力历史/计时清零、sensor 失效/刷新检查通过 |
| 高度扫描 | `(60,345,3)` 和 `(60,9,3)`，刷新后命中坐标有限 |
| 运行时观测历史清单 | 空；未启用 observation history，不声称验证非空策略历史 |
| 控制步数 | 2 个零动作控制步，其余直接 reset 不推进物理积分 |
| 地形 USD 网格 / 总三角形 | 1,440 / 14,644,080 |
| 环境构建 | 45.625 秒 |
| 监督器总耗时 | 68.283 秒 |
| CPU 峰值 RSS | 10,524,008 KiB，约 10.04 GiB |
| GPU 每秒采样峰值 | 9,220 MiB，约 9.00 GiB |
| 子进程 / 监督器退出码 | 0 / 0 |
| 源码/诊断/审计文件哈希 | 137 项运行前后及事后核对一致 |
| 退出后 GPU0 | 18 MiB，利用率 0%，无计算进程 |
| 残留测试进程 | 未检出 |

事后独立核对了全部 31 条 journal：初始记录及 10 组 before/comparison/checked。每批 prior 与上一批完整 current 一致；只替换指定 ID 后恰好得到新 current；episode_index 对被 reset 环境递增；实际/预期/checked 票据一致，363 张票据连续，各块位置及 15 类计数与结果 JSON 一致。此核对没有重新执行仿真。

控制台、Kit 和 IsaacLab 日志均无 `[Error]`、碰撞构建失败或 Python traceback。既有 Lite3 足端 visual USD 引用警告仍存在，本次不据此确认视觉资产完整性。GPU3 的无关进程未受影响。

## 结论边界

过期 allocator 引用修复已获得真实运行验证，当前检查脚本成功覆盖多批局部 reset 和受控自动 reset。未 reset 状态不变性只比较 `_reset_idx` 紧邻边界；不要求其他机器人跨物理控制步保持位置不变。

自动 reset 通过诊断 time_out term 和指定 counter 触发；不代表自然跌倒、地形完成及所有终止路径或长回合稳定性已验证。当前无非空 observation history；高度扫描有限性不等于 ray 命中归属正确。事务失败不消耗票据仍依赖既有离线回归，本次没有向真实场景注入写入失败。

共享地形银行、机器人重叠时的碰撞过滤、运动隔离、冻结策略、G4 吞吐预算、checkpoint 调度状态恢复及 PPO 均未验证或启动。初始化更快只是本次观测，不能解释为训练吞吐提升。

## 下一步建议，等待确认

按原计划进入共享地形接入的离线准备：以两个机器人共用同一 tile 为首个运行候选，梳理并实现显式启用的共享 bank 接入、机器人间碰撞过滤、扫描目标限定和 reset 状态隔离；补充离线测试并形成两环境检查方案。首先检查既有 Isaac Lab clone/filter 实现能否满足这些约束，不能仅凭分配器索引认定隔离成立。

下一阶段先完成代码/配置与可审阅方案，再汇报请求两环境真实运行授权；本轮没有开始该阶段，也不直接扩容或训练。

## 证据

[结果](result.json)、[manifest](manifest.json)、[监督器执行](execution.json)、[最终判定](outcome.json)、[trace](trace.log)、[reset journal](result.resets.jsonl)、[控制台](console.log)、[Kit 日志](kit.log)、[IsaacLab 日志](isaaclab.log)、[GPU 采样](gpu_samples.jsonl)、[退出后 GPU](gpu_after.csv)、[退出后计算进程](gpu_compute_after.csv)、[测试进程检查](test_processes_after.json)、[独立票据复核、源码核对和产物哈希](post_run_review.json)。
