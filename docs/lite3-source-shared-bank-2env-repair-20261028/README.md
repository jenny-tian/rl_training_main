# 两环境共享地形复测：接触阳性对照缺失（2026-10-28）

状态：`shared_bank_smoke_failed_contact_positive_control_missing`。

用户批准的一次修复后真实复测已执行并自动停止。**扫描 ID 维度错误未再出现，环境构建、初始 reset、扫描干扰探针和 20 个零动作控制步均完成；但接触 journal 没有任何接触记录，两个机器人的地面接触阳性对照均缺失，整体失败。** 五批局部 reset 未进入。没有重试、改卡、延长预算、扩容或训练。

监督器的原始 `outcome.status` 为通用标签 `startup_check_failed`，本次实际失败发生在 20 步之后，不能解读为环境初始化失败。这里保留原始 JSON。

## 固定输入与命令

- 运行提交：`27441a8e3188cb2e4f8c8f796b9836676cff0481`，启动前工作树干净。
- 模型提交：`b452a1f404d37155400242f18e454fed8bcc51b8`，与 gitlink 一致，子模块干净，6 项模型资产哈希匹配。
- GPU0 / RTX 3090：启动前 18 MiB，无该卡计算进程，可用显存超过 8 GiB。
- 2 环境、1 行共享 bank、15 等权物理列、L1、12×4 m tile；地形 seed 10001，环境与 sampler seed 130196。
- 启动／总时限 120／240 秒，进程组清理宽限 10 秒。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_repair_20261028
```

该目录原件保留，不能重用。本次从 [离线修复版本](../lite3-source-shared-bank-offline-repair-20261028/README.md) 运行，没有修改运行代码或已安装依赖。

## 实际覆盖

| 检查 | 结果 |
|---|---|
| 环境构建与初始 reset | 完成；两张初始票均为 row 0、column 0、flat |
| 共享 bank 与重叠起点 | `(1,15,3)`；两个完整初始 root state 一致，位置均为 `[-4.5,-28.0,0.375]` |
| USD 碰撞组审计 | inverted filter、global ground、自身环境组及允许关系检查通过 |
| 刚体接触报告 API | 两环境共 34 个刚体均具备 API，阈值为零；tile guard 覆盖全部刚体 |
| 两套地面扫描 | 345／9 条射线；hit cache 为 `(2,345,3)`／`(2,9,3)`，ID cache 维度断言通过，全部命中同一静态 ground mesh |
| 扫描干扰探针 | 将机器人 1 抬高 0.8 m 后，机器人 0 的地面扫描保持不变，平地高度检查通过 |
| 零动作控制步 | 20 步完成，每步状态／观测／奖励／扫描有限，无 terminated 或 truncated |
| 诊断仿真时长 | 控制 dt 0.02 秒，共 0.4 秒；physics dt 0.005 秒 |
| 接触 journal | **0 条**；全部 20 步的 `ground_contact_points` 均为 `{}` |
| 地面阳性对照 | **失败**；不能用零条跨环境记录证明没有跨环境接触 |
| 五批局部 reset | 未进入，完成 0 批；本次真实扫描证据不覆盖这些局部 reset |
| 结果文件 | 没有 `result.json`；保留包含 22 条记录的 `result.shared.jsonl` |

失败位置为 `source_shared_terrain_diagnostic.py:217`：

```text
AssertionError: contact-report positive control missing: both robots must touch ground
```

journal 只在回调收到 contact header 后写接触记录，没有独立的回调调用次数。因此现有证据无法区分回调没有执行、回调收到空 header batch，或没有产生可报告的接触。也没有记录足端高度、接触力或逐个物理子步的即时 contact report，不能由 root 高度变化认定已经真实接地。

静态 USD 关系与 API 存在性检查通过，是结构证据；接触检测管线缺少阳性对照，因此本次仍不能资格化共享物理隔离、完整 G3/G4、自动 reset、策略运动或 PPO。

## 自动停止与资源收尾

监督器在运行第 **15.0036 秒**从 console 检测到 `fatal-exception-recorded`，自动设置 `stop_reason=fatal_exception` 并按既定机制清理本次进程组。总耗时 **15.8275 秒**，停止请求到监督过程结束为 **0.8239 秒**，未耗尽 10 秒清理宽限，无人工信号操作。

trace 包含 `environment-close-complete` 和 `simulation-app-close-start`，不包含 `simulation-app-close-complete`。这验证了本次真实异常后的自动识别与进程组收尾，不能称为 Kit 自然关闭流程通过。子进程退出码为 0、监督器为 1；致命标记确保零退出码不会掩盖失败。

GPU 每秒采样共 15 条，峰值 **2,820 MiB**。退出后 GPU0 为 **18 MiB、0%**，无该卡计算进程；监督器 PID 3890155、子进程／进程组 3890196 已不存在，无匹配测试进程或同组成员。GPU2、GPU3 的无关任务 PID 保持不变，未对它们执行操作。未取得最终 CPU 峰值 RSS 报告。

控制台、Kit、Isaac Lab 日志无地形 cooking failure 或 `[Error]` 行；上述 Python traceback 已足以判失败。原有足端 visual USD 引用警告仍存在。

## 事后验证与下一阶段建议

运行前后及归档时核对 **140 项源码／审计文件、14 项固定依赖、6 项模型资产**，均保持一致。运行原件与归档副本逐文件 SHA-256 相同。此阶段只新增运行证据与文档，没有修改运行代码，也没有重复执行此前已通过的 293 项离线回归。

归档 JSON／JSONL 均可解析，README 链接有效，暂存区原始证据哈希匹配。文档及结构化记录的 `git diff --check` 通过；完整检查仅报告原始 `console.log`、`kit.log` 自带的行尾空白，为保持证据逐字节一致而原样保留。

只读查看本机依赖后，可以确定已有两种接触读取接口：订阅 `subscribe_contact_report_events`，以及官方测试在物理步之后使用的即时 `get_contact_report()`。Isaac Lab 的 `ContactSensor` 另通过 tensor contact view 读取力。`SimulationContext` 默认关闭接触处理，而 `ContactSensor` 构造时重新开启；本次没有记录运行时 setting 值，不能直接将该开关认定为根因。查阅文件的路径和哈希另记在 `post_run_review.json`，没有更改已固定依赖清单，也没有运行这些官方仿真测试。

建议下一阶段仅做**接触报告链路的离线排查与修复准备**：

1. 核对本机 GPU 物理步、订阅回调与即时报告的调用顺序和生命周期，补充运行时开关、回调次数、header 数及接触数据数的可观测性。
2. 审阅每个物理子步之后按 collider 归属读取报告的候选方案；用接触点和对端身份验证地面阳性对照与跨环境接触，不能用净力抵消或空报告代替隔离证据。
3. 用离线回归覆盖空报告、回调错误、跨环境事件和缺失阳性对照等失败路径，保持失败即停与既定运行边界；提交可审阅修改后，再单独申请下一次真实运行。

该下一阶段尚未批准，本次没有据此修复代码、扩大检查或再次启动仿真。

## 证据

[manifest](manifest.json)、[execution](execution.json)、[outcome](outcome.json)、[shared journal](result.shared.jsonl)、[trace](trace.log)、[console](console.log)、[Kit 日志](kit.log)、[Isaac Lab 日志](isaaclab.log)、[GPU 采样](gpu_samples.jsonl)、[退出后 GPU](gpu_after.csv)、[退出后计算进程](gpu_compute_after.csv)、[进程检查](test_processes_after.json)、[事后核对与文件哈希](post_run_review.json)。
