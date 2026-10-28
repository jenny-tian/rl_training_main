# 两环境共享地形复测：接触采集前 stage 校验失败

日期：2026-10-28。状态：`shared_bank_smoke_failed_pre_capture_stage_guard`。

用户批准的一次 GPU0 两环境复测已完成并自动停止。**环境构建、初始 reset、两套地面扫描及扫描干扰探针完成；接触采集前的 stage 一致性校验失败，20 个控制步、80 次逐物理子步采集和五批局部 reset 均未开始。** 本次没有重试、修改运行代码、扩容或训练。

监督器保留通用结果标签 `startup_check_failed`。本次失败发生在环境初始化之后、接触订阅安装之前，不能解释为环境构建失败，也不能解释为已执行的接触回调保持静默。

## 固定输入与执行

- 运行提交：`e5c29d84cc19de139c566909a5bfcb205f9e8fb6`；启动前工作树干净。
- 模型提交：`b452a1f404d37155400242f18e454fed8bcc51b8`；子模块干净，6 项资产哈希匹配。
- GPU0 / RTX 3090：启动前 18 MiB、0%，无该卡计算进程，可用显存超过 8 GiB。
- 两环境、单行 15 列共享 bank、seed 130196；初始两张票均为 row 0、column 0、flat，两机器人初始 root state 完全相同。
- 启动时限 120 秒、总时限 240 秒，另有 10 秒进程组清理宽限。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_contact_repair_20261028
```

原运行目录保留，不能重用。复测基于[接触报告离线修复](../lite3-source-shared-bank-contact-offline-repair-20261028/README.md)，已安装依赖未修改。

## 实际结果

| 检查 | 结果 |
|---|---|
| 环境构建、初始 reset | 完成；两机器人位置均为 `[-4.5,-28.0,0.375]` |
| USD 碰撞组与报告 API | 静态关系检查通过；34 个刚体具备接触报告 API，阈值为零 |
| 地面扫描 | 345／9 条射线及 mesh-ID 维度检查通过，均命中同一静态地面网格 |
| 扫描干扰探针 | 机器人 1 抬高 0.8 m 后，机器人 0 的地面扫描不变 |
| USD stage ID | `9223001` |
| `IPhysxSimulation.get_attached_stage()` | `0`，触发严格相等校验失败 |
| `/physics/disableContactProcessing` | `false`；本次检查时接触处理已开启 |
| 接触订阅／即时采集 | 订阅尚未安装，poll 与 callback 次数均为 0 |
| 计划中的控制步／物理子步 | 0／0；不代表初始化期间没有物理更新 |
| 地面接触阳性对照、物理隔离 | 未验证 |
| 五批直接 reset | 未进入 |
| 结果文件 | 无 `result.json`；journal 保留 5 条记录及失败 summary |

失败调用链为 `run_shared_diagnostic` → `capture_substeps` 进入阶段 → `ContactCapture.runtime`：

```text
AssertionError: contact runtime stage mismatch or processing disabled/unknown
```

结合 journal 可确定触发条件是 stage 返回值不一致，而不是接触处理开关。**本次不能验证逐子步采集是否恢复接触报告，也不能将尚未安装订阅时的零回调次数称为回调静默。** 两环境足端 body origin 初始高度均约 0.04851 m，足端净力为零，前小腿存在非零净力；这些初始化遥测不提供对端身份，也不是成对接触或隔离证据。

## 自动停止与资源释放

监督器在第 **14.3627 秒**识别 console 致命标记，设置 `stop_reason=fatal_exception`。总耗时 **15.2670 秒**，停止请求到监督结束 **0.9043 秒**，未触发 120／240 秒超时；无需人工信号操作。

子进程退出码为 0，监督器退出码为 1。trace 包含 `environment-close-complete` 和 `simulation-app-close-start`，不含 `simulation-app-close-complete`。因此本次证明异常后的监督器收尾有效，不能认定 Kit 自然关闭流程通过。

14 次 GPU 采样的显存峰值为 **2,820 MiB**；退出后 GPU0 为 **18 MiB、0%**，无该卡计算进程，无匹配测试脚本的残留进程。未记录启动时的 OS PID／进程组 ID，事后检查基于 `/proc` 中脚本参数及 GPU 计算进程列表。未操作其他 GPU 上的任务。

控制台、Kit、Isaac Lab 日志未发现 `[Error]` 或地形 cooking failure；Python traceback 已明确使检查失败。既有足端 visual USD 引用警告仍存在。

## 事后复核及下一阶段范围

**141 项源码／审计文件、22 项固定依赖、6 项模型资产哈希保持一致。** 原运行证据与归档副本逐文件 SHA-256 相同；JSON／JSONL 可解析。原始日志中的行尾空白按原样保留，不清洗证据。本阶段仅增加证据和进展文档，未改运行代码，也未重复此前 334 项离线回归。

只读核对本机源码发现：

1. 绑定仅描述 `get_attached_stage()` 返回“通过 IPhysxSimulation 接口附加的 stage”，未明确规定普通 USD 生命周期下返回 0 的完整语义。
2. `SimulationCfg.create_stage_in_memory` 默认是 `False`；Lab 的 `attach_stage_to_usd_context()` 对非 memory stage 会在显式 PhysX attach 前返回。本次未单独记录该配置的运行时值。
3. 官方 `physicsBase` 测试会先显式附加 memory stage；现有 CPU fixture 固定返回匹配的 stage ID，未覆盖普通 USD 场景路径。
4. 本次 Kit 日志记录为 stage `9223001` 初始化 TimeSampleStorage，环境和扫描也已完成。单凭 attached-stage 返回 0，不能断言没有物理场景，或已经确定此前空报告的根因。

因此建议下一阶段仅离线核对普通 USD／显式附加场景的生命周期与接触接口契约，修正有证据支持的 stage 校验及诊断，并补齐相应回归测试。不能为绕过失败而直接调用 `attach_stage`：绑定注明它会先 detach 旧场景并重新建立物理对象。也不能直接删除校验后宣称接触问题已解决。

该离线修复阶段按用户逐阶段确认的约定等待批准；后续真实运行仍需另行确认。本次未移除校验、未手动 attach、未执行第二次仿真。共享物理隔离与 G3／G4 仍未通过。

## 证据

[manifest](manifest.json)、[execution](execution.json)、[outcome](outcome.json)、[shared journal](result.shared.jsonl)、[trace](trace.log)、[console](console.log)、[Kit 日志](kit.log)、[Isaac Lab 日志](isaaclab.log)、[GPU 采样](gpu_samples.jsonl)、[退出后 GPU](gpu_after.csv)、[退出后计算进程](gpu_compute_after.csv)、[残留进程检查](test_processes_after.json)、[事后核对与文件哈希](post_run_review.json)。
