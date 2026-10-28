# 两环境共享地形检查：初始化失败（2026-10-28）

状态：`shared_bank_initialization_failed_ray_mesh_id_shape`。

用户批准的一次两环境真实检查已执行并停止。新增 `update_mesh_ids=True` 启用了本机 `MultiMeshRayCaster` 的不兼容路径：初始化分配三维 ID buffer，更新方法直接写入二维 raycast ID，触发 shape mismatch。**环境构建没有完成，未进入初始 reset、重叠碰撞、扫描干扰探针或五批局部 reset 检查。没有自动重试、扩容或训练。**

## 固定输入

- 代码提交：`87b5c89fadedf2eb404bf116139f9a86f6487f6f`；启动前工作树干净。
- 模型提交：`b452a1f404d37155400242f18e454fed8bcc51b8`；子模块干净、资产哈希通过监督器验证。
- GPU0 / RTX 3090；运行前 18 MiB，无该卡计算进程。
- 配置候选：2 环境、共享 bank 1 行、15 等权物理列、L1、12×4 m tile、地形 seed 10001、环境与 reset sampler seed 130196。没有得到初始化后的 origin/root 断言结果。
- 既定启动/执行时限 120/240 秒，进程组清理宽限 10 秒。实际未触发这两个超时阈值。

本次实际命令：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_20261028
```

该目录已经存在，原件保留，不能重用。准备方案见 [上一阶段记录](../lite3-source-shared-bank-preparation-20261028/README.md)。

## 失败位置与运行边界

`gym.make()` 构建环境时，`ObservationManager._prepare_terms()` 查询 `height_scan` 维度，触发 ray caster 的首次缓冲更新。本机依赖文件：

`/home/tjx/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster.py`

- 第 344 行开始的 `_initialize_rays_impl()` 将 `ray_mesh_ids` 分配为 `(num_envs, num_rays, 1)`。
- 第 398 行直接执行 `self._data.ray_mesh_ids[env_ids] = mesh_ids`；本次 `mesh_ids` 形状为 `(2,345)`。

控制台错误：

```text
RuntimeError: shape mismatch: value tensor of shape [2, 345] cannot be broadcast to indexing result of shape [2, 345, 1]
```

trace 最后依次为 `environment-construction-start`、`fatal-exception-recorded`、`simulation-app-close-start`，没有 `environment-construction-complete` 或 `initial-reset-start`。场景构建期间已启动模拟器，因此“诊断控制步为零”不表示完全没有发生底层初始化仿真。

| 项目 | 实际结果 |
|---|---|
| Scene creation / simulation start 日志计时 | 1.179 / 1.821 秒，仅为这两个子阶段 |
| 地形导入日志 | 24 个 mesh parts；244,068 个三角形，四档摩擦材质 |
| ground ray mesh 读取日志 | 24 个 mesh prim 合并为 122,082 顶点、244,068 faces |
| 完整环境构建 / 初始 reset | 未完成 / 未进入 |
| 20 个零动作诊断控制步 | 未进入，完成 0 步 |
| 碰撞组运行审计、接触报告与地面阳性对照 | 未进入 |
| 抬高机器人 1 的扫描干扰探针 | 未进入 |
| 五批局部 reset / 分配票据核对 | 未进入，完成 0 批 |
| 结果 JSON / shared journal | 均未产生 |
| 监督器总耗时 | 114.3168 秒 |
| GPU 每秒采样峰值 | 2,818 MiB，约 2.75 GiB；109 条采样 |
| CPU 峰值 RSS | 未取得最终报告，不作数值声明 |
| 子进程 / 监督器退出码 | 0 / 1；不能用子进程的零退出码判定通过 |
| 运行前后及事后哈希 | 139 项源码/审计文件、10 项本地依赖均一致 |
| 退出后 GPU0 | 18 MiB、利用率 0%，无计算进程 |
| 残留测试进程 | 未检出 |

控制台、Kit 和 Isaac Lab 日志没有地形 cooking failure 或 `[Error]` 行；控制台有上述 Python traceback，足以判定失败。既有足端 visual USD 引用警告仍存在。GPU2、GPU3 的无关计算进程保持原有 PID，没有操作它们。

## 收尾限制：异常没有被监督器自动识别

异常后子进程停在 `simulation-app-close-start`，没有及时自行退出。当前监督器只自动识别地形 cooking 错误与超时，未将 Python traceback / `fatal-exception-recorded` 作为立即停止条件。

确认失败后核对了子进程 PID、PGID 和本次输出目录，向本次子进程组 `3853110` 发送 SIGTERM，监督器随后完成收尾并判为失败。操作记录见 [manual_stop.json](manual_stop.json)。没有操作其他训练任务。

原始 [execution.json](execution.json) 的 `stop_reason` 和 `failure_marker` 为 `null`，因为停止信号来自本次人工收尾，而非监督器内部触发。这里保留原字段，不能将其解释为无异常或已验证 Python 异常的自动停止机制。`outcome.json` 因缺少有效结果而正确返回失败，监督器退出码为 1。

## 离线诊断

[reproduce_mesh_id_shape.py](reproduce_mesh_id_shape.py) 从固定依赖中提取原始 `_initialize_rays_impl()` 和 `_update_buffers_impl()`，在 CPU 上执行这两个未修改的方法体；仅用占位父类初始化和 rank-2 ID 返回值替代真正的 ray 初始化与 Warp 调用。

345 / 9 条射线各测试整批 `[0,1]` 和单环境 `[1]`，四种情况均复现维度错误，见 [offline_diagnosis.json](offline_diagnosis.json)。345 条整批情况与实际 traceback 一致；其余三种属于离线诊断，不声称已运行真实传感器。这一步没有启动 Kit、物理仿真或修复项目/依赖源码。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  docs/lite3-source-shared-bank-2env-20261028/reproduce_mesh_id_shape.py
```

上一阶段 246 项离线回归未执行这个真实传感器更新方法，因此通过结果没有覆盖该接口不兼容。本次仅新增诊断证据，不将已有回归计数冒充修复验证。

## 下一步建议，等待用户确认

先进行离线修复：在项目内处理 mesh-ID 返回值和缓存的维度契约，保留扫描命中归属检查；补充真实方法的整批/单环境更新回归。同时让监督器识别致命异常 trace，验证关闭卡住时可在既定清理宽限内停止进程组。完成代码、离线验证和可审阅记录后，再申请一次两环境真实复测。

本轮不修改运行代码或已安装依赖，不重试。共享隔离、运动资格、吞吐、扩容与 PPO 的状态仍为未验证。既有 60 环境 reserved-row G2 通过记录不受本次判定替代。

## 证据

[manifest](manifest.json)、[execution](execution.json)、[outcome](outcome.json)、[trace](trace.log)、[console](console.log)、[Kit 日志](kit.log)、[Isaac Lab 日志](isaaclab.log)、[GPU 采样](gpu_samples.jsonl)、[退出后 GPU](gpu_after.csv)、[退出后计算进程](gpu_compute_after.csv)、[进程检查](test_processes_after.json)、[事后复核与产物哈希](post_run_review.json)。
