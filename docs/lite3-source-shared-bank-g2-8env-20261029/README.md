# 共享 bank 八环境 G2 实测记录

日期：2026-10-29

冻结代码：`a34b1b7e03e16bf61512be67555e1ba3bf269988`

机器人模型：`b452a1f404d37155400242f18e454fed8bcc51b8`
结果：**八环境 shared-bank G2 通过；G3/G4 未运行。**

## 固定运行范围

本次从冻结提交的一次性运行只使用 GPU0、8 个环境、单行 shared bank、15 个地形列、seed `130196`，命令及完整源码/依赖哈希保存在 [manifest.json](manifest.json)。显存准入线为 12,288 MiB，启动/总预算为 120/240 秒；未加载策略，也未启动 PPO。

监督器退出码为 0，`outcome.json` 和 `result.json` 均报告通过，未记录 stop reason、失败标记、致命异常或汇总异常。运行耗时 44.765 秒，环境构建 2.637 秒，CPU 峰值 4,080,644 KiB，GPU 采样峰值 2,822 MiB。GPU0 收尾快照为 18 MiB。GPU3 在启动前已有其他计算进程，本次未使用或操作该设备；运行后快照仍记录 GPU3 的进程。进程快照文件中的 `pgrep` 结果匹配了执行快照采集的 shell 本身，不能把它误读为仿真进程残留。

## 离线复核结果

- 用项目的无 Kit 报告校验器重新检查 `result.json`，`shared_report_passed`、`shared_lifecycle_passed` 和 `tensor_summary_passed` 均为真。
- `result.shared.jsonl` 有 20 个控制步、每步 4 个采样子步，共 80 次接触采样；另有 81 次运行时状态校验。8 个环境的地面阳性对照均通过，接触点计数为 env0–7：276、268、273、272、274、274、276、272；跨环境接触为 0，采集错误为空。
- 接触拓扑有 136 个唯一机器人传感刚体（每环境 17 个）、24 个唯一地面碰撞体。每个传感刚体的过滤集合都包含全部 136 个机器人刚体和全部 24 个地面目标，共 160 个唯一目标。容量为 65,536，日志记录峰值使用 61；力残差满足报告容差。
- `result.shared.resets.jsonl` 包含初始 8 张分配、33 组 reset 前/分配比较/边界检查记录。32 批直接 reset 覆盖 120 张票；批次为 8 个单环境、16 个三环境不连续/乱序批次和 8 个全环境批次。env7 随后发生一次自动 timeout reset。实际分配与独立确定性回放一致，票号完整连续为 0–128：一个完整的 120 票块，以及 9 张尾票。未被 reset 的环境分配保持不变，33 次边界检查均报告 untouched buffers 相等；记录的 history buffers 为空。
- 自动 timeout 只截断 env7。自动 reset 的扫描结果有限，紧接着的一个物理控制步通过，未触发额外 reset。

完整原始文件、逐物理子步接触记录、逐 reset 日志、运行输出、资源快照及最终结果均保留在本目录。`SHA256SUMS` 列出本目录其余文件的 SHA-256（清单不对自身做哈希）。

## 日志复核与证据边界

日志中有 32 条 Lite3 足部视觉 USD 引用缺失警告。本次 G2 没有验证视觉资源完整性；该警告不被本次结果追认为已解决。还记录到 MaterialX 可选扩展、PhysX 接口依赖/弃用、IOMMU 与 P2P 性能提示，以及外力更新和未启用 heading 命令配置提示。未发现 `[Error]`/`[ERROR]`、Python traceback、fatal exception、CUDA/PhysX 错误或 cooking failure 标记；监督器没有超时。

G2 的 reset 边界比较只覆盖每次 `_reset_idx` 前后，并明确排除了物理积分步；当前运行没有非空 observation history buffers。因此，结果不证明运动过程中的跨环境隔离、边界行为、一般 history-buffer 清理或 checkpoint 恢复语义。零动作接触探针不等于运动策略隔离测试。未预登记吞吐目标，也未测量 G4 所要求的完整性能指标；本次不授予 G3、G4、120 环境或 PPO 资格。
