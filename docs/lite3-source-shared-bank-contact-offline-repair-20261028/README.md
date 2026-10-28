# 共享地形接触报告链路：离线修复与待批准复测

日期：2026-10-28。基线：`10abdaec517a6b562d2801c6443a08c123904599`。

本阶段获准范围是离线排查、修复、测试、提交与推送。**没有启动 Isaac Sim、GPU 仿真、策略或训练。真实复测须按用户逐阶段确认的约定另行批准。**

## 问题与已知边界

[上一次真实复测](../lite3-source-shared-bank-2env-repair-20261028/README.md)完成了初始化、扫描干扰探针与 20 个控制步，但接触 journal 没有记录，缺少两个机器人的地面接触阳性对照，未进入五批局部 reset。旧诊断只记回调中的 header，不记回调调用次数，也不记足端高度或接触力，无法区分回调未调用、空回调和机器人尚未触地。

本机绑定说明 `get_contact_report()` 的缓冲只保留一个物理步。`ManagerBasedRLEnv.step()` 的 decimation 为 4，单步物理时长 0.005 秒；仅在每个控制步末尾读取可能漏掉前三个子步中的报告。`PhysicsContext._step()` 在 `simulate()` 后阻塞等待 `fetch_results()`，随后 RL 环境调用 recorder 的 post-physics hook。

**此前回调静默的根因尚未证实。** Isaac Lab 默认禁用 contact processing，但 ContactSensor 构造时会重新启用；上一轮没有记录运行时设置，不能据此断言该开关就是原因。

## 实现

- 新增纯 Python `source_shared_contact_report.py`，在 20 个控制步的全部 **80 个物理子步**后读取即时报告。核对连续的 `_sim_step_counter` 及每个控制步的 4 次采集；缺失或重复采集均失败。
- 临时包装当前空 recorder 实例的 `record_post_physics_decimation_step`，保留原方法调用，并在 `finally` 恢复原实例属性或移除临时覆盖。没有新增 recorder term、修改类或其他实例，因此不会触发 RL step 中为活跃 recorder 额外计算 observation 的分支。
- 保留接触回调作为独立诊断渠道，记录调用次数、header/data 数量，区分静默和空回调。回调与即时报告分别计数，**仅即时报告中验证通过的 found/persist 接触点计入地面阳性对照**。
- 采集前及每个物理子步检查 attached stage 与当前 USD stage 一致，且 `/physics/disableContactProcessing` 明确为 false；不会自动修改运行时开关。
- 校验 header stage、事件类型、已审计刚体与 collider 的归属、数据 offset/count、重复或未引用的数据区间，以及 position/normal/impulse/separation 的维度与有限性。lost 事件不计入阳性对照。任一渠道的跨环境接触、未知归属或坏数据均失败。
- 回调异常显式留存，由该物理子步结束后的采集 hook 抛出，使失败发生在下一个物理子步开始之前。采集尝试、空报告、错误和最终 summary 都刷新到 journal；采集作用域异常时也释放订阅、恢复 hook。
- 初始状态与各控制步末尾记录足端 **body origin 相对 tile 的高度**、足端及全身 net contact forces，并按名称匹配 articulation 与传感器的不同排列。高度不是 collider 离地间隙，net force 也不替代成对接触证据。
- `shared_report_passed()` 必须同时验证完整 80 步覆盖、两环境阳性对照、零错误与原有扫描/reset 条件。监督器把新增 helper 纳入运行前后文件哈希检查。

两环境、单行 15 列 bank、初始 flat 重叠、seed 130196、20 个零动作控制步、五批直接 reset 和运行预算保持既定方案。

## 本机依赖核对

[依赖审计](../lite3-source-shared-bank-preparation-20261028/dependency_audit.json)从 14 项扩充为 22 项，原 14 项路径与哈希保持不变，全部安装文件哈希验证一致。新增锁定 recorder、Lab/core simulation context、core physics context、PhysX bindings、接触文档及官方测试源码。

注意本机文档存在过时示例：实际绑定枚举是 `ContactEventType.CONTACT_PERSIST`（单数），实现按 `.pyi` 使用；即时 API 按已安装官方测试解包为 `(headers, data)`，摩擦锚点属于另外的 full-report API。官方 PhysX 测试文件仅阅读，未执行。

## 离线验证

**334 passed in 13.95s，无跳过。** 包括既有 293 项与新增 41 项。原始输出见 [regression.log](regression.log)，命令、源文件哈希和验证范围见 [validation.json](validation.json)。

新增测试执行哈希锁定的 `ManagerBasedRLEnv.step`、`PhysicsContext._step` 和 recorder hook 的完整实际函数体，仅去除类型注解，并以 CPU 桩替代物理接口与管理器，验证：

- 仅在控制步第一个子步出现的短暂接触被捕获，读取顺序严格位于 `fetch_results` 后、scene update 前；静默/空/正常回调可区分且不重复计数。
- malformed header、错 stage、坏 offset/count、非有限点、未知归属、lost、重复/孤立数据均不能伪造阳性对照。
- 跨环境接触、回调异常、轮询 API 错误、原 hook 异常或中途禁用 processing 后，在下一物理子步前停止，并恢复 hook、释放订阅、输出失败 summary。
- 缺采集、提前结束或缺任一地面阳性对照不能通过；完整诊断的合成集成测试确认失败时不会调用 reset 检查，传感器力按足端名称正确映射。
- 完整回归继续覆盖 ray caster、共享 bank、分配、局部 reset、监督器限时清理、地形分块和 tile guard。

这些结果证明 Python 采集与拒绝失败的逻辑，不证明真实 GPU 接触报告已经恢复。没有执行真实 PhysX、Warp raycast 或 Kit 关闭测试；共享物理隔离与 G3/G4 仍未资格化。两次原始失败归档保持不变。

## 待批准的一次真实复测

运行前再次核对干净提交、22 项依赖与资产哈希，以及 GPU0 空闲且可用显存至少 8 GiB。使用全新目录，继续由监督器限时和清理。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_contact_repair_20261028
```

**本阶段未执行此命令。** 启动时限 120 秒、总时限 240 秒，另留最多 10 秒进程组清理。只运行一次，任何失败即停止，不自动重试、放大、延长预算或训练。通过也只支持该代码与配置下的两环境短时 smoke，不能推导完整运动隔离资格。

**后续真实复测（2026-10-28）：** 用户确认后，从干净提交 `e5c29d8` 执行了一次上述命令。初始化和扫描探针完成，但接触采集前因 USD stage `9223001` 与 PhysX attached stage `0` 不一致而失败；接触处理开关为 false。订阅尚未安装，80 次采集与 reset 尚未进入，本次不能验证真实接触报告已恢复。监督器自动收尾并释放 GPU0，没有修改运行代码或重试。详见 [完整复测记录及接口语义复核](../lite3-source-shared-bank-2env-contact-repair-20261028/README.md)。
