# 共享地形接触诊断：修正 stage 生命周期校验

日期：2026-10-28。基线：`ac3bb5bcdf8e9361e0ed6a790b0218a71d25003c`。

本阶段按用户批准完成离线排查、代码修复、回归验证与归档。**未启动 Isaac Sim、GPU 仿真、策略或训练。** 上一轮 stage 校验失败的证据保留在[真实复测记录](../lite3-source-shared-bank-2env-contact-repair-20261028/README.md)。

## 问题与依据

上一轮实际记录为：USD stage ID `9223001`、`IPhysxSimulation.get_attached_stage()` 返回 `0`、接触处理开关为 false。旧诊断要求两个 ID 无条件相等，因此在安装接触订阅前退出，未执行控制步。此前的 CPU fixture 固定返回相同 ID，遗漏了实际使用的普通 USD 生命周期。

本机源码说明这两个 ID 不能无条件等同：

- `SimulationCfg.create_stage_in_memory` 默认 false；Lab 在普通 USD 场景下直接使用 USD context 的 stage。
- `attach_stage_to_usd_context()` 对普通 USD 场景提前返回；memory stage 才走显式 PhysX attach。
- `SimulationManager._create_simulation_view()` 用 `get_current_stage_id()` 创建物理 tensor view，并发出物理就绪事件。core `SimulationContext.physics_sim_view` 返回该对象；view 的 `is_valid` 表示其物理对象是否仍可访问。
- PhysX binding 描述 `get_attached_stage()` 为通过 IPhysxSimulation 接口附加的 stage；另有 IPhysxStageUpdate 生命周期。官方 ContactReportDemo 在场景运行期间使用同一订阅接口，未要求诊断先重新 attach。

因此本次修正的是**将显式附加 ID 误用为所有场景的唯一有效性条件**。这不等于已经确定更早一次空接触报告的根因，也不保证即时接触 API 在真实 GPU 上已恢复。

## 修复

新增 `ContactRuntime` 读取现有环境与模拟器状态，在采集前及每个物理子步后核对：

| 条件 | 普通 USD 场景 | memory stage |
|---|---|---|
| 初始 stage、sim.stage、scene.stage、scene.stage_id | 均为同一正 ID | 同左 |
| USD context ID | 必须与该 ID 相同 | 记录，但不要求相同；无渲染的内存场景可以不附加到 context |
| PhysX 显式 attached ID | 允许 0 或同一 ID；其他值拒绝 | 必须为同一 ID |
| 物理 view | 非空、有效，且仍为进入诊断时的同一个对象 | 同左 |
| 生命周期状态 | 创建模式未变、timeline 正在播放、接触处理开启 | 同左 |

诊断只读取这些状态，不调用 attach/detach，不重建物理视图，不修改场景或接触开关。若运行中视图失效、被替换、USD context 改变或 timeline 暂停，在对应 post-physics hook 抛错并恢复临时 recorder hook、释放接触订阅。

接触报告的 header 仍必须携带真实的正 stage ID；**不会把 header 的 0 当成当前 stage 别名**。数据范围、有限接触点、跨环境接触检查、两机器人地面阳性对照及全部 80 次采集要求保留。

结果 schema 更新为 `shared_contact_capture_v2`，记录最终完整 runtime 状态与 81 次有效校验（采集前 1 次 + 80 个子步）。结果验证器拒绝缺失 runtime 证据、旧 v1 结果或采集不完整的成功声明。

## 离线验证

**362 passed, 51 warnings in 17.22s，无跳过。** 相比既有 334 项，新增 28 项生命周期覆盖。完整原始输出见 [regression.log](regression.log)，准确命令、环境与测试前后源码哈希见 [regression_command.json](regression_command.json)，完整验证清单见 [validation.json](validation.json)。

测试执行固定依赖中的实际 Python 函数体，仅移除类型注解与属性／绑定装饰器：Lab stage attach 分支、SimulationManager 创建物理视图的过程、core stage/view 属性与 tensor view 的有效性属性，以及既有 RL decimation、物理 step 和 recorder hook。物理接口与 tensor 后端由 CPU 桩替代，未执行 GPU 或官方仿真测试。

- 普通 USD 初始化不显式 attach，依然创建绑定正确 stage ID 的有效物理视图；该路径下 attached ID 为 0 时，合成接触数据完成 80 次采集。
- memory stage 保留显式附加要求；普通 USD 的非零错误 ID、缺失／错误 context、错初始／场景 ID、无效或被替换的 view、暂停及模式改变均不能通过。
- 运行中改变 USD context、替换／失效 view 或暂停，在首个 post-physics hook 停止，不进入下一物理子步。
- 复用上次原始 journal 中的 `9223001/0/false` 值，补充明确标为 CPU 桩提供的生命周期证据，验证修正后的条件分支；不将这些新增合成证据冒充上一轮真实遥测。
- header ID 为 0 或其他 stage 仍拒绝。原有空报告、回调错误、跨环境接触、坏数据与 reset 前置门槛继续通过回归；普通 USD 路径也不会因空报告得到成功结果。
- 完整回归继续覆盖两套扫描、共享 bank、分配与局部 reset、监督器清理、地形分块及 tile guard。

51 条 warning 均来自读取已安装 tensor API 文件时，同一上游 docstring 中 `\l` 转义的 DeprecationWarning；不是本项目代码或仿真运行告警。保留原始输出，未修改或屏蔽安装文件。

依赖审计由 22 项扩展为 **29 项**，原 22 项记录完全保留，全部哈希一致；新增固定配置、stage 工具、环境生命周期、SimulationManager、tensor view 和 USD context 接口。4 个修改的 Python 文件语法检查通过，旧失败归档及已安装文件未变。新增 runtime helper 位于监督器原已监控的脚本内，仍受运行前后哈希校验覆盖。

## 待批准的一次真实复测

沿用两环境、GPU0、seed 130196、20 个零动作控制步、80 个物理子步和五批直接 reset。运行前核对干净提交、29 项依赖与 6 项资产、GPU0 空闲且可用显存至少 8 GiB。使用全新目录：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_stage_repair_20261028
```

**本阶段未执行。** 只运行一次；启动 120 秒、总时限 240 秒，另留最多 10 秒清理。任何失败即停止，不自动重试、延长或扩容。通过也仅支持该配置的两环境短时检查，不能代替完整物理隔离、G3/G4 或策略资格验证。按用户逐阶段确认的约定，真实复测需再次确认。
