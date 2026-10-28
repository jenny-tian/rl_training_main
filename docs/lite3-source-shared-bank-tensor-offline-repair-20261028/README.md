# 共享地形接触诊断：使用 GPU 成对 tensor 数据

日期：2026-10-28。基线：`1669785`。用户已批准离线排查与修复，本轮未启动仿真。

**代码已改为读取 GPU 成对接触 tensor；394 项离线回归通过，真实 GPU 复测尚未执行。** 旧报告为空的历史证据保持原样，见[上次实测](../lite3-source-shared-bank-2env-stage-repair-20261028/README.md)。共享物理隔离仍未通过资格验证。

## 为什么前面反复修复

之前补了回调采集时机、逐子步轮询，又修正了普通 USD stage 的错误前提，但没有先核对 GPU 模式的报告支持范围。CPU 后端桩直接提供合成接触事件，证明了 Python 校验和时序，无法证明实际 GPU 会生成这些事件。将这些局部修补推进到真实复测，造成多轮可避免的失败。

上一轮已经记录到两环境足端有净接触力，而 80 次即时报告全部为空、回调调用为 0；这将问题指向数据输出链路。此次查到：

- 本机 `SimulationManager.set_physics_sim_device()` 的 CUDA 分支设置 `suppressReadback=True`；`IPhysx.is_readback_suppressed()` 可读取实际运行状态。
- [PhysX Direct GPU 文档](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/DirectGPUAPI.html)明确说明，GPU narrow-phase 接触数据不复制到 CPU contact reporting；创建场景后也不能靠切换这些场景标志修复。
- [本机对应 107.3 tensor API 文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)提供成对力矩阵及接触点读取。本机已安装 Python 方法和 native 插件包含这些 GPU 接触读取实现。
- 107.3 的通配符过滤默认一对一，不能把它当成跨机器人所有刚体组合。其文档给出的多对多方式是每个传感刚体提供显式的对端路径列表。

这是修复接口选择的依据。**上次实测未记录实际 suppressReadback，因此不能追认上次运行开关的值，也不能声称已经用真实 A/B 实验证明全部空报告的因果关系。** 当前仍需运行一次实际 tensor 检查。

## 改动

`source_shared_tensor_contact.py` 复用现有场景生命周期校验和 recorder 子步挂钩，从现有物理 simulation view 创建一个接触 view。传感对象是全部 34 个机器人刚体；每个刚体的过滤列表显式包含全部机器人刚体和所有已启用地形 collider。检查原生接口实际解析的 `sensor_paths` 和逐传感器 `filter_paths`，允许重排序，但拒绝缺失、重复、未知路径或运行中改变。地面由当前 USD 中的碰撞网格枚举；本机 terrain utility 创建的静态 Mesh 自身是其碰撞对象路径。

每个物理子步读取并复制净力、成对力矩阵和六个接触数据 buffer；使用实际物理 dt `0.005 s`，不使用控制 dt `0.02 s`。真实生效的回读状态、设置值和 tensor 设备在采集前及每个子步记录，要求一致且不变。不会修改回读、GPU dynamics、碰撞过滤或物理场景。

验收条件保留两环境地面阳性对照、20 控制步、连续 80 子步和 reset 前置门槛，另增加：

1. 完整笛卡尔积对端覆盖；跨环境接触点数大于 0，或成对力有任何非零分量，立即失败。接触点存在但合力抵消仍失败。
2. 明确分配 16,384 个接触点容量；占满、越界、范围重叠、非整数计数或非有限有效数据失败。只读取有效范围，不把未使用 buffer 尾部当成接触。
3. 每对接触点的法向力之和须与独立成对力矩阵一致；全部对端力之和须与净力一致。绝对／相对容差均为 `1e-4`，检查失败拒绝采集完整性。净力仅作交叉校验，不代替对端证据。
4. 监督器识别接触容量不足和丢弃接触的 native 日志，包括本机插件中的 `Contact buffer overflow`、`Dropping contacts`、`maxNbContactDataBlocks` 等标记；退出后迟到日志也受原有最终扫描约束。

结果 schema 为 `shared_tensor_contact_capture_v1`，旧 CPU contact schema 不再满足当前共享检查的成功条件。旧 collector 保留用于既有失败场景的回归测试；新诊断不订阅 CPU 接触事件。

已有静态[资产审计](../lite3-usd-collision-geometry-audit-20261027.json)记录 22 个碰撞形状：6 Cube、12 Capsule、4 Sphere；六项资产哈希保持一致，没有依赖 Direct GPU 不支持的自定义 Cone/Cylinder 接触。地面仍是既有静态三角网格。

## 验证和边界

完整回归：**394 passed, 315 warnings in 28.98s，无跳过**。覆盖接触、扫描、共享 bank、分配/reset、监督器进程清理、地形分块与边界。新增 32 项测试，原来的集成测试同步切换到新诊断链路。

测试执行固定依赖的实际 `SimulationView.create_rigid_contact_view`、`RigidContactView` 构造、路径属性和三种 getter 函数体，使用 CPU 合成后端，不加载 GPU native 插件。确认了实际 Python 分配形状、六个返回 buffer 的顺序、dt 传递、异常传播，以及真实 RL step/recorder 调用顺序。另执行完整 `run_shared_diagnostic()` 的 CPU 集成测试：成功时完成 80 子步再进入 reset；空接触时不 reset；跨环境接触在第一个子步停止。

故障覆盖包括缺失对端、只有净力、容量占满/超限、坏范围、重叠范围、部分截断、力矩阵不一致、非有限数据、错误法向量、原生 getter 失败、运行中设备/视图/路径/时步改变。unused buffer 中 NaN 和解析路径重排序不会制造误报。

315 条 warning 为 AST 读取同一个已安装 API 文件时重复报告上游 docstring 的 `\l` 转义，未修改安装文件。依赖由 29 项扩展为 32 项，旧 29 项完全保留；新增 terrain utility、PhysX tensor native 插件和核心 native 插件的哈希。新脚本已加入监督器源码哈希清单。

这些测试仍不证明真实 GPU 数据和路径解析已通过；也不证明所有物理接触都绝不会丢失。容量、数据一致性和日志检查是本次有界探针的验收措施。通过后的结论也仅限两环境短时检查，不是完整 G3/G4 或训练资格。

[回归日志](regression.log)、[准确命令与运行期间哈希](regression_command.json)、[验证清单](validation.json)。

## 待确认的一次真实复测

从干净提交运行前，核对 32 项依赖、6 项资产和 GPU0 资源；GPU0 必须空闲且可用显存至少 8 GiB。新目录不可复用旧证据：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_2env_tensor_repair_20261028
```

只运行一次，失败停止，不自动重试；保留原 120/240 秒预算和 10 秒清理宽限，完成后归档并释放资源。本阶段没有执行该命令。按用户此前逐阶段确认的要求，真实仿真需获得这一阶段的确认。
