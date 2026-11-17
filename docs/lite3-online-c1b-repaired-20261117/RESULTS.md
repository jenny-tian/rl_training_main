# C1-b 剩余短检查报告：重置 API 错误，名额已用完

2026-11-17。用户批准了 [上一阶段继续提案](../lite3-online-c1b-20261117/NEXT_STAGE_PROPOSAL.md)。本阶段只启动了一次 GPU 1 短检查，初始重置失败后立即停止；未启动正式训练、目标评估或保持评估。

## 执行结果

已用剩余 `smoke_2`：16 个环境、每环境最多 600 个控制步、禁止在线参数更新。上次修复的延迟类型写入顺利经过，随后访问尚不存在的 `reset_buf` 报错。执行耗时约 12.33 秒，退出码 1；采集器还没有执行 `env.step`。

| 项目 | 本阶段实际值 |
|---|---|
| 采集器准备/策略/等待控制步 | 0 / 0 / 0 |
| 基础 actor 前向调用 | 0 |
| 基础策略/在线适配器更新 | 0 / 0 |
| 已完成策略回合 | 0 |
| 修复后再次仿真 | 0 次 |
| C1-b 短检查名额 | 累计已用 2/2，均失败，剩余 0 |
| GPU 1 入场空闲显存 | 23468 MiB，门槛 10240 MiB |
| 采样 GPU 总占用峰值 | 3756 MiB，2 秒采样，非进程独占显存 |

0 个控制步指采集循环的控制交互，不包括初始化的仿真引擎操作。两次失败各占用 9600 个环境控制步的名额预算，累计 19200；实际采集控制步均为 0。正式训练和评价的 924800 步预算尚未使用，但不能用它补充短检查名额。

此次仍没有新的上台阶成功率、在线收益或保持结果。B10200 检查点哈希不变：

`aba3a52e362fbb6f6c860bb8db384c12a84d13f6f39b53cae16465eaf2a84b79`

## 两个代码问题及离线修复

### 1. 初始重置提前访问 step 返回状态

实际故障是 `reference_online_c1b_env_repaired.py:139`：

```text
AttributeError: 'ManagerBasedRLEnv' object has no attribute 'reset_buf'
```

当前安装的 Isaac Lab 在 `manager_based_rl_env.py` 的 `step()` 第 204–206 行才创建 `reset_buf`、`reset_terminated`、`reset_time_outs`。构造函数和首次 `reset()` 不创建它们。新增的配对重置代码假设这些属性已经存在，违反了该版本的调用顺序。

候选修复 [reference_online_c1b_env_reset_api.py](../../scripts/tools/reference_online_c1b_env_reset_api.py) 删除这三处赋值。原生 `_reset_idx` 继续负责场景、各管理器和回合计数的重置，终止管理器在下一步重新计算终止状态。显式重置不改写已经返回的终止张量，采集器仍在重置前复制当步标记。

其余配对起点、0/1 步执行器延迟、安全历史和命令计算沿用登记版本。根状态和关节状态仍直接写入模拟器，保持原有 forward/update 顺序；未增加全场景写入，避免额外推进未重置环境的执行器延迟缓冲区。

### 2. 完整 CPU 流程发现日志序列化错误

用候选重置入口运行现有采集器的 CPU 接口测试后，进一步复现：`update_logs` 中来自 `np.flatnonzero` 的 `env_id` 是 `np.int64`，原 JSON 保存函数无法序列化。正常结束或异常保存时都可能受影响。这是离线测试发现的问题，第二次 GPU 仿真没有执行到这里。

候选 [reference_online_c1b_collect_reset_api.py](../../scripts/tools/reference_online_c1b_collect_reset_api.py) 仅在日志写入边界递归转换 NumPy 标量为等值 Python 标量，继续保留原写入函数的非有限值拒绝；不把任意对象转换成字符串，不改学习数据和控制流程。原采集器保持哈希不变，包装器执行后恢复原保存函数。

前两轮离线检查偏重独立组件，未覆盖新多回合入口的完整调用顺序和结果保存。这是此次实现与验证的不足，不能把两次接口故障解释为策略能力或奖励问题。

## 扩大后的离线验证

相关检查合计 **60 项通过**。其中新增 8 项覆盖：

- 调用当前安装版本的原生初始重置方法，重现 `reset_buf` 缺失；候选重置通过并到达观测刷新。
- 局部配对重置保留其他环境根状态、动作历史和回合计数，保留已返回的终止标记，下一步原生终止计算重新生成结果。
- 原采集器完整 600 次 CPU 控制循环，重现旧 JSON 保存错误；包装器修复后完成 3 回合的准备、策略和异步等待记账。
- 在准备时注入终止条件，检查立即停止、保存部分记录、物理计数与控制步关系。
- 合成训练中完整经历 64 个回合，Frozen 更新为 0、Updated 更新 64 次；保存末态后载入目标和保持评价入口，评价参数冻结且关闭探索。
- NumPy 数值转换保留类型和值，继续拒绝 NaN。

这些测试通过解析当前安装的源码，执行原生 `reset`、`_reset_idx`、`step`、动作管理器和终止管理器的方法体，配合实际延迟缓冲区及锁定采集器。场景、物理、actor 和基础检查点为 CPU 测试替代实现；没有导入 AppLauncher、启动物理引擎或 GPU 仿真。合成训练的参数更新与假想控制步只属于单元测试，不能计作在线学习结果、仿真预算或机器人能力证据。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_curriculum/test_c1b_native_reset_contract.py \
  tests/ea_lpl/test_c1b_delay.py tests/ea_lpl/test_c1b.py \
  tests/ea_lpl/test_current_safety.py tests/ea_lpl/test_current_shadow.py \
  tests/ea_lpl/test_residual_bridge.py \
  tests/source_curriculum/test_reference_online_interface.py \
  tests/source_curriculum/test_reference_online_failure_capture.py \
  tests/ea_lpl/test_reset_table.py
```

修复环境、日志包装、待批准继续入口、证据脚本和新增测试均通过语法编译。[EVIDENCE.json](EVIDENCE.json) 记录测试 XML 的计数与哈希、实际运行异常、计数、原生状态属性赋值位置、原始产物和修复源码哈希。[LOCK.json](LOCK.json) 保留本次执行前登记。

原始数据位于工作树 `logs/reference_six/online_c1b_repaired_20261117/smoke_2/`。原失败源码和第一版修复源码均保留，未就地修改；没有为收集证据再次启动仿真。现有安全链仍使用仿真射线与接触信息，离线接口通过不等于实机部署可用。

## 下一阶段边界

当前已用完全部获批短检查名额。已准备 [下一阶段提案](NEXT_STAGE_PROPOSAL.md)：请求新增恰好 1 次 16 环境 × 600 步检查，最多增加 9600 个环境控制步名额预算；通过后才继续原在线训练和两组评估，失败则停止。提案未获批准前不登记或运行该新增名额。

研究主线保持：冻结六类基础策略 → 6 cm 台阶 Frozen/Updated 机制小试 → 独立目标收益与基础技能保持 → 再讨论跨地形持续积累。此前固定码独立确认 9/16 对 9/16 的负结果继续有效，本次不增加转向分支或修改奖励。
