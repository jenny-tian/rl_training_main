# C1-b 小试阶段报告：首次重置失败，已完成离线修复

2026-11-17。状态：按接口错误停止条件结束本阶段；没有启动在线训练或正式评估，修复版尚未经过仿真验证。

## 本阶段执行到了哪里

用户批准了 [状态条件残差小试](../lite3-online-targets-current-20261117/NEXT_STAGE_PROPOSAL.md)。已实现 8 对 Frozen/Updated 会话、每个适配器 18 个参数、配对重置与探索、逐完整回合更新、包含准备和等待的控制步账本，以及训练后目标/保持评估入口。基础策略为冻结的 B10200，目标为 6 cm 上台阶。

首次短检查使用 GPU 1、16 个环境，计划上限 600 个控制步且禁止参数更新。环境初始化后，第一次显式配对重置在写入执行器延迟时出错。进程以退出码 1 结束，耗时约 12.35 秒；此后没有启动其他仿真。

| 项目 | 本阶段实际结果 |
|---|---|
| 短检查名额 | 使用 1/2，首次失败；仅剩 1 次 |
| 采集器控制步 | 0，准备/策略/等待计数均为 0 |
| 基础 actor 前向调用 | 0，故障发生于控制循环之前 |
| 基础策略更新 | 0，检查点哈希未变 |
| 在线适配器更新 | 0；离线单元测试中的合成更新不属于在线交互 |
| 正式训练、目标评估、保持评估 | 均未启动 |
| 修复后仿真 | 0 次 |
| GPU 入场可用显存 | 23468 MiB，高于门槛 10240 MiB |
| 采样到的 GPU 总占用峰值 | 3377 MiB，采样间隔 2 秒，非进程独占显存统计 |

0 个控制步指采集器尚未执行 `env.step`，不表示环境初始化完全没有仿真引擎工作。此次没有形成可报告的上台阶成功率或学习收益。

## 根因及修复

这是本次新增逐环境重置代码的整数类型兼容问题。当前安装的 Isaac Lab `DelayBuffer` 使用 `torch.int32` 存储延迟；重置代码创建的延迟张量默认为 `torch.int64`。旧的整批切片赋值允许转换类型，新的按环境索引赋值要求源和目标类型一致，因此报错：

```text
Index put requires the source and destination dtypes match,
got Int for the destination and Long for the source.
```

错误位置是 `reference_online_c1b_env.py:136`。修复在写入前读取缓冲区的实际类型和设备，转换延迟张量，然后调用原生接口。延迟仍是按配对键生成的 0/1 步整数，未改变分布、配对关系或其他环境的历史。

已用当前安装的原生 `DelayBuffer` 和 `CircularBuffer` 在 CPU 上复现同一错误，并验证修复。复现使用合成整数值，未声称恢复了故障时未保存的原始延迟张量。修复测试覆盖部分环境写入、未重置环境历史保留、缓冲区存储类型变化及非法输入拒绝。

原运行源码已在执行前通过 [LOCK.json](LOCK.json) 登记哈希，现保持不变。修复单独位于：

- [c1b_delay.py](../../source/rl_training/rl_training/ea_lpl/c1b_delay.py)：延迟写入兼容函数。
- [reference_online_c1b_env_repaired.py](../../scripts/tools/reference_online_c1b_env_repaired.py)：候选环境；与失败版本相比，功能差异只有调用兼容函数。
- [reference_online_c1b_continue.py](../../scripts/tools/reference_online_c1b_continue.py)：待批准的继续入口，继承失败名额，只允许使用剩余 `smoke_2`；原采集器和学习协议保持锁定。

最初离线检查未覆盖原生缓冲区的索引写入路径，导致问题到仿真启动时才暴露。本次补上了直接调用已安装原生缓冲区的回归测试。CPU 检查通过不能替代完整环境中的重置验证。

## 验证与证据

本次合并离线检查为 **52 项通过，耗时 2.94 秒**，覆盖适配器、延迟修复、当前安全过滤、影子计算、残差桥接、当前接口、异常现场保存和重置表。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/ea_lpl/test_c1b_delay.py tests/ea_lpl/test_c1b.py \
  tests/ea_lpl/test_current_safety.py tests/ea_lpl/test_current_shadow.py \
  tests/ea_lpl/test_residual_bridge.py \
  tests/source_curriculum/test_reference_online_interface.py \
  tests/source_curriculum/test_reference_online_failure_capture.py \
  tests/ea_lpl/test_reset_table.py
```

修复环境、兼容函数、继续入口和证据脚本均通过 Python 语法编译检查。[EVIDENCE.json](EVIDENCE.json) 保存失败运行信息、异常现场、实际计数、原生 CPU 复现结果，以及修复源码、原生缓冲区源码和本地原始产物的哈希。证据脚本检查了失败进程已退出、原运行锁一致，以及 B 检查点哈希保持：

`aba3a52e362fbb6f6c860bb8db384c12a84d13f6f39b53cae16465eaf2a84b79`

原始日志和数组留在工作树的 `logs/reference_six/online_c1b_20261117/smoke_1/`，包括 `execution.json`、`runtime.log`、`error.json`、`failure_capture.json`、`failure_trace.npz`、配置、地形几何与噪声表。没有为了补证据重跑仿真。

## 预算与后续边界

| 内容 | 原上限（环境控制步） | 当前账目 |
|---|---:|---|
| 两次短检查 | 19200 | 首次名额预留 9600 已消耗，实际控制步 0；剩余名额最多 9600 |
| 8 对会话在线交互 | 320000 | 未开始 |
| 256 个独立目标评估回合 | 268800 | 未开始 |
| 320 个保持评估回合 | 336000 | 未开始 |
| 总额 | 944000 | 保留失败名额，待批准继续的剩余额度最多 934400 |

不因实际控制步为 0 就恢复失败名额。上一阶段的 4 次接口短检查也维持原账，不与本轮混用。

当前尚不能判断状态条件残差是否有收益。此前固定码在独立起点上为 9/16 对 9/16、姿态失败由 0 增至 1 的负结果继续保留。

下一步详见 [剩余名额继续提案](NEXT_STAGE_PROPOSAL.md)。其中明确请求允许以剩余一次完整短检查通过为继续条件，不把首次失败写成通过。原提案要求接口错误后立即停止，用户也要求阶段间先汇报再批准，因此本阶段只完成离线修复和归档。
