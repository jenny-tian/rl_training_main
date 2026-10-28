# 监督器结果汇总离线修复

2026-10-28，基线 `1de8ce6`。用户明确批准修复监督器汇总，并使用现有证据离线验证。本阶段没有启动 Isaac Sim、物理积分、GPU 探针、策略或 PPO。

**汇总导入缺陷已修复，407 项回归通过；已归档成功结果可在独立 Python 进程中生成通过的汇总。**

## 问题与改动

监督器此前在仿真子进程成功退出后导入 `source_shared_terrain_diagnostic`，该模块加载任务包，进而在未启动 Kit 的父进程中报 `ModuleNotFoundError: No module named 'pxr'`，无法写入 outcome.json。上次真实仿真的原始证据及复现堆栈见 [实测记录](../lite3-source-shared-bank-2env-tensor-repair-20261028/README.md)。

- 将 `shared_report_passed` 和控制步常量移入 `source_shared_report_validation.py`。诊断脚本与监督器共用此函数；其函数 AST 与上次实现完全一致，未放宽验收条件。依赖仅为已有 tensor/contact 校验代码及 NumPy，不导入任务包、torch 或 Kit。
- 将新模块加入监督器源码哈希清单。
- 提取 `finalize_run`，把源码/依赖哈希复核、显存样本读取、结果解析与验收包在异常记录边界内。失败写入 `finalization_error`（阶段、异常类型、消息），并保存 execution.json 和失败 outcome.json。
- 保留子进程退出码、停止原因、cooking 和迟到 fatal 标记；有效结果文件不能覆盖它们。普通启动和 reset-lifecycle 模式沿用既有判据。
- 写入目录本身不可写、磁盘故障或进程被强杀仍无法保证保存文件；写入失败会传播，未宣称能处理全部系统级异常。

## 验证

[完整回归日志](regression.log)：**407 passed, 315 warnings in 28.84s**，无跳过。命令、环境、测试期间源码哈希见 [regression_command.json](regression_command.json)。315 条 warning 与上一轮相同，来自对安装版 tensor API docstring 的 AST 读取。

新增 13 个案例覆盖：

1. 全新 Python 子进程中执行真实监督器 main，仅读取已归档 result/execution/GPU samples；替代预检和启动函数，不启动任何仿真。独立 import hook 明确禁止 rl_training、isaaclab、isaacsim、omni、pxr、torch。真实成功报告产生通过 outcome；缺地面阳性对照的变体产生失败 outcome；新模块确实进入 manifest。
2. 缺失结果、坏 JSON、错误顶层类型、错误 topology、源码缺失或变化、坏 GPU 样本、导入异常、验收函数异常、子进程失败，以及迟到接触丢弃标记均保存失败 outcome。原子进程错误信息保留。

第一轮针对性测试有一项测试断言写错：错误 topology 被原验收函数正常判为 false，不需要抛异常；测试原先却要求异常字段非空。修正该测试预期后完整回归通过，未为通过测试修改验收判据。

另用 [replay_finalization.py](replay_finalization.py) 在临时目录直接执行新汇总函数，读取同一份真实结果，成功生成 outcome。输出明确标记为离线回放，见 [offline_replay.json](offline_replay.json)。原始实测目录没有被回填 outcome；此回放校验的是当前汇总代码哈希，不将它冒充上次 GPU 的运行版本。

从仓库根目录复核（不启动仿真）：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python docs/lite3-source-supervisor-finalization-offline-repair-20261028/replay_finalization.py
```

原归档 16 项文件 SHA256 全部一致，32 项安装依赖、6 项机器人资产均一致，详见 [validation.json](validation.json)。本轮没有改变接触采集、碰撞过滤、物理参数或训练配置。

## 当前结论与边界

上次真实 GPU 的两环境 20 零动作步接触/扫描/直接 reset 证据通过；本轮独立进程离线验证修复了其汇总路径。原监督器历史退出码仍未知，不改写历史结果，也不声称修复后的完整 GPU 流程已重新运行。

本次批准的离线修复已完成。更大规模仿真、运动隔离、自动 reset、G3/G4 或训练不在本阶段范围内，继续遵守逐阶段确认要求。
