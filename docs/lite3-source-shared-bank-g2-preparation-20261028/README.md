# 共享 bank 两环境 G2 准备

2026-10-28，基线 `8ef6539`。用户选择“先准备共享 bank G2”，本阶段完成诊断、离线测试和运行方案。**418 项测试通过；尚未进行共享 G2 的真实 GPU 检查。**

## 固定范围

两环境共用一个 bank 行，15 等权地形列，L1、12×4 m tile；地形 seed 10001、环境和 allocator seed 130196。沿用已有共享 smoke 中初始两机器人重叠于 flat 的设计。显式组合 `--shared-bank --reset-lifecycle` 进入新模式；两个参数分别使用时，保留原短时共享 smoke 和 60 环境 reserved-row G2 模式。

依次执行：

1. 原有共享碰撞组、初始重叠、地面扫描/遮挡探针与 20 零动作控制步。GPU 成对接触数据须完整覆盖 80 个物理子步，两环境地面阳性对照均存在，跨环境接触为零。
2. 三批 `[0]`、`[1]`、`[1,0]` 循环 30 次，共 90 批、120 次直接 reset。每次在 `_reset_idx` 紧邻边界比较未 reset 环境的分配、row/column、origin/root/关节、动作、命令、episode、传感器原始缓存和已配置历史；选中状态按既有规则清理。独立共享 allocator 逐票核对，不能使用被事务替换后的旧 allocator。
3. 每批 reset 后 forward 刷新两套扫描，核对 shared ground mesh ID、有限值和平入口高度，重算观测。
4. 仅在此诊断恢复 time_out term，将环境 0 的 episode counter 设到 `max_episode_length - 1`，执行一个零动作控制步。要求只出现 `[0]` 的 timeout、恰好一批自动 reset、无其他终止，reset 边界仍作同样不变性核对。
5. 再执行一个零动作步，验证自动 reset 后第一次物理积分的观测、奖励、地面扫描有效，且没有额外 reset 或终止。

初始化 2 张票 + 直接 reset 120 张 + 自动 reset 1 张 = **123 张票**，严格覆盖一个完整 120-ticket block，并分别报告 3 张尾票；共 91 批被审计 reset。总计 22 个零动作控制步，其中前 20 步属于接触探针。

成功结果使用 `shared_bank_g2_passed`，监督器依据显式模式核对 G2 覆盖字段。短时 smoke 的 8 张票、5 批 reset 结果不能满足新模式。两个独立 JSONL 分别保存接触/整体诊断和完整 reset 生命周期；失败仍保存已有记录并由监督器收尾。

## 验证与边界

[回归日志](regression.log)：**418 passed, 318 warnings in 31.72s**，无跳过。准确命令、环境和测试期间源码哈希见 [regression_command.json](regression_command.json)。

- 实际共享事务分配器 + CPU 模拟 managers 完成全部 123 票和自动 reset，验证各块计数及 hook 恢复；注入自动 reset 污染另一环境、错误 timeout IDs、恢复后额外 reset 和非有限观测均失败。
- 原有 contact 集成测试增加 G2 分支，证明空接触或跨环境接触会在进入 G2 之前停止。此测试替代 G2 调用以验证派发；完整 G2 由上一项独立覆盖。
- 全新 Python 进程中的监督器 main 测试覆盖新参数组合、向子命令传参、manifest 类型、成功/失败汇总，禁止加载 Kit/任务包。模拟成功报告只验证监督器判定，不作为物理证据。
- 60 环境旧 G2、共享 smoke、接触采集与其余回归继续通过。
- 一次中间回归碰到既有 `/proc/.../stat` 读取与子进程退出的竞争：进程已消失时会抛 ProcessLookupError。测试探测函数现将此异常与 FileNotFoundError 一样视为进程不存在，没有修改生产进程清理机制。
- 318 条 warning 来自读取本机 tensor API docstring 的既有转义警告，增加的 3 条对应新增 G2 集成用例。

依赖清单从 32 扩为 34，新增本机 termination_manager 与 time_out 定义文件；原 32 项内容和哈希保持不变。源码确认 manager 在物理积分后增加 episode counter，再计算 timeout 并调用 `_reset_idx`，观测在 reset 后计算。6 项资产和历史实测归档均未改动，见 [validation.json](validation.json)。

**这些仍是离线验证。** 当前真实配置若没有非空 observation history，报告空清单，不把测试桩历史视为真实历史验证。只用两环境不能覆盖 60/120 环境的离散 ID、大批 reset 或性能；注入 timeout 不代表所有自然终止路径。GPU 接触 tensor 仅覆盖前 20 步，reset 阶段新增的两步没有成对接触采集，因此不以其结果宣称全过程物理隔离、运动策略资格或 G3/G4 通过。

## 待批准的一次实际检查

完成本阶段提交和推送后，冻结干净 revision，核对 34 项依赖和 6 项资产；GPU0 须无计算进程且可用显存至少 8 GiB。只运行一次，启动预算 120 秒、总预算 240 秒，清理宽限 10 秒。超时、异常、缺证据或任何验收失败立即停止；不自动重试、改卡、延长预算或扩容。

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --reset-lifecycle --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_g2_2env_20261028
```

本阶段未执行该命令。完成后必须复核原始 journal、完整结果/执行/outcome、控制台与 Kit 日志、GPU 和进程释放，并归档提交。准备阶段不包含真实运行授权，沿用用户要求“每完成一步向我汇报，并告诉我下一步要做什么，我同意后才可以进行”。真实检查完成后再决定共享布局扩容及 G3 冻结策略准备。
