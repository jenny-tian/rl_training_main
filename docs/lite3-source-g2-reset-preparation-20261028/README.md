# G2 六十环境异步 reset 检查准备（2026-10-28）

状态：`offline_prepared_runtime_pending_user_approval`。

承接用户在会话 `01a180fe-c866-7c52-935e-b25dbb6d7878` 的要求：“每完成一步向我汇报，并告诉我下一步要做什么，我同意后才可以进行”。本阶段只完善诊断与离线测试；没有启动 Isaac Sim、共享地形运行、策略或 PPO。G1 最大启动通过规模仍为 60，G2 尚未通过。

## 本次实现

沿用 G1 的启动检查和监督器，通过显式 `--reset-lifecycle` 开启 G2；仅接受 60 环境。默认 G1 仍执行一个零动作步。G2 使用现有 reserved-row 地图，L1、60×15 origins、地形 seed 10001、环境与 sampler seed 130001。

新检查器 `scripts/tools/source_parallel_reset_lifecycle.py` 在首次 reset 和首步成功后执行：

1. 单环境 `[0]`，不连续且输入乱序的 `[59,7,41,1,19]`，六批轮转的 45 环境，最后 24 环境；共 9 批、300 次直接 reset。
2. 用 `env.step()` 触发 `[2,23,58]` 的自动超时 reset，要求恰好一批、无其他终止。
3. 加上初始 60 次分配，共 363 张票据；逐项比对同种子的独立 allocator 重放，核对 3 个完整 120-ticket 块的 15 类地形计数，另外报告 3 张尾票，不把尾块当完整配比。

Source 当前关闭 time_out。诊断启动器仅在 G2 模式恢复 Isaac Lab 的 time_out term，最后将三个指定环境的 episode counter 设为 `max_episode_length - 1`，再执行零动作步。该检查覆盖管理器自动 reset 路径，不代表自然跌倒、完成地形或长回合运行已经通过；生产配置文件没有改变。

在每次 `_reset_idx` 紧邻的调用前后，检查未 reset 环境的 assignment、row/type、origin、完整 root state、关节位置/速度、命令、action/prev_action、episode counter、已配置的 observation history 及 sensor 原始缓存/时间戳不变。检查被 reset 环境的 origin/root 对齐、状态有限、动作/episode/history 清零、接触力历史与接触计时清零、传感器失效标记置位。随后刷新两类高度扫描，检查有限性与失效标记消除，重新计算观测。

观测历史检查依运行时实际 buffer 清单执行；若配置未启用历史，结果明确记录空清单，不声称验证了非空策略历史。传感器快照读取 `_data` 原始缓存，不触发 lazy update。reset 边界比较排除物理积分期间的正常运动；自动 reset 的其他环境不会被要求跨整个控制步保持位置不变。传感器有限性不等于 ray 命中归属或运动隔离通过。

每批开始即追加 journal，成功后写入票据、选中环境的 reset 前 root/origin/命令，以及所有快照 buffer 的前后 SHA-256。失败保留已完成记录和 trace。监督器要求 363 张票据、3 完整块、300 次直接 reset、指定 3 个自动 reset 和共 10 批全部满足，才接受结果；沿用碰撞错误实时停止、日志收尾扫描、进程组清理、GPU 采样和代码哈希检查。

## 离线验证及限制

命令使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/rl_training` 与 `/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest`。关闭插件自动加载是因为本机无关的 typeguard pytest 插件无法导入当前 typing_extensions 的 NoExtraItems。

- 新测试覆盖批次规模、票据连续性/配比/尾块、错误票据注入、逐项未 reset buffer 污染、选中环境清理失败、非有限状态、lazy sensor 无副作用快照、history 批次维度，以及监督器拒绝不完整报告。
- 全目录收集发现既有 `test_source_family_runtime_allowlist.py` 引用缺失的 `scripts/tools/runtime_progress.py`。已经确认基线 HEAD 同样引用该文件且没有跟踪它；该模块不计入本轮通过数，没有为此修改其他工作流。
- 合并运行其余测试时，347 项通过，地图生成测试因其他测试留下的模拟 Isaac 模块触发“不加载仿真模块”断言。地图生成测试随后在独立 Python 进程执行；其余模块也单独重跑，见日志。该测试隔离问题没有被描述为真实仿真启动。
- Python 编译和 `git diff --check` 通过。

最终分组结果为 **346 + 2 = 348 项通过**，见 [回归日志](regression.log) 和 [独立地图生成日志](generator-isolated.log)。以上属于离线检测能力与既有逻辑回归，不证明真实 PhysX reset 行为。

## 等待用户确认的下一步

只执行一次 60 环境 G2 检查。GPU0 运行前重新查询，要求无计算进程且可用显存至少 8 GiB；工作树/模型干净、资产哈希一致。启动时限 120 秒，总时限拟为 240 秒，额外时间用于多批 reset 和审计输出；时限触发后仍可能需要监督器最多 10 秒清理宽限。失败即停止，不自动重试、延长预算或扩容。

待批准命令（本阶段未执行）：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --reset-lifecycle --num-envs 60 --seed 130001 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_g2_60env_reset_20261028
```

执行前冻结届时干净提交；manifest 记录代码和模型 revision、诊断代码哈希、设备快照与命令，输出目录必须尚不存在。验收还须核对退出后的 GPU/进程清理、完整控制台和 Kit 日志，并将结果归档。无论通过或失败都先汇报，再请求下一阶段授权。

G2 之后的共享地形隔离、小规模扩容、冻结策略运动检查、吞吐与 PPO 仍需后续逐步确认。本检查不覆盖 checkpoint 调度恢复、实际运动隔离、自然终止的所有路径或训练能力。
