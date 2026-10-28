# Lite3 Source 并行地形 importer/reset smoke 方案

版本：v0.1 draft，2026-10-27
状态：`diagnostic_completed_limited`
执行授权：`false`

本方案只检查默认关闭的 Source 并行地形 importer 和异步 reset 接线，不运行策略、不评估完整 episode、不启动 PPO、不更新权重。2 环境 Isaac Sim smoke 已完成；结果不构成 120 环境资格，执行授权仍为 `false`。

## 1. 目标与边界

使用 2 个 Lite3 环境构建真实的 2×15 Source L1 地图，检查环境启动时的独占 terrain row/origin、首个零动作物理步、两类高度扫描，以及仅 reset 环境 0 后的行列/origin/root-state 更新。2 环境只用于接线 smoke，不足以证明 120 环境的物理隔离、显存、吞吐或训练资格。

脚本：[source_parallel_terrain_reset_smoke.py](../scripts/tools/source_parallel_terrain_reset_smoke.py)。它通过显式 config 参数打开并行 sampler，固定 seed `130001`，执行一次零动作 control step，再调用 ManagerBasedRLEnv reset 路径处理环境 0。脚本不加载 checkpoint、不创建 PPO runner，输出中标记 `qualification_claim=false`。

## 2. 运行前检查

1. 本次执行前从独立 worktree 运行；对应离线测试已通过，拟选 GPU 无其他 Isaac Sim 实例占用。
2. 只使用 2 个环境；不可在此 smoke 中改成 120。确认生成配置含 2 行、15 等权列，`terrain_origins.shape == (2, 15, 3)`，并行 sampler seed 显式记录。
3. 本次运行前 GPU0 与 GPU1 均空闲；最终在 GPU0（RTX 3090）执行。资源快照会变化；后续资格运行前重新检查拟选 GPU，若显存余量低于 8 GiB、出现冲突仿真进程或设备不可用，则停止并重新安排资源。
4. 使用 headless 模式。结果和 trace 写到新的路径，禁止覆盖已有结果。

本次实际命令：

```bash
cd /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
PYTHONPATH=source/rl_training python scripts/tools/source_parallel_terrain_reset_smoke.py \
  --headless --device cuda:0 --num-envs 2 --seed 130001 \
  --output /tmp/lite3_source_parallel_reset_smoke_20261027_gpu0_retry3.json \
  --trace-output /tmp/lite3_source_parallel_reset_smoke_20261027_gpu0_retry3.trace.log
```

## 3. 通过条件

- 环境启动成功，实际环境数为 2，origin grid 恰为 `(2, 15, 3)`，所有 origin 有限。
- 首次全量 reset 为两个环境分配不同 row，`terrain_levels[i] == i`；列在 `[0, 14]`，每个 `env_origin` 等于对应 `terrain_origins[row, column]`。
- 机器人根位置与 default root state 加分配 origin 对齐；首个零动作控制步的观测、回报、终止标记和两个 height scanner ray hits 全部有限，且没有环境意外终止。
- reset 只覆盖环境 0 后，环境 0 消费恰好一张票并继续使用 row 0；环境 1 的票据、origin 和 root position 保持不变；环境 0 的新 root position 使用更新后的 origin。
- 输出 JSON 完整记录 seed、设备、环境数、初次分配、异步 reset 分配及各断言状态，且不声称完成 qualification。

## 4. 停止条件与后续门槛

遇到重复/非有限 origin、root 不在对应 origin、异步 reset 改动未 reset 环境、首次物理步发生终止、scanner 产生非有限 hit、GPU 余量不足、初始化超过 120 秒或脚本异常时，立即停止并保留日志；不自动重试、不扩成更多环境、不转入训练。

通过此 smoke 仅表示 2 环境的 importer/reset 接线和首个物理步正常。之后还需单独方案检查邻接 tile 越界、机器人活动范围、碰撞过滤、完整 reset 传感器/命令历史、120×15 地图生成时间与 GPU 显存、并行吞吐和 checkpoint 调度状态恢复。该 smoke 不能替代上述资格，也不自动授权训练。

## 5. 运行记录

2026-10-27 在 GPU0、seed `130001` 上运行成功，进程退出码为 0。结果：[归档 JSON](lite3-source-parallel-terrain-allocation-runtime-smoke-gpu0-2026-10-27.json)；完整阶段 trace 保存在 `/tmp/lite3_source_parallel_reset_smoke_20261027_gpu0_retry3.trace.log`。

实测 origin grid 为 `(2, 15, 3)`。初始环境 0 分配到 row 0 / column 4（`slope_cross_slope_left`），环境 1 分配到 row 1 / column 1（`multi_scale_rough`）；一个零动作控制步后只 reset 环境 0，它分配到 row 0 / column 0（`flat`），环境 1 的票据、origin 和 root position 保持不变。root/origin 对齐、两类 height scanner 有限输出及异步 reset 断言均通过。未加载策略，PPO 更新数为 0。

运行日志有 Lite3 足部视觉 USD 引用缺失警告；smoke 检查的是物理启动/reset、有限传感输出和位置不变量，不据此判断视觉资产完整性。诊断未检查长时运行、实际碰撞隔离、120×15 地图资源或并行吞吐。
