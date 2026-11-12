# 低台阶基础模型正式训练

用户在低台阶准备阶段完成后明确批准正式训练，并询问耗时。本轮执行此前提交的配置，不改变奖励、地形或 PPO 参数。代码与验证见 [准备报告](../lite3-low-stairs-preparation-20261112/README.md)，准备版本为 `07e878b`。

## 已批准的预算

- GPU 1，4096 环境，种子 42，actor、critic 和 Adam 从随机初始化开始。
- 六类地形原生分块；上下台阶每级 2–5 cm，初始课程行 0，踏面 30 cm。
- `terrain_profile=low_stairs`、`reward_variant=low_stairs`，actor 45 维、critic 393 维。
- 固定 10000 次 PPO 更新，每次 24 个控制步，共 983,040,000 次环境交互，每 100 次保存检查点。
- 不加载旧检查点或 3 次启动验证的临时模型；6–8 cm、10 cm 台阶不进入本轮训练。
- 监督器最多运行 48 小时，达到固定训练预算、墙钟上限或检测到运行错误即停止；不会自动延长或重新启动。
- 完成训练后详细汇报，再经用户确认进入正式行为评估。其他项目进程保留。

## 耗时解释

原六类地形从头训练 10000 次实际耗时 53180.67 秒，约 14.77 小时；原训练最后 50 次的平均采样加优化时间为 5.64 秒/次。最近的核心奖励修正版是 500 次续训，实际 56.70 分钟，最后 50 次约 6.62 秒/次。这次的更新预算是该续训的 20 倍，不能直接比较两次总耗时。

准备阶段仅有 3 次更新，每次采样加优化为 14.03、13.26、10.09 秒；据此给出的 28–38 小时只是启动阶段粗估，不能视为稳定速度。新奖励每个控制步增加足端地形射线和有效落脚状态计算，GPU 1 同时有其他项目的仿真进程。这些因素可能增加耗时，但目前没有分项计时，不能将全部差额归因于新增奖励或共享负载。启动后用实际完成的更新估算剩余时间，并注明样本窗口。

## 运行与状态

运行目录：`logs/reference_six/low_stairs_training/scratch_seed42_10000/`。

训练命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/reference_six_train.py \
  --mode train --updates 10000 --seed 42 --gpu 1 \
  --terrain-profile low_stairs --reward-variant low_stairs \
  --output logs/reference_six/low_stairs_training/scratch_seed42_10000
```

监督器独立于终端运行，日志保存在输出目录的 `runtime.log`；`execution.json` 记录运行结果，`report.json` 在每次更新后保存进展。最终检查点必须同时核对 `completed_updates=10000`、`phase_updates=10000`、`from_scratch=true`、`validation_only=false`，Adam 步数 200000，及完整模型 SHA256。

本文件先登记批准范围。启动状态与速度快照将在确认训练正常更新后写入同目录 `startup_status.json`；运行中的记录不表示训练或能力验收已经完成。
