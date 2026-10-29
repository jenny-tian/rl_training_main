# 新版两环境 shared-bank G2 复验

日期：2026-10-29。结果：**本次两环境 G2 通过**。

用户批准后，从干净提交 `401ea45d758f03c575abecc8ac4615773678fb4d` 执行一次已准备的 GPU0 检查。使用 2 环境、1×15 shared bank、seed130196，可用显存准入线 8192 MiB，启动／总时限 120／240 秒。没有重试、加载策略或初始化 PPO。

本次更新了基座扫描周期改为 0.005 s 后的启动、零动作接触与 reset 兼容性证据。运行入口、参数、模型及源码／依赖哈希见 [manifest.json](manifest.json)，实际结果见 [outcome.json](outcome.json) 和 [result.json](result.json)。

| 检查项 | 结果 |
| --- | --- |
| 监督器退出／状态 | 0 / `diagnostic_smoke_passed` |
| 用时／环境构建 | 21.473603 s / 2.634263 s |
| GPU 采样峰值／退出后 GPU0 | 2820 MiB / 18 MiB |
| CPU 峰值 RSS | 4,087,232 KiB |
| 零动作接触探针 | 20 控制步、80 物理子步、81 次运行时状态校验 |
| 接触阳性对照 | env0 276、env1 268 个地面接触点计数 |
| 跨环境接触 | 0 |
| 接触拓扑 | 34 个机器人刚体、24 个地面碰撞体、每传感刚体 58 个过滤目标 |
| 接触容量／峰值使用 | 16384 / 8 |
| 直接 reset | 120 次，90 批 |
| 自动 reset | 仅 env0 timeout，随后一个恢复控制步通过 |
| 总分配／reset 批次 | 123 张连续票号 0–122 / 91 批 |
| 冻结输入 | 144 项源码／资产清单和 34 项依赖哈希均一致 |

CPU 离线复核重新运行项目报告校验器，并从逐子步的接触力与法向重建全部 80 个接触矩阵，核对独立 pair/net 力、计数及缓冲容量。从逐 reset 日志重新回放分配器，票据与登记顺序一致，未 reset 环境的分配不变；91 次边界检查通过，history buffers 为空。详见 [post_run_review.json](post_run_review.json)。本阶段没有修改生产代码，因此沿用冻结提交上已完成的 589 项 CPU 回归；没有为归档重复运行整套测试。

运行后 GPU0 无计算进程，按精确脚本参数匹配未发现本次运行残留进程。GPU3 的既有他人任务没有被本次运行使用或操作。原始九个运行文件逐字节归档，SHA-256 在 [execution_receipt.json](execution_receipt.json) 中核对；资源与进程快照另存。

日志仍有 **8 条足部视觉 USD 引用缺失警告**。没有 Error、Python traceback、致命异常、cooking failure 或超时记录。本结果不证明视觉资源已修复，不测策略通行能力，也不证明运动中的隔离、逐子步扫描刷新时间或 G4 吞吐达标。reset 不变性比较仅限 `_reset_idx` 前后，排除物理积分。

可用以下命令复核已归档证据，不启动仿真：

```bash
CUDA_VISIBLE_DEVICES='' /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  docs/lite3-source-shared-bank-g2-2env-recheck-20261029/review.py
```

下一阶段建议按原准备方案执行**一次新版八环境 G2**，等待用户单独批准：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --reset-lifecycle --num-envs 8 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_g2_8env_height_recheck_20261029
```

八环境使用 12288 MiB 显存准入线，预计核对 129 张票、33 批 reset、env7 自动 timeout 及恢复步。运行前核对干净提交、相同运行输入及新输出目录。此前准备目录的 `not_run` 字段和输出目录空闲断言属于当时快照；已完成的两环境结果不应删除或覆盖。八环境、剩余单格策略检查、完整 G3/G4、120 环境和 PPO 均未自动获得授权。

用户关于平地 PPO 六类地形能力与新步态学习的问题，已结合不同工作树保留的历史报告单独整理为[六族证据与学习空间审查](../lite3-flat-ppo-six-family-review-20261029/README.md)。
