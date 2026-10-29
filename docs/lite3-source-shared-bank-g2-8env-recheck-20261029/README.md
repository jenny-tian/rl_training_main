# 新版八环境 shared-bank G2 复验

日期：2026-10-29。结果：**本次八环境 G2 通过**。

用户批准后，从干净提交 `fba100c5ee15eb069f4a35ebd141bc2b0ecc50f3` 执行一次 GPU0 检查。使用 8 环境、1×15 shared bank、seed130196，可用显存准入线 12288 MiB，启动／总时限 120／240 秒。没有重试、加载策略或初始化 PPO。

本次运行的 144 项源码／资产清单、34 项依赖及机器人模型 revision，与[新版两环境 G2](../lite3-source-shared-bank-g2-2env-recheck-20261029/README.md)一致，并通过当前文件哈希验证。它补齐基座 scanner 周期改为 0.005 s 后的八环境启动、零动作接触及 reset 兼容性证据。完整冻结输入见 [manifest.json](manifest.json)，结果见 [outcome.json](outcome.json)、[result.json](result.json)。

| 检查项 | 结果 |
| --- | --- |
| 监督器退出／状态 | 0 / `diagnostic_smoke_passed` |
| 用时／环境构建 | 44.969775 s / 2.592817 s |
| GPU 采样峰值／退出后 GPU0 | 2822 MiB / 18 MiB |
| CPU 峰值 RSS | 4,069,648 KiB |
| 零动作接触探针 | 20 控制步、80 物理子步、81 次运行时状态校验 |
| env0–7 地面接触点计数 | 276、268、273、272、274、274、276、272 |
| 跨环境接触 | 0 |
| 接触拓扑 | 136 个机器人刚体、24 个地面碰撞体、每传感刚体 160 个过滤目标 |
| 接触容量／峰值使用 | 65536 / 61 |
| 直接 reset | 120 次，32 批 |
| 自动 reset | 仅 env7 timeout，随后一个恢复控制步通过 |
| 总分配／reset 批次 | 129 张连续票号 0–128 / 33 批 |

CPU 离线复核重建了全部 80 个接触矩阵，逐一核对原始力、法向、pair/net 力、计数和缓冲容量。独立分配器回放核对 8 张初始票、120 张直接 reset 票及 1 张自动 reset 票；32 个直接批次覆盖单环境、不连续／乱序及全环境 reset。未 reset 环境的分配不变，33 次边界检查通过，history buffers 为空。env7 timeout 输出及下一控制步均通过。详见 [post_run_review.json](post_run_review.json)。

本阶段没有修改生产代码，沿用冻结源码上已完成的 589 项 CPU 回归；没有为归档重复整套测试。9 个原始运行文件逐字节保存并校验，记录于 [execution_receipt.json](execution_receipt.json)。运行后按精确脚本参数匹配无残留进程，GPU0 无计算进程。GPU3 的既有他人任务未被本次运行使用或操作。

日志保留 **32 条足部视觉 USD 引用缺失警告**，没有 Error、Python traceback、致命异常、cooking failure 或超时记录。此次 G2 不证明视觉资源已修复、策略可以通过地形、运动中的隔离、逐子步扫描刷新时间或 G4 吞吐达标。reset 不变性比较仅限 `_reset_idx` 前后，排除物理积分。新版两环境与八环境 G2 通过仍不能授予完整 G3/G4、120 环境或 PPO 资格。

以下命令仅复核归档，不启动仿真：

```bash
CUDA_VISIBLE_DEVICES='' /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  docs/lite3-source-shared-bank-g2-8env-recheck-20261029/review.py
```

## 下一阶段待批准规格

建议开始补齐六类地形的冻结策略基线，先执行**一次多尺度粗糙 L1 单环境安全检查**，即原先 14 个待测用例中按 bank 列顺序的第一项。此次 G2 没有执行该用例。

- `case_id=multi_scale_rough_L1_forward_lane`，bank row0 / column1。
- geometry seed10001、initial-state seed20030、固定 difficulty `0.6294702331215974`。单格网格此前已与实际 bank 列精确匹配。
- 12×4 m 轨道，登记峰谷差候选 0.02 m、空间尺度 0.2/0.4 m；命令 `(0.3,0,0)`。
- 使用冻结 `model_1998.pt`，单环境、单回合，GPU0 至少 12288 MiB 可用显存，2000 控制步上限，120／240 秒启动／总时限。
- 通过条件仍为安全前进 9 m 并连续稳定 50 步，保持既有高度、姿态、接触及关节速度约束；失败保留证据并停止，不自动重试、换 seed、进入下一个用例或更新策略。

权威输入继续使用[已冻结准备清单](../lite3-source-followup-preparation-20261029/preparation.json)。已离线验证 225 项输入哈希、显式单回合规格和新输出目录，见 [next_case_review.json](next_case_review.json)。[next_stage.json](next_stage.json) 仅记录下一阶段提案，不授予执行授权。

从本归档提交后的干净工作树执行以下命令，须先获得下一阶段批准：

```bash
PYTHONPATH=source/rl_training /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/tools/source_policy_prequalification.py \
  --case-id multi_scale_rough_L1_forward_lane \
  --run-dir /tmp/lite3_source_policy_prequalification_multi_scale_rough_L1_forward_lane_bank_20261029
```

该单条件通过也不代表六族通过率或 shared-bank 运动隔离通过。有关平地 PPO 已有迁移能力与后续学习空间，沿用[六族证据审查](../lite3-flat-ppo-six-family-review-20261029/README.md)，本次未启动策略训练。
