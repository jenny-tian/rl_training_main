# 共享 bank 八环境 G2 离线准备

2026-10-29，基线提交 `e5dcae1c3f363722e5d812c6d04c5f4d6137e5a1`。本阶段把两环境 shared-bank 诊断扩展到八环境，并完成离线验证。**437 项测试通过；尚未启动八环境仿真。**

## 固定范围

本次只检查 GPU0、8 个环境、一个 shared bank 行、15 个等权地形列、12×4 m tile、地形 seed 10001、allocator/reset seed `130196`。固定种子的初始分配为：环境 0、1、7 共享 flat 第 0 列，环境 2 使用 `slope_rough_uphill` 第 8 列，环境 3 使用 `slope_downhill` 第 3 列，环境 4、5、6 使用 `stairs_descending` 第 7 列。初始重叠只用于接触与扫描探针，不代表运动隔离通过。

G2 生命周期覆盖：

- 初始分配 8 张票；
- 四种批次模式循环 8 次：单环境 `[7]`、不连续批次 `[6,1,3]`、乱序批次 `[4,2,0]`、倒序全批 `[7,6,5,4,3,2,1,0]`；
- 直接 reset 共 120 张票、32 批；随后只让环境 7 触发一次自动 timeout reset；
- 总计 129 张票、33 个审计 reset 批次、一个完整 120-ticket block 和 9 张尾票。

接触证据保留现有 GPU 成对 tensor 契约，八环境要求每个机器人刚体与全部机器人刚体及启用静态地面碰撞体形成显式过滤笛卡尔积，连续 20 个零动作控制步覆盖 80 个物理子步。两环境容量 16,384 保持不变；八环境冻结为 65,536，出现饱和、丢弃日志、成对力/净力不一致或任何跨环境接触都立即失败。八环境 shared-bank 不能省略 `--reset-lifecycle`。

## 离线验证

准确命令和环境见 [regression_command.json](regression_command.json)，日志见 [regression.log](regression.log) 与 [regression.junit.xml](regression.junit.xml)。结果为 **437 passed, 354 warnings, 0 failed**；warning 来自本机 tensor API docstring 的既有转义警告。

新增覆盖包括八环境全拓扑和最高环境 ID 的跨环境故障、八环境完整生命周期和票据尾块、报告中环境 7/拓扑/遮挡/尾票缺失的拒绝、shared-bank 八环境缺少生命周期标志的 smoke/supervisor CLI 拒绝，以及八环境 12,288 MiB 显存准入门槛。监督器 manifest 会记录本次实际门槛。

依赖审计仍为原 34 项，哈希 `00c464e37c4f4a66b3e6a2696ae76666cbbe49c1d801c7c064922676a70665b9`；机器人模型子模块固定为 `b452a1f404d37155400242f18e454fed8bcc51b8`。本阶段没有安装或修改依赖、没有加载策略、没有 PPO 更新。

## 待执行的一次实测

提交并冻结干净 revision 后，只允许执行一次以下命令：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --reset-lifecycle --num-envs 8 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_g2_8env_20261029
```

启动前必须重新运行 `nvidia-smi`，确认 GPU0 无计算进程、可用显存至少 12,288 MiB，且 `/tmp/lite3_shared_bank_g2_8env_20261029` 不存在。启动或总时限、接触容量、初始化、扫描、控制步、reset、报告验证或收尾任一失败都停止；不自动重试、不换卡、不延长预算、不扩容。该准备记录不授予 G3/G4、运动策略或 PPO 资格。

当前工作树仍有未提交改动，监督器会拒绝从 dirty worktree 启动；实测前必须记录最终提交和全部 manifest 哈希。
