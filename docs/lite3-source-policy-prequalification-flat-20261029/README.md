# Flat L1 单环境策略安全前置检查

日期：2026-10-29。冻结提交：`3b79cdda6574d9c7936437243f980166b65e2614`。

**结果：本次单条件安全检查通过；G3/G4 仍未取得资格。** 用户同意前一阶段提出的单次 GPU0 检查后执行了一次，没有重试、扩容或 PPO 初始化。

## 运行条件与结果

使用冻结的 `model_1998.pt` actor，checkpoint SHA-256 为 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。环境为一个 12×4 m Flat L1 tile，geometry seed 10000、initial-state seed 20000，命令为 `vx=0.3 m/s, vy=0, yaw_rate=0`。物理步 0.005 s、控制步 0.02 s，最多 2000 控制步；完成条件为前进 9 m 后连续稳定 50 步。

GPU0 启动前占用 18 MiB、可用 24,558 MiB，无计算进程，满足 12,288 MiB 门槛。按启动 120 s／总计 240 s 预算运行，监督器和 worker 均退出 0，没有超时、fatal exception 或 cooking failure。

| 指标 | 本次结果 |
| --- | --- |
| 控制步／模拟控制时间 | 1514／30.28 s |
| 首次到达 9 m | 第 1465 步 |
| 最后连续稳定步数 | 50 步，重建结果与报告一致 |
| 最终前进距离 | 9.295170 m |
| 最小相对地形安全高度 | 0.308404 m |
| 最大 projected-gravity XY 范数 | 0.065376 |
| 最大关节速度 | 17.300005 rad/s |
| 安全拒绝／无效状态／非足端接触事件 | 均为 0 |
| 动作投影／权重更新 | 均为 0 |
| 横向偏移峰值 | 0.719969 m |
| 最终 yaw | -0.286225 rad |
| 速度误差均值／RMS | 0.029396／0.033508 m/s |
| 监督器耗时 | 88.228709 s |
| GPU0 采样显存峰值／退出后占用 | 2,750／18 MiB |

运行时读回确认全部 17 个刚体被 guard 覆盖，基座扫描周期为 0.005 s，正常终止集合和 guard/接触/姿态阈值与冻结条件一致。最终通过 `completion` 结束，没有触发 manager 终止或自动 reset。

## 归档与独立复核

本目录保留全部 9 个原始运行文件，其内容与 `/tmp/lite3_source_policy_prequalification_flat_20261029` 一致。实际命令、单次授权和主机进程核对见 [execution_receipt.json](execution_receipt.json)，监督结果见 [outcome.json](outcome.json)，回合与完整动作轨迹见 [result.json](result.json)，逐步状态见 [samples.jsonl](samples.jsonl)。

[review_run.py](review_run.py) 仅在 CPU 上复核归档。它重新核对 185 项冻结输入、9 个原始文件哈希、1514 条连续状态与动作记录、动作裁剪和执行值、每步安全指标、完成及 50 步稳定条件、摘要数值和运行后 GPU 状态。检查通过，结果见 [post_run_review.json](post_run_review.json)。重跑复核的命令为：

```bash
CUDA_VISIBLE_DEVICES='' /home/tjx/anaconda3/envs/env_isaaclab/bin/python \
  docs/lite3-source-policy-prequalification-flat-20261029/review_run.py
```

actor 的运行前后张量哈希均为 `aeeeb55466a7f7369e7bc041708201c0a965779e6240a7be8751fe75c1b9a0d4`。GPU0 运行后无计算进程，主机 `pgrep` 未发现本次 Python supervisor 或 worker。GPU3 原有其他用户进程仍在，本次未操作该进程。

## 证据边界与记录说明

- 横向偏移达到 0.72 m、最终 yaw 约 -0.286 rad，未触发本次登记的边界或安全条件；本结果不等于直线跟踪质量通过。后续应审查横漂，不能事后添加新阈值改写这次结果。
- 本次仅一个 Flat L1 回合，没有覆盖其他地形、shared-bank 运动、不同地形列、物理子步跨环境接触、全刚体运动轨迹或异步 reset。新版扫描配置的 shared-bank 启动/reset 仍需重新取得实测证据。
- 日志中保留了 4 条足部视觉 USD 引用缺失警告，以及 CUDA 可见设备/枚举提示和 6 条 GPU skip 警告。运行日志将 GPU0 标为 active，资源采样确认 GPU0 使用与释放，未出现 Error、Traceback 或监督器 fatal 标记；本次未解决这些警告的一般原因或验证视觉资源完整性。
- 原始 `manifest.json` 由准备记录展开，继承了准备阶段的 `status`、`simulation_run=false` 和 cell 的 5-episode 种子范围字段。这些是准备元数据，不代表实际执行状态或本次运行数量。本次实际授权和单次命令在 `execution_receipt.json`，实际 1 回合结果在 `result.json`/`outcome.json`；原始 manifest 按采集内容保留，未改写。
- 沙箱内只读 GPU 查询无法访问驱动，改用主机只读查询后成功；真正的监督运行只启动了一次。该前置资源查询不计为仿真重试。

下一阶段建议先离线审查横向漂移与记录字段，冻结其余地形的单环境安全检查顺序，并准备新版 shared-bank 启动/reset 复核范围。本阶段在归档和推送后停止，等待用户同意；本次结果不授权 G3/G4、训练或下一次 GPU 运行。
