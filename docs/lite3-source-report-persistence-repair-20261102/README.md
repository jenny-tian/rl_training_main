# 报告持久化修复：CPU 验证完成，仿真重跑待同意

状态：`cpu_persistence_repair_passed_runtime_confirmation_pending`，2026-11-02。

用户批准的本阶段仅包含代码修复和 CPU 验证，不包含仿真重跑。
本阶段没有启动 Isaac Sim 或 GPU 任务，Source 训练 transitions / 更新均为 0。
此前唯一一次运行仍保持 `incomplete_runtime_artifacts`，不追改成通过，不补造记录。

## 完成的修改

### 关闭前持久化

`source_scan/smoke_artifacts.py` 为只依赖标准库的写盘与验收模块。
每次 JSON 更新先完整序列化，再写同目录临时文件，flush/fsync 后原子替换，
最后 fsync 目录。无效 JSON 或替换前写入失败不会破坏上一份完整 JSON。
已存在的结果或 sidecar 一律拒绝覆盖。

`source_training_integration_smoke.py` 在调用环境和应用关闭回调之前保存最终结果。
`result.json`、`result.steps.jsonl`、`result.cleanup.json` 各自承担结果、逐步原始记录
和清理状态，运行身份、代码 revision、checkpoint SHA 和 journal SHA 相互关联。
子进程日志只表示仿真断言执行完成，最终通过与否由父进程验收决定。

### 每步追加证据

每个控制步追加并 fsync 一条 JSONL，之后原子更新进度快照。
记录 raw / base / requested / executed action、projected / paused、done、timeout、
episode counters、reset 前后 terrain columns；结束步保存完整 terminal observations、
终止掩码、IDs、步数、hold、completed 和活动结束项。

因此中途退出仍能保留此前完成写盘的记录。若恰好在物理步执行后、写盘完成前中止，
该步仍可能没有完整记录；监督器会拒绝不完整结果，不把这种情况当作成功。

### 清理状态和父进程验收

环境和应用分别写入 `closing` / `returned` / `failed` 状态。
Kit 可以在应用 close 内结束进程，所以父进程允许应用状态为 `closing`，但同时要求
环境 close 已返回、退出码为 0、进程组已清空、结果及逐步证据完整、无持久化错误。
这表示进程在请求关闭后已退出，不声称应用 close 在 Python 层返回。
写盘或清理异常会使结果失败，且仍尝试剩余清理回调。

新增 `source_training_integration_supervisor.py`：

- `validate` 仅离线读取已有文件，不启动 GPU/Isaac。
- `run` 只允许显式同意、全新输出目录、干净提交、已核对远端 SHA 和输入哈希的
  2 环境/24 步/零 PPO 更新方案；检查 GPU0 资源后仅启动一次，不自动换卡或重试。
- 父进程核对 24 条连续且同身份记录、动作形状与有限性、纯 timeout 快照及终止观测，
  并从每步地形列重算每单元 transition/reset 归属，与 exposure 台账逐项比对。
- 缺文件、`starting` 报告、零退出码下缺记录、身份/seed/hash 不符、清理失败、
  非零退出或残留进程组都不能通过。资源监控不完整也不接受。

策略、checkpoint、环境几何、seed、噪声标准差、安全阈值、24 步上限及 timeout
注入方式均未更改。本阶段只修改诊断的记录与验收路径。

## CPU 验证结果

**220 passed**：新增持久化/监督器测试 29 项，原有训练环境、安全、预算、
并行分配/reset、tile bounds 和动作约定回归 191 项。见 `focused.junit.xml`。
测试命令设置 `CUDA_VISIBLE_DEVICES=''`，不启动真实模拟器。
原有训练回归包含一次合成数据 CPU PPO 更新，仅验证更新链路，不是 Source 训练。

关键证据：

- CPU 子进程在应用关闭回调直接 `os._exit(0)`：完整结果、24 条动作/终止记录存留，
  验收通过，清理状态明确标为 application `closing`。
- 在第 5 步 `os._exit(17)`：已 fsync 的 5 条记录及 running 进度保留，验收拒绝。
- 退出码 0 但文件缺失或仅为 starting：拒绝。
- 环境关闭异常、环境关闭直接退出、journal 关闭失败：拒绝，证据保留。
- 原子替换失败或非有限 JSON：保留旧的完整文件；写盘失败仍尝试两个清理回调。
- 修改终止观测、地形计数归属、seed、身份、revision、checkpoint 或 journal：拒绝。
- 通过假启动器走实际 smoke main() 的 finally 路径，启动失败后应用直接退出 0，
  最终失败报告在退出前保存。该用例从未导入真实 Isaac 或构造场景。
- 新监督器对上一轮真实归档的只读检查拒绝验收，历史原文件哈希不变。

`cpu_examples/` 是从本次 pytest 临时目录复制的**合成 CPU 数据**，不是新仿真数据；
`cpu_audit.json` 记录测试数量、示例来源、源码哈希和历史归档拒绝结果。
代码语法、`git diff --check` 及历史原始归档 SHA 校验通过。

这些结果验证文件持久化及验收逻辑，**不替代真实 Kit 重跑**。
fsync/原子替换增加了诊断 I/O，这一短探针不能用于训练吞吐结论。
本阶段也未完成 G3/G4、120 环境资格检查或正式训练入口。

## 下一阶段提案：新目录执行同等范围检查，等待同意

只提议一次新的 2 环境、1 行 shared bank、24 全局控制步（48 个诊断 transitions）
检查。继续运行 1 的登记 seed、actor 均值加 std=1 独立噪声；第 24 步仅对 env 0
注入 timeout。GPU0 启动前重新核验至少 12288 MiB 可用，总超时 240 s，清理宽限 15 s。

新输出目录：`/tmp/lite3_source_training_integration_smoke_v2_20261102/`。
旧目录和历史归档保留。新一轮也不创建 PPO/optimizer，不训练、不扩大规模，
失败不重试；结束后归档汇报，再等待用户批准下一阶段。

`next_stage.json` 仍为未批准提案。获得同意后，在新 run manifest 中填入本阶段
最终提交、远端核对 SHA 和代码哈希，再调用新监督器：

```bash
PYTHONPATH=source/rl_training \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/source_training_integration_supervisor.py run \
  --manifest /tmp/lite3_source_training_integration_smoke_v2_manifest_20261102.json \
  --output-dir /tmp/lite3_source_training_integration_smoke_v2_20261102 \
  --checkpoint /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt
```

该命令本阶段未执行。当前完成修复、测试和归档后暂停。
