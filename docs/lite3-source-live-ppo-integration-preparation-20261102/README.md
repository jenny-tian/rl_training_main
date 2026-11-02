# 八环境真实 Source PPO 联调准备

状态：`prepared_cpu_verified_waiting_for_live_launch_approval`，2026-11-02。

用户批准了本阶段的实现和离线验证。本阶段真实Isaac启动0、GPU任务0、真实Source
交互/更新0；没有创建生产预算台账或预留预算。以下真实运行仍须单独批准。

## 已完成

- 专用`source_live_ppo_probe.py`：原始登记actor-only初始化，std=1；fresh critic、
  optimizer和计数。policy45/critic393，保留六家族和15个L1 cell。第二个rollout
  使用第一次更新后的actor/critic，完成后停止，不接正式训练或自动扩展。
- 通用runner增加可选观察回调，逐控制步保存原始/执行动作、log probability、
  critic值、原始和bootstrap后奖励、终止前后观测、reset和曝光。无观察回调时，
  原有收集/更新路径保持；回归继续覆盖原版PPO数值一致性。
- 每次更新前保存完整rollout（含returns、advantages和分布参数），初始及每次
  更新后保存actor/critic/optimizer快照；这些仅为诊断证据，不选择或发布模型。
- 复用逐子步接触、运动、扫描和reset审计。原运动验收器的步数参数默认仍为96；
  本次显式使用48，要求192个子步。检查跨环境接触、地面接触阳性、力重建、tile
  与scanner归属、reset快照和可观测的恢复首步，不把5ms离散记录当连续时间证明。
- 父进程在CPU上重建动作分布/log probability、critic和terminal值、timeout
  bootstrap、GAE returns/advantages；核对保存动作、Adam参数step、模型变化及
  checkpoint。每项原始文件绑定SHA，篡改文件并重算SHA也不能绕过数值关系检查。
- 新监督器拒绝未批准清单；固定设备、步数、更新数、时限、输入及依赖。启动前
  检查资源，监督运行和清理，退出后再次核对输入。退出0但证据缺失仍拒绝成功。
- 预算台账位于Git公共元数据目录，所有worktree共享，不污染其他工作区。采用
  非阻塞文件锁和原子落盘；启动前整笔预留，换输出目录或worktree不能重复启动。
  正式入口也检查此台账并预留预算，已花费的run不能套用旧的从零开始清单。

## 待批准的唯一一次真实运行

| 项目 | 固定值 |
|---|---|
| 用途 | 真实Source PPO端到端诊断，非正式Pilot或技能资格 |
| 设备 | GPU0，cuda:0；不自动换卡 |
| 环境/共享bank | 8环境，1行，保留六类/15个L1 cell |
| rollout/PPO | 24步×2次完整更新，48全局控制步，384真实transitions |
| 更新内部 | 5 epochs×4 minibatches，共40次Adam step |
| 物理 | 200Hz物理/50Hz控制，192个子步，固定前进命令0.3m/s |
| 随机种子 | run1；PPO130001、geometry30000、sampler131001 |
| reset标识 | 40000，仅bookkeeping；当前reset不随机化 |
| 初始化 | 原始model_1998.pt，仅加载actor均值；std1，全新critic/optimizer |
| 注入超时 | 第12步env[0]；第36步env[1,4,7]，counter预置1999 |
| 资源 | 启动前GPU0至少12288MiB可用；原始记录磁盘至少2GiB可用 |
| 时限 | 启动120秒、进程240秒、TERM后15秒强制清理 |
| 接触数据 | 136个机器人刚体，所有ground目标，capacity65536；事件上限1GiB |
| 再执行 | 不自动重试、不resume、不追加交互或更新 |

任务/安全失败正常结束对应回合并继续采样；输入非有限、跨环境接触、实际包络越界、
扫描归属错误、pre-action pause、数据或计数不完整等为硬停。强制超时可能与失败
同时出现，此时不加timeout bootstrap。普通失败数本身不是联调失败条件。

验收必须同时具备：48条逐步记录、192次物理采样、两份完整rollout、初始及两次
更新后的模型状态、两次完整PPO/40次Adam更新、全部数值关系一致、完整退出证据。
最后一步发生reset时，没有下一子步可观察的恢复不能声称已验证。
应用`closing`只在子进程正常退出、进程组为空且其他记录完整时接受，不声称Python
层app.close返回。监督器只清理本次进程组，卡级显存包含其他任务，不当作独占占用。

本轮保留短探针范围：不能据此通过完整G3/G4、120环境运行、长期稳定性、训练吞吐、
六家族技能或Source-only选模资格。不读取Development/Sealed反馈；旧平地策略失败
仍是学习目标，不另设“旧策略必须全部通过”的门。

## 384条交互如何计入既有预算

这次诊断使用正式run1 seed130001，**384条全部占用既有上限**，不增加任何硬上限。
GPU/磁盘门通过后、创建子进程前整笔预留384。即使启动失败、只走了一部分或关闭
异常，也保留整笔预留；实际尝试/返回/存储交互与完成更新数另外报告，不自动退回。
准备阶段未写生产台账，当前已预留数为0。

| run1节点 | 原硬上限 | 预留384后余额 | 最多剩余120×24完整rollout | 合计计费交互 |
|---|---:|---:|---:|---:|
| numerics | 262144 | 261760 | 90 | 259584 |
| midpoint | 1048576 | 1048192 | 363 | 1045824 |
| pilot | 2097152 | 2096768 | 728 | 2097024 |

原91/364/728是从零开始的候选节点，不能在预留384后照搬给run1。现有正式入口
因此明确拒绝已有预算的run，后续须提交修订的累计锁；不会擅自自动改为90/363。
run2/run3本次不花费。此前不更新权重的冻结策略探针保留为诊断记录，本次是首次
拟带真实梯度更新的联调。即使不采用诊断后权重继续训练，也不删除其预算消耗。

## 已执行的离线验证

整组回归 **461 passed，0 failed，0 skipped**，原433项加本阶段28项；约203.68秒。
随后仅将台账移入公共`.git`元数据目录，新增跨worktree路径测试；针对这一调整
复核 **42 passed，0 failed**（含1项新增测试，其余41项为重跑）。合计覆盖462项
不同测试，不把重跑重复计数。完整命令、JUnit和日志在`raw/`。

测试覆盖：预算并发锁、陈旧/损坏台账、部分失败不退预算、正式入口拒绝已花费seed、
未批准/增量/改seed/改设备/改时限拒绝、超时与空证据退出0拒绝、本次进程组清理、
完整两次合成PPO更新、普通失败继续采样，以及篡改动作、bootstrap、GAE、模型、
Adam step或缺失快照等拒绝路径。整目录历史缺失runtime_progress的范围限制沿用
上一阶段；这里没有声称整目录全绿。

`cpu_example/`是明确标记的CPU合成示例：随机初始actor/fresh critic，生产安全桥、
生命周期、PPO和记录回调，假物理张量。其“cuda:0”后端字符串是模拟元数据，不表示
GPU已运行。示例完整经过48步、2次合成更新，包含一次普通失败和两批强制超时。
实际入口则强制登记checkpoint初始化。合成示例只有使用显式`allow_synthetic=True`
才能通过离线fixture验收，实际监督器不开放该选项。

示例保留模型、optimizer、rollout、控制记录和无损gzip事件。`review.py`会从归档
副本再次执行数值/物理记录验收，只用CPU，不重新执行仿真或PPO更新。
225项生产/资产输入与44项已安装核心依赖已登记，冻结运行提案时再检查一次。
360条回归warning来自既有PhysX文档字符串转义；pytest自动插件关闭，依赖未修改。

## 冻结与执行命令

工作目录为本worktree。提交、私有仓库同步并核对后，准备命令生成**未批准**清单，
绑定当时精确HEAD、225项输入、44项依赖、模型和预算台账快照，不查询GPU或创建台账：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_live_ppo_supervisor.py prepare \
  --manifest /tmp/lite3_source_live_ppo_run1_20261102.proposal.json \
  --checkpoint /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt \
  --output-dir /tmp/lite3_source_live_ppo_run1_20261102 \
  --attempt-id lite3-source-live-ppo-run1-20261102-v1
```

`run_proposal.template.json`是可审阅的参数模板，revision为空，不是可执行批准文件。
只有用户单独同意这次启动后，才从冻结提案创建保留全部参数的批准副本，记录用户
同意，并将`user_approved/status`改为`true/approved`。原提案保留。随后唯一一次运行：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_live_ppo_supervisor.py run \
  --manifest /tmp/lite3_source_live_ppo_run1_20261102.approved.json \
  --checkpoint /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt \
  --output-dir /tmp/lite3_source_live_ppo_run1_20261102
```

当前阶段在冻结提案与汇报后停止，运行后也须归档、提交、核对并再次暂停。
