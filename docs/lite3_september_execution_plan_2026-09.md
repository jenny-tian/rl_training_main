# Lite3 2026 年 9 月执行方案

版本：2026-09-06。对应 [学期计划](lite3_online_adaptation_semester_plan_2026-09_to_2027-02.md) 中 T01–T07。

本次交付是计划更新，不代表已经运行下面的物理实验。复用现有 model_1998；仅在实际复测发现策略问题后才讨论平地微调。四类复杂地形是后续 residual 在线适应对象，不进入基础 PPO 训练。

## 1. 本月底交付与当前状态

| 工作 | 当前状态 | 9 月交付 |
|---|---|---|
| PPO 三轮训练链 | 已核对，均为平地；最终模型/配置哈希一致 | 引用现有审计，登记复测使用版本 |
| 已有 100000 条 replay | manifest 已读取，完整哈希和语义待查 | `asset_validation.json`：reuse/convert/recollect 及理由 |
| 多速度平地录像 | 入口已存在，本轮未执行 | 0.1/0.3/0.5 m/s × 5 种子，原始轨迹及汇总 |
| 纯本体感知 residual | 49 维组件已有，非平地 worker 全链路待适配 | T03 schema 与回归报告 |
| 干净 residual 初始化 | 待构建 | 初始化 checkpoint、来源清单、等价性和恢复报告 |
| 低台阶短程学习 probe | 依赖 T03/T04/T05 | 最多 20 回合的更新/恢复证据和预算测量 |

以上新产物写到代码 worktree 下 `artifacts/semester_2026/09/` 的独立版本目录。零 residual 初始动作、网络未训练和不发生探索是不同概念：高斯策略零均值仍能采样非零动作；严格 base 等价检查要显式 gain=0。

## 2. 9/6–9/9：T01 资产核验

输入为学期计划第 2 节登记的模型、manifest 和 replay。

操作顺序：

1. 确认代码在 worktree 内执行；保存当前差异和环境版本快照，避免只记 Git 提交却遗漏未提交代码。
2. 对模型、agent.yaml、env.yaml 重算 SHA256；比较 base manifest。
3. 对 replay 重算 SHA256，核对 replay manifest 中 baseline_manifest_sha256 是否绑定当前 base manifest。
4. 以可信本地 checkpoint 的加载方式检查字段与尺寸、有限值、frozen 状态、样本数、终止/截断/discount 和 observation 布局；不为了新协议直接改写旧哈希。
5. 核对旧 replay 奖励与拟使用的新在线奖励是否一致。若不一致，判断字段是否足够重算；足够则生成新版本与转换清单，不足才补采。
6. 将结论写入 asset_validation.json。模型能在平地成功不等于 replay 自动满足新实验语义。

完成条件：每个文件都有已验证或未验证状态；不得把读取 manifest 当作验证整个数据。已有训练链审计不需要重复从头研究。

## 3. 当前可执行的诊断命令

以下命令基于当前源码参数核对，尚未在本次计划更新中启动物理 rollout。使用 Bash，在同一终端依次执行。GPU 必须先确认可用；示例 `cuda:0` 只表示待选设备，不承诺当前空闲。

```bash
cd /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation
source /home/tjx/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD:$PWD/source/rl_training:/home/tjx/IsaacLab/source"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

登记输入和创建唯一输出目录：

```bash
LITE3_BASE="$PWD/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt"
LITE3_DEVICE=cuda:0
test -f "$LITE3_BASE"
sha256sum "$LITE3_BASE"
mkdir -p "$PWD/artifacts/semester_2026/09"
LITE3_RUN=$(mktemp -d "$PWD/artifacts/semester_2026/09/base03_seed7.XXXXXX")
```

模型预期哈希：`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。不匹配先停止，不能继续沿用旧模型身份。

第一条部署路径平地录像，0.3 m/s、seed 7、最多 2000 控制步：

```bash
python scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py \
  --base-checkpoint "$LITE3_BASE" \
  --mode base \
  --task FlatOnline-Deeprobotics-Lite3-v0 \
  --flat-retention-probe \
  --seed 7 --terrain-seed 7 \
  --forward-speed 0.3 \
  --max-steps 2000 \
  --residual-gain 0 \
  --control-timeout-ms 50 \
  --run-dir "$LITE3_RUN" \
  --device "$LITE3_DEVICE" \
  --headless --video --video-length 2000
```

这个 flat-retention probe 会把生成道路几何设为零台阶、零粗糙，并令 adapter 不追加台阶特征；它用于检查现有部署路径。不是多地形正式评测命令。50 ms 沿用当前仿真探针截止值，不能据此声称满足 20 ms 真机实时性。

查看结果和视频：

```bash
sed -n '1,180p' "$LITE3_RUN/episode_summary.json"
find "$LITE3_RUN/videos" -maxdepth 1 -type f -name '*.mp4' -print
```

若需四足轨迹进行拖腿分析，另用现有普通 PPO play 输出 NPZ；该路径与 online worker 的重置/安全包装不同，不能混成一条试验：

```bash
LITE3_GAIT_RUN=$(mktemp -d "$PWD/artifacts/semester_2026/09/gait03_seed7.XXXXXX")
cp "$LITE3_BASE" "$LITE3_GAIT_RUN/model_1998.pt"
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Flat-Deeprobotics-Lite3-v0 \
  --checkpoint "$LITE3_GAIT_RUN/model_1998.pt" \
  --num_envs 1 --seed 7 \
  --forward_speed 0.3 \
  --rollout_steps 2000 \
  --telemetry_output "$LITE3_GAIT_RUN/telemetry.npz" \
  --video --video_length 2000 \
  --video_output_dir "$LITE3_GAIT_RUN/videos" \
  --tracking_camera --headless --device "$LITE3_DEVICE"
```

普通 play 会在 checkpoint 旁导出策略，因此使用新目录中的模型副本，避免覆盖历史导出文件。它可能继续跨环境 reset，分析 NPZ 时必须分段，不能把 reset 后位置跳变累计为前进距离。

`qualify_base_policy.py` 当前不提供本方案所需完整固定速度+录像参数，因此本月不把它当作上述测试的替代入口。`train_residual_sac.py` 的默认源训练任务也不作为平地初始化命令直接启动。

## 4. 9/10–9/16：T02 多速度与平地保持基线

先检查第一条录像和日志，再用同样模式扩展到速度 0.1/0.3/0.5 m/s，种子 7/11/19/23/31，共 15 条轨迹。每条最多 2000 步；速度和 seed 变化时同时更新输出目录，seed 和 terrain-seed 保持匹配。

T02 需为 online worker 补齐四足高度/接触/速度等原始字段和稳定的汇总程序，优先复用 `foot_metrics.py` 与 adapter；不能只依赖 `heldout_*` 指标统计纯平地。封装批量命令前先通过一个完整案例。

候选资格标准（在本批运行前固定，非已测结果）：

- 主速度 0.3 m/s：去掉预先指定的前 5 秒启动段后，平均绝对前向速度误差不超过 0.08 m/s；有效段净前进不低于指令积分的 70%。
- 低/高速分别报告，初始诊断目标误差不超过 0.05/0.12 m/s；若 0.5 m/s 不通过，优先将论文主速度限定为 0.3 m/s，不为扩展速度立即重训。
- 轨迹无物理跌倒；单独记录姿态保护、持续身体接触、超时和环境终止。零跌倒不自动等于无安全事件。
- 前后腿抬脚高度、支撑时间和拖地要有统计与录像，不强制四腿瞬时相同。
- 计算完整循环的 p50/p95/p99 与 deadline miss；模拟推进和视频开销单列。

失败处理：先用相同配置重现，比较 PPO 和部署路径的观测、动作缩放、PD、初始化及命令，再判断是否模型缺陷。需要微调时只改平地训练，保留现有模型为对照，并重新绑定数据来源。

## 5. 9/17–9/23：T03/T04 纯本体感知与初始化

### 需要修改的模块

| 模块 | 要补齐的内容 |
|---|---|
| `online_adaptation/isaaclab_adapter.py` | 明确 proprio 输入模式，49 维布局、单位、缩放及 schema 标识 |
| `online_adaptation/cli.py` 与 `play_online_adaptation.py` | proprio 选项贯穿建网、恢复和记录；不自动追加 geometry 特征 |
| `online_adaptation/reward.py` 与 worker | 去除主模式精确边缘位置/台阶高度奖励和专用 gate；保留可解释的状态估计信号 |
| `online_adaptation/agent.py`、`sac.py` | 核对 gain、动作投影、replay 与 Q 查询的一致动作定义 |
| 初始化工具（待新增） | 使用已有平地数据预热 Critic，创建全新 Actor/optimizer、空 online replay，并封存完整初态 |

T03 与 T04 按依赖顺序实施。49/59 维检查点不能只通过裁切权重或补零宣称等价；新的本体感知初始化独立生成。平地数据只有真实字段布局一致时才可直接复用。

初始化方案：Actor 均值输出零初始化，Critic 从已核验 flat replay 做预热；先沿用现有 1000 次预热作为开发起点。Actor 不导入 v10/v24 权重；全局在线 replay、episode-local replay 和 n-step 队列从空状态开始。平地 baseline 单独加载。

验收：gain=0、相同安全投影下最终动作差不超过 1e-6；确定性均值为零；PPO 参数和观测接口不变；保存/恢复后动作与状态一致；数值检查和数据来源完整。平地仅零动作 replay 不能覆盖所有非零残差 Q 值，若需小幅探索数据，应另设有录像的平地采集协议，并算入初始化成本。

## 6. 9/24–9/30：T05/T07 与小规模闭环

先开发 supervisor 的清单和 session 恢复：区分场景分区与是否学习，不继续用 `--evaluation-only` 同时承担两件事。月末先做一个正常平地案例和一个开发低台阶案例，后者最多 20 回合、每回合最多 2000 步；不接续 v24。

记录：环境/控制/更新/锁等待耗时、每个 Actor 版本实际使用步数、有效 replay 数、拒绝更新原因、终止原因、回滚/soft reset、每回合视频和 checkpoint 大小。台阶样本可能失败，月末验收看闭环正确，不以学会为必须条件。

停止条件：非有限状态或参数、checkpoint 恢复不一致、观测 schema 错配、未能保存正式录像或磁盘预算不足时停止批量；普通未完成/物理失败则按预先规则记录并继续到预算上限。

## 7. 代码验证与交付清单

有代码改动时先跑相关测试；涉及观测、回放、target 或恢复时再跑完整 online_adaptation 回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/online_adaptation -q
```

这条命令依赖第 3 节的环境和 PYTHONPATH。CPU 测试通过不代表物理行为通过；每次正式物理验证都保存视频。纯文件/哈希检查不产生 rollout。

月底应有：

- asset_validation.json 和引用现有训练链的模型登记；
- 五种子多速度原始日志、录像、flat_baseline_report.json；
- 49 维 residual 的 schema、干净初始化及来源清单；
- PPO 动作等价、初始化恢复及相关回归结果；
- 最多 20 回合 probe 的更新/执行证据、失败清单和录像；
- 每回合墙钟/视频/checkpoint 大小及 100/500 回合预算估计；
- 10 月地形生成与通用进度的待办及进入条件。

## 8. 滑期处理

9/23 若 T03 尚未通过，9/24 后优先完成接口与恢复，不跳过验收启动长程学习。现有平地 PPO 仍可用于准备地形与日志。如果月底只完成 T01–T04，应明确交付这些成果并顺延低台阶 probe；不为赶日期修改安全阈值或把旧残差结果当作新实验。
