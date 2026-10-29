# G3 前置条件：离线修复与单环境测试准备

日期：2026-10-29。基线为 `f313f77`，本目录对应其后的离线修复提交；实测前由监督器核对全部输入哈希、拒绝 dirty worktree，并记录最终 `git HEAD`。

用户在恢复会话后批准本离线阶段，并要求完整阶段结束后汇报、经同意再继续。**本阶段没有启动仿真或 GPU 测试，没有初始化 PPO；G3/G4 仍未取得资格。**

## 修复与审查结果

相对地形高度算法在 10 月 23 日已实现。本次发现其物理步刷新只在单环境扫描脚本中覆盖，Source 环境的默认配置仍按控制步刷新。已把 `height_scanner_base.update_period` 统一设为 `sim.dt=0.005 s`，覆盖单环境、reserved-row 和 shared-bank；前视高度扫描仍按原周期工作。

安全高度继续取根部世界 Z 减去基座九条射线中的最高地形命中点；有效条件仍为全量 XYZ 有限、每条射线到当前根部的水平距离不超过 0.08 m。新增拒绝非实数浮点扫描张量。扫描无效返回零高度，原 0.20 m 阈值触发停止，没有放宽阈值或退回世界 Z。八环境测试覆盖第 7 环境的 NaN、过期 XY、低净空和无效根位置，其他七个环境的高度及安全判定不受影响。

`FrozenPolicySafetyBridge` 现在先检查每个安全状态字段的形状、设备、数值有效性和动作映射，再更新过滤器记忆。NaN 与阈值比较可能返回 false，因此单独检查输出动作不足以阻止无效高度、姿态或速度通过。新增用例也拒绝跨环境广播的高度、错误的关节维度、非布尔掩码、零动作缩放及无可行余量的关节范围。

新增 `FrozenSourceActor` 仅接受已登记 SHA-256 的 `model_1998.pt`，加载其 `45→512→256→128→12` ELU actor。它不构造 PPO、critic 或 optimizer，不恢复训练状态，不更新权重；反序列化前先校验文件哈希，模型设为 eval 并关闭梯度。新入口复用环境包装和回合采集函数，绕开旧诊断脚本中 `OnPolicyRunner` 的初始化。

## 离线证据

- 新增故障用例在修复前得到 18 failed、6 passed，复现配置遗漏和桥接问题；修复后的针对性回归 115 passed。
- 扩展回归 **563 passed、0 failed、0 skipped**，含已有 shared-bank 接触/扫描/reset/边界/监督器测试和新的加载、验收及运行时配置故障测试。354 条 warning 来自已有 PhysX tensor API docstring 转义。准确命令见 [regression_command.json](regression_command.json)，结果见 [regression.log](regression.log) 与 [JUnit](regression.junit.xml)。
- CPU 对比 349 组输入，新 actor 与现有 TorchScript 的最大绝对输出误差为 **0**；检查了冻结参数和随机流不变。见 [actor_parity.json](actor_parity.json)。这只证明所测输入上的推理一致性。
- [preparation.json](preparation.json) 固定了 185 个输入哈希，包括源码、运行脚本、checkpoint、参考导出、模型资产和 34 项既有依赖；[post_review.json](post_review.json) 记录本阶段结论。最终目录文件见 [SHA256SUMS](SHA256SUMS)。

这些 CPU 测试不证明实际传感器刷新时序、完整策略安全性或动态物理隔离。

## 下一次拟执行的单环境测试

此测试用于取得独立于并行 G3 的单条件策略安全证据。**它本身不是 G3，也不能给其他地形或共享 bank 授予资格。**

| 项目 | 固定条件 |
| --- | --- |
| 环境/地形 | 1 环境，1×1 Flat L1，12×4 m，非 shared-bank |
| 策略 | `model_1998.pt`，SHA-256 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c` |
| 种子/回合 | geometry 10000，initial-state 20000；1 回合，不自动重试 |
| 命令 | `vx=0.3 m/s`，`vy=0`，`yaw_rate=0`，不启用 heading command |
| 时间/完成 | 0.005 s 物理步，0.02 s 控制步；最多 2000 步/40 s；前进 9 m 后连续稳定 50 步 |
| 设备/预算 | 物理 GPU0，无计算进程，可用显存至少 12,288 MiB；启动 120 s、总计 240 s、清理宽限 10 s |
| 正常终止 | `source_terrain_tile_bounds`、`illegal_contact`、`bad_orientation_2` 保持启用；不注入 timeout |
| 边界/阈值 | 全部 17 个刚体，半径 0.27 m、余量 0.001 m；非足端接触 5 N/连续 3 步；姿态 XY 范数 0.65 |
| 过滤器 | 最小高度 0.20 m、最大关节速度 30 rad/s；非足端接触判定 1 N/10 步宽限；任何 pause 立即停止 |

通过要求完整 completion 和 50 步 hold 证据、无正常 manager 终止、无安全拒绝/无效状态/非足端接触事件、动作 trace 连续完整，以及 actor/输入哈希未变化。达到步数上限按失败处理。过滤器保留原宽限，但本次安全资格的通过条件要求整回合非足端接触事件数为零。运行入口会检查实际刚体映射、扫描周期、控制周期、guard 参数、终止集合和运行命令；监督器继续使用既有失败标记识别、固定超时和进程组清理。

待用户另行同意后，仅运行一次：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_policy_prequalification.py \
  --run-dir /tmp/lite3_source_policy_prequalification_flat_20261029
```

必须在本工作树的干净提交运行；目标目录必须不存在。程序将逻辑 `cuda:0` 绑定到已检查的物理 GPU0 UUID。失败即保留 manifest、日志、逐步样本和 outcome，不换卡、不扩大规模、不延长时限或重试。异常发生在输出目录创建之前时，只产生终端错误，不应误称已有运行证据。

## G3 仍需满足的条件

既有正式 Source manifest 保持 `draft_not_for_run`、`execution_authorized=false`；原 SafetyFilter 记录仍为 `not_qualified`。本目录是新的前置准备记录，没有改写历史诊断的结果。

即使下一次 Flat 测试通过，也只覆盖这一个策略、版本、地形和命令条件。共享 bank 的初始分配和异步 reset 会涉及其他地形，因此仍需相应独立安全资格。改变了基座扫描周期后，历史八环境 G2 证据仍绑定原提交 `a34b1b7`，新版本须重新验证相关启动/reset 路径。后续还须准备并审查真正的 G3 运动采集：物理子步接触、全刚体位移/速度/加速度、边界、扫描归属、异步终止/reset 及恢复首步。原 G2 的 20 步零动作接触探针不能代替这些证据。

本阶段结束后停在确认点：建议下一阶段执行上述一次单环境安全检查并归档结果，再向用户汇报后续阶段。
