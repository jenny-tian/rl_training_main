# Lite3 在线训练失败审计与修复（2026-09-18）

已确认训练数据处理和候选模型选择存在缺陷，修复后通过单元回归和三个 Isaac Lab 实例验证。这里验证的是代码路径；尚未用修复版重新完成 100 回合训练，不能据此宣称成功率已经提高。

## 历史证据

审计目录为 `artifacts/semester_2026/09/lora_ppo_multiepisode/seed101_edge_slip_v2_100ep_corrected_v3b`。

- `learning_state.json`：100 回合，正式成功 12 次，跌倒 38 次；682 次更新中接受 679 次，冻结探针触发回退 13 次。最后 10 回合成功率为 0，尚未达到稳定成功标准。
- 最终选用第 80 回合的冻结探针验证模型。开发集复核通过 103、113，未通过 107、109，即 2/4。
- 13 份单回合 summary 报告 `completion_qualified=true`，其中第 70 回合存在安全违规，supervisor 正确排除，所以正式成功数为 12。13 回合全部在保存时丢弃了最后一段不足 256 步的数据。
- 38 条 `terminal_learning_update` 日志中，23 条前面没有真正执行 PPO 更新。日志事件数量不能当作有效学习次数。
- 第 100 回合：1776 步出现 29.16 N 严重非足接触，回退丢掉 240 步；1777 步再次回退丢掉 1 步；1778 步真正终止时只剩被安全投影屏蔽的末行。终止诊断错误地记录了重置后的 `x=-3.5, y=0`，掩盖了越界后的真实跌倒状态。

## 确认的原因与修改

1. **成功奖励没有进入更新。** 原 worker 到达终点后直接收尾，`finalize_for_checkpoint` 丢弃部分 rollout。现在合格完成会结束当前 rollout、禁止终点之后 bootstrap，并立即更新；接受后的候选保存下来等待冻结评估。
2. **失败经验先被回退删除。** 仿真 supervisor 使用 `--retain-terminal-learning` 时，现在在首次严重接触或真实终止处结束回合，以 -8 终止奖励完成当前 rollout 的更新，再由周期性冻结探针决定是否回退。默认保护模式仍支持即时回退。全块没有可用 PPO 行时明确记录未尝试更新，不能伪报学习。
3. **刚完成的失败学习在下一回合被覆盖。** supervisor 不再每遇到一次训练失败就立刻加载 recovery checkpoint；候选持续训练至下一次确定性探针。只有探针通过才提升 recovery，失败则退回已验证分支。训练成功本身不等于保存后的新候选已经通过验证。
4. **保存时错误地把候选标成 stable。** 部分/终止 rollout 不能提升 stable；保留候选时同时保存 `has_unverified_candidate` 和此前的 stable 状态，恢复训练后仍能回退。
5. **终止诊断和验收不同步。** LoRA 在 pre-reset 回调中复制位置、足端位置、奖励信号与接触诊断；速度误差改为 XY 范数。跌倒、重置和安全暂停的历史记录持续影响成功判定。固定台阶的边缘滑移窗口现在覆盖边缘两侧，修复了前沿前方与后沿后方被提前 return 排除的问题。
6. **SAC 奖励与验收存在差异。** 接入已定义但漏传的横向偏移惩罚；当前帧姿态先参与完成判定，再计算完成奖励。终止快照补全 Y 与足端 Y。奖励语义版本同步升级，拒绝旧实验状态直接混入新奖励训练。
7. **配对启动脚本覆盖正确参数。** 默认探索标准差从旧脚本的 0.05 对齐为 0.01，启用回合终止学习，动态计算协议 hash，并使用新的输出目录及结果兼容性检查。

## 验证

在线学习测试集：`1146 passed, 7 skipped`；`git diff --check` 通过。

另将待提交文件独立导出，核对不依赖未提交的其他修改：1144 项直接通过；两项 provenance 测试因导出目录缺少 Git 元数据失败，补建临时 Git 测试环境后两项均通过，无需修改产品代码。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:source/rl_training \
  /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests/online_adaptation
```

真实仿真输出位于 `artifacts/semester_2026/09/lora_ppo_multiepisode/audit_fix_20260918/`。三次运行均退出码 0，使用 RockyRoad、terrain seed 101、0.3 m/s、最多 2000 步、rollout 256、rank 2、std 0.01、Actor/Critic 学习率 5e-5。

| 样例 | 输入与触发 | 结果 | 最后部分 rollout |
| --- | --- | --- | --- |
| `failure` | 旧 episode-0089，仿真 seed 106，GPU 0；1776 步严重接触 | 到达 x=4.6407，但横向越界，`passed=false`；终止奖励 -8 | 240 步更新被接受，丢弃 0 步 |
| `completion` | 旧 episode-0002，仿真 seed 9，GPU 0；1765 步合格完成 | x=4.5005，未跌倒或越界，`passed=true`；最后奖励 5 | 229 步更新被接受，丢弃 0 步 |
| `terminal` | 旧 episode-0089，仿真 seed 106，GPU 1；1778 步真实终止并自动重置 | `fell=true, reset_count=1, passed=false`；保留重置前 x=4.6507；终止奖励 -8 | 242 步更新被接受，丢弃 0 步 |

`terminal` 是专门覆盖 pre-reset 回调的仿真回归：仅该运行将 worker 严重接触提前停止阈值调为 1e6 N，让环境自身的 5 N / 连续 3 帧终止先发生。正式默认阈值仍为 20 N。该运行的最终 PPO 行被安全投影屏蔽，但失败奖励仍经 GAE 影响同一 rollout 中更早的有效行。

三个 checkpoint 都满足：pending rollout 为空、候选仍标为未验证、stable 版本低于 active 版本，且没有在保存时撤销最后更新。前两个仿真启动后又补充了“安全暂停也持续标记失败”的判定；已检查它们的逐步记录，均没有安全暂停，不影响上述结果。

回归覆盖完成/失败末块更新、候选保存及恢复后的回退、终止信号不受原位 reset 改写、历史安全失败不能变成功、台阶两侧滑移窗口、SAC 当前帧完成资格及 supervisor 候选跨回合保留。

## 后续训练口径

- LoRA 学习版本：`boundary_samples_prereset_sticky_safety_v4`；多回合 manifest v4。完成资格版本：`in_lane_obstacle_interaction_edge_slip_v3`，当前协议 hash 为 `6243db772970452fd70811e7160739e277faa98b05f6c0398cad1e9eafd4f94d`。使用新输出目录，不能直接续写旧 v3b 目录。
- SAC 奖励版本：`contact_swing_lateral_current_pose_completion_v2`。旧实验 checkpoint 可以冻结评估，但不能作为兼容的新奖励训练状态直接恢复。
- 下一轮仍需开发集训练与冻结探针，之后在独立种子上复核。单次完成、损失下降、接受更新或到达终点都不等于稳定成功。
- 固定台阶边缘滑落、横向绕行、跌倒和安全违规均不能算通过。当前 `edge_slip_events` 针对固定台阶边缘；RockyRoad 尚无逐块岩石边缘识别，`edge_slip_events=0` 不能证明每个凸起都没有滑落。其当前通过证据是挑战区内行走、足端抬脚、稳定样本和无安全失败，仍需视频/局部接触证据才能确认更细的跨石动作质量。

本次提交包含修复依赖的既有严格验收、接触协议、LoRA 保护和 SAC 更新调度代码及测试。仿真 checkpoint、大体积训练产物和其他历史文档不纳入提交。
