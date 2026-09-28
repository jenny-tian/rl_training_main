# G0.4：统一步数账本、回合边界恢复与冻结评估

日期：2026-09-28。依据 [v2.2](lite3_execution_plan_v2_2_2026-09-28.md)，接续 [共同任务奖励/终止](lite3-g0-task-objective-2026-09-28.md)。

## 本轮实现

新增 `interaction_budget.py`、`managed_trial.py` 和统一开发入口 `run_adaptation_session.py`，实际接入 LoRA/SAC 两个 worker。旧多回合脚本保留历史行为；新入口显式使用 finite-horizon-v1。

- 每次 `env.step()` 前写入并 fsync 一条 reservation，返回后写 commit。多环境调用按环境数量计费，不把一批 transition 当成一步。
- warmup、adaptation、selection、flat-selection 共享适应预算；selection/flat-selection 还有共同子上限。最终 evaluation、retention 分开计费。
- manifest 锁定 base SHA、任务协议、算法参数、地形/learner seed、源码哈希和预算上限。日志使用连续编号及哈希链，进程间使用独占文件锁。
- worker 的输出目录、输入 checkpoint、学习/评估模式和可消费步数必须与已登记 trial 匹配。SAC checkpoint 只能写到该 trial 的 `session.pt`。
- 进入冻结评估时锁定最后一个完整回合的候选 checkpoint，之后不能返回 adaptation/selection，也不能改参数或替换候选。
- 冻结 SAC 改用纯推理结束路径：推进历史状态，但不写 replay、不更新网络/优化器。LoRA 冻结路径保持原有零更新行为。

当前开发 supervisor 调度 adaptation/evaluation 两阶段并选择最后一个完整候选。账本已支持其他成本类别，但尚未自动调度4000步warmup、晋升探针、平地选择/保持或按成绩选checkpoint，不能把这些计划能力写成已实现。

## 恢复语义

恢复点是**完整物理回合边界**。下一个 worker 从已提交 checkpoint 恢复 learner，开启一个新的物理回合；不声称恢复被中断的 PhysX 中间状态。

- 已完成 trial ID 不可再次启动；重复确认 commit 不重复记账。
- checkpoint/summary/完整 trace 均已写完但监督记录中断时，可核对已确认步数后补记完成。
- 中途崩溃时，已确认成本保留。若最后一个 reservation 无法确认是否完成物理调用，该步保留为 `uncertain_steps` 并占用预算，不能静默退款。新 trial 使用新的编号和固定派生的 reset seed。
- journal 末尾发生不完整写入时保留原字节证据，回到最后一条完整事件；中间记录损坏或 manifest 被修改则拒绝恢复。
- actor/critic、Adam、replay、alpha、稳定候选等沿用已有 checkpoint 机制；新增 LoRA learner RNG 保存，SAC 恢复未使用的 critic 更新额度。
- 物理 reset 在隔离的随机数上下文中执行，避免重新采样执行器延迟时覆盖 learner 的随机流。reset seed 由 learner seed、phase、trial 编号派生，两算法同编号可配对。
- actor 更新额度属于上一物理适应 episode，新回合清零；critic 未消费额度继续保留。

源码或协议改变必须新建 session。旧 session 的已花成本和失败记录继续保留，不能通过编辑 manifest 扩大预算或换成有利结果。

## 封存边界

新 common-objective worker 不接受没有 managed session 的 sealed 运行。sealed manifest 必须声明锁定协议、算法配置哈希、候选选择规则哈希及地形成员关系；配置变更、种子不匹配、评估后返回训练均被拒绝。

这保留了“按已锁定算法在封存地形内适应”的接口语义，并禁止冻结评估反馈回调参。本轮只做封存拒绝/锁定单测，**没有生成新封存种子或运行真实封存实验**；当前统一 supervisor 仅开放 development。

## 验证结果

每个 session 分四次启动监督进程：先跑一个128步trial、重启完成余下128步、冻结评估16步、再次启动已耗尽的评估。实际 trial 数依次为1/2/3/3。

| session | 确认/计费步数 | learner/replay续接 | 冻结评估 |
|---|---:|---|---|
| LoRA | 272 / 272，uncertain=0 | Critic Adam step 4→8；评估后仍8 | updates=0，策略张量与锁定候选完全一致 |
| SAC，alpha=1 | 272 / 272，uncertain=0 | replay 128→256；全部Q更新被TD门槛拒绝 | updates=0，replay写入0 |
| SAC，alpha=0.02 | 272 / 272，uncertain=0 | replay 128→256；Critic Adam step 2→131 | updates=0，replay写入0，actor张量不变 |

低温度SAC的critic未使用额度从126续接，第二回合128条新增数据、129次Q更新后剩125，与账目一致。三个session均确认第二trial读取第一trial的checkpoint，评估读取第二trial的checkpoint；重复启动已完成评估新增0步。

原alpha=1案例没有被删除：其平均TD误差约11.9，超过既有10的门槛，所以没有Adam状态。另开alpha=0.02只用于产生非空优化器状态并验证恢复，未放宽TD保护门槛，不构成G1正式超参选择或任务收益证据。

本轮实际消费816个环境transition；与前两组G0诊断累计28,914步，原G0-base-pair按旧登记单列。GPU最多同时使用两张。

结果与完整命令保存在 `artifacts/g0_session_ledger_20260928/`：

- `validation.json`：成本、checkpoint链、Adam步数、replay及冻结张量校验。
- 各session的 `ledger/manifest.json`、`ledger/events.jsonl`、`session_summary.json`。
- 每trial的 `invocation.json`、runtime log、trajectory、summary和checkpoint。
- `*-resume_stages.json`：四次独立启动的累计状态。
- `validated_smoke_source.tar.gz`：低温度SAC复核时的完整源码快照，已核对其哈希等于该session绑定的源码哈希。随后补强的入口身份/输出目录校验另经单测验证。

最终测试：**1550 passed, 7 skipped**。覆盖真实子进程 `os._exit()` 后的锁释放与预算保留、torn journal、并发所有权、重复ID、预算上限、选优共享成本、冻结回流阻止、不可变checkpoint、reset/learner RNG分离及SAC推理不写replay。

## 下一步

G0仍需验证并锁定初态扰动范围，完成统一探针/平地保持调度和连续短跌倒停止规则，再形成无待定字段的最终 `protocol_v2_2.json`。当前入口属于工程诊断，不替代正式封存协议或G3实验设计。

这些门槛满足后才进入G1的目标域critic预热及LoRA/SAC小规模适应比较。本轮没有启动长训练，所有GPU smoke已结束。
