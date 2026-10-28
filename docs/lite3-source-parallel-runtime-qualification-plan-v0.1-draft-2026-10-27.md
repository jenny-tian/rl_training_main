# Lite3 Source 并行运行资格检查方案

版本：v0.1 draft，2026-10-27
状态：`g1_16env_startup_passed_32env_pending`
执行授权：2 环境 smoke、8/16 环境启动检查已获用户同意并完成；32 环境及后续运行、PPO 训练未授权。

## 1. 目的与当前结论

本方案承接[2 环境 importer/reset smoke](lite3-source-parallel-terrain-importer-smoke-plan-v0.1-draft-2026-10-27.md)，检查并行 Source 地图扩大到候选环境数时的初始化、异步 reset、物理隔离和资源占用。2026-10-28 的 G0 静态复核确认当前工作树中 12×4 m 几何与准备 manifest 一致，逐环境 tile-bound guard 已实现，并有离线测试；因此可进入 G1 的小规模启动 smoke。既有 2 环境 smoke 仍不覆盖新 guard，也不表示 120 环境可安全或高效运行。G1–G4 和训练授权仍未完成。

检查结果最多支持“指定代码、指定几何、指定设备和指定环境数通过并行运行资格检查”。它不等于 Source 地形/策略 qualification，不批准 PPO 更新，也不批准在线适应或机器人实机运行。

## 2. 必须先冻结的输入

资格运行不得从含有混合暂存和未暂存改动的开发工作树直接启动。运行前须建立独立 worktree，并记录不可变的代码 revision、依赖环境、运行脚本 SHA-256 和输出目录。以下输入也须写入本次 run manifest：

- 最终拟测的 `num_envs`、15 个子地形名称/顺序、等权列、地形生成 seed、reset sampler seed 和环境 seed。
- 当前 G0 已核对 Source 配置和 45-cell 准备 manifest 均使用 `size=(12.0, 4.0)`、`flat_length_m=3.0`、登记的 `start_x_m=1.5` 与 `completion_x_m=10.5`；坡/粗糙地形候选使用 2 m exit-flat。当前工作树仍有未提交改动，G1 启动前须记录完整代码快照/patch 哈希，并把该快照与模型子模块及运行脚本哈希写入 run manifest。
- 解析后的 terrain 配置：实际 `num_rows`、`num_cols`、tile 长宽、入口/出口、completion 点、各族几何参数、`terrain_origins` 范围，以及 `terrain_out_of_bounds` 是否启用。
- 若测运动中隔离，须锁定已通过单环境安全资格的冻结策略 checkpoint 与 SHA-256、命令范围、episode 长度和终止规则；不允许加载训练器或执行梯度更新。若没有可用且独立资格化的冻结策略，运动隔离门保持阻塞。
- 设备及启动前资源快照。运行前重新查询 `nvidia-smi`；从 GPU0 或 GPU1 中只选择一张确认空闲且可用的卡。若启动前可用显存低于 8 GiB、有冲突仿真进程或设备异常，则停止，不改用未检查的卡。

`nvidia-smi` 在本方案编写时可正常查询；资源状态会变化，不能把当前快照当作之后运行的资源承诺。

## 3. 分阶段检查

每一阶段单独记录配置、日志和结果；失败即停止，不自动重试，也不跳到更大规模。

### G0：静态配置与地图契约

不推进物理仿真。按拟测环境数构建地形配置，断言实际 origin grid 为 `(N, 15, 3)`、所有坐标有限、15 个列名与登记顺序一致、列比例均为 `1/15`，且 `terrain_levels[i] == i` 表示环境所有权而非难度等级。对每个环境记录初始化 tile 和相邻 tile 的实际世界坐标；计算 footprint、tile 边界、completion/最大允许运动范围及相邻机器人最近可能距离。

通过条件：所有检查与解析后的 manifest 一致；不同环境没有重复的 ownership row；根据真实 tile 几何和机器人占用范围，能证明静态 spawn 不重叠。任何配置值或几何边界不明时，不进入物理阶段。

**2026-10-27 Lite3 USD 静态碰撞审计：** 已固定 `deep_robotics_model` 提交和入口 USD/PhysX payload 哈希。默认 `Physics=PhysX` payload 含 22 个碰撞 primitive（6 个 Cube、12 个 Capsule、4 个 Sphere）。按拥有刚体的局部坐标计算，最大保守包络为躯干盒体的 `0.2643860811 m`；当前候选 `collision_radius_m=0.27 m`，另加 `0.001 m` 数值余量。球形逐刚体包络对刚体朝向不敏感，也不需要再对单个刚体的关节姿态重复膨胀；但此静态审计不证明运行时资产/刚体映射一致，也未测量控制周期内的最大位移或加速度。因此它只支持候选值，不能单独解除 G3 资格门。审计哈希和逐刚体数值见 [碰撞几何审计记录](lite3-usd-collision-geometry-audit-20261027.json)。

**2026-10-27 历史发现：** 当时的准备索引使用 10 m tile，而工作树及 manifest 使用 12 m；旧 2 环境 smoke 不能单独证明后续代码版本。

12×4 m 轨道容纳从 `x=1.5 m` 起步、在 tile-local `x=10.5 m` 完成并继续 hold；坡面退出过渡后平段从 `x=10.0 m` 开始。当前工作树与 manifest 已对齐这些值，静态记录见 [G0 静态审计](lite3-source-parallel-g0-static-audit-20261028.json)。

Source 地图的 `border_width=0.0`，并使用 terrain-origin 网格；terrain-generator row 沿 x 前进方向相邻，column 沿 y 横向相邻。并行配置为每个环境占有唯一 row、15 个横向 tile column。默认整图 `terrain_out_of_bounds` 仍关闭，因为它不能保护内部 tile 边界；Source 环境现在另有逐环境 guard，对 `terrain_levels`/`terrain_types` 选定的 tile 和所有刚体位置逐一检查。静态 USD 审计得到逐刚体最大包络 `0.2643860811 m`，候选半径为 0.27 m 加 0.001 m 数值余量。按 USD 关节 anchor 位移长度和三角不等式，任一刚体坐标系到躯干根坐标系的保守距离小于 `0.69266 m`；登记的 50-step hold 结束时，考虑该关节工作空间后，到 guard 前向刚体中心界限还有约 `0.236 m`。四边界、row/column 选择、异步环境隔离、completion/hold 和失败关闭均有离线测试。G0 静态契约通过；此结果不证明动态加速度、控制周期内越界或运行时物理隔离，需在 G3 验证。

**当前逐环境 guard 的动态边界：** 代码在每个控制步检查所有刚体中心的当前位置，并用当前线速度外推一个 `env.step_dt`。这是提前终止启发式，不是连续时间证明；目前没有独立测得或限制最大刚体加速度/控制步位移。G3 必须记录冻结策略下的极端刚体速度、位移、姿态和 guard 触发前后状态；若数据超出 margin/lookahead 假设，须先调整机制并重跑离线与小规模验证。

登记的 completion 是 root 相对 reset origin 前进 9.0 m；50-step hold 的名义前进距离为 `50 × 0.02 × 0.3 = 0.3 m`。静态 G0 依据候选半径和 USD 关节 anchor 工作空间核对了该 completion/hold，运行时最大位移仍由 G3 实测。不能把 tile-local `x=10.5 m` completion 坐标直接作为 guard threshold。

独立离线测试覆盖四边界、row/column 选择、非 reset 环境不受影响、completion+hold、线速度 lookahead 触发和失败关闭。当前 manager termination 已接入该 guard；G1 的 2 环境复测需确认运行时 `body_ids` 完整解析并记录环境内终止原因，G3 再测动态越界行为。

### G1：逐级地图构建与启动检查

先针对当前代码快照重跑 2 环境 smoke（旧归档不覆盖新增 guard），再依次测试 8、16、32、60，最后才测试 120 个环境。每一级均使用独立的结果路径，只生成地图并检查启动，不加载策略、不训练。记录地图构建时间、CPU 峰值 RSS、GPU 峰值显存、mesh/triangle 数、环境数和 origin grid；断言 root state 与对应 `terrain_origin` 对齐、origin/观测/两个 height scanner 输出均有限。

通过条件：当前级别无重复 spawn、非有限值、意外终止、地图缺失或资源错误；实测资源有可解释余量。任一级失败则保留证据并停止，不继续放大。

### G2：异步 reset 与跨环境不变性

在当前通过的规模下，对单环境、多个不连续环境及较大 reset 批次执行异步 reset。每次记录 reset IDs、分配编号、row/column、origin、root state、命令和传感器缓存。断言只更新被 reset 环境；未 reset 环境的 allocator ticket、terrain row/type、origin、root state 和观测历史均保持不变。连续执行足够批次以覆盖至少一个完整 120-ticket block，并核对家族/方向计数和尾块报告。

通过条件：reset 次数、ticket 顺序和比例与分配器记录完全一致；失败写入不消耗票据；未 reset 环境没有位置或状态跳变。当前适配尚未实现 checkpoint 调度状态恢复；本检查不能据此宣称可从训练 checkpoint 无缝续跑。

### G3：运动中物理隔离与边界行为

只有在 G0–G2 通过、并且冻结策略与命令范围已有独立资格后才执行。以冻结策略在候选环境数中运行限定时长的只读 episode，不进行 PPO 初始化、优化或权重保存。覆盖相邻 rows、不同 terrain columns、异步结束/reset 和最大允许行走距离；记录跨环境接触对、环境边界穿越、terrain collision、ray-caster 命中归属、reset 后首次 physics step 和所有 termination 原因。

G3 开始前，G1/G2 必须通过，且冻结策略、命令范围和运行时刚体映射已锁定。G3 使用只读策略、不进行 PPO 初始化或梯度更新，检查 guard 的动态边界保护。通过条件为无跨机器人接触、无越界进入邻接环境、无未预期终止，且传感器始终读取预期地形；任何违反立即停止。

### G4：GPU、地图和吞吐预算

在每个已通过的环境规模下记录建图时长、峰值 CPU RSS、峰值 GPU 显存、首步延迟、稳定期 control steps/s、reset p50/p95 延迟、有效 transition/s，以及 GPU 温度/利用率。运行前/后保存 `nvidia-smi` 输出；不与其他仿真任务并行争用所选设备。

运行前须先依据候选 `num_envs × rollout_length` 和期望完成时间写明吞吐门槛。若没有预先登记的目标吞吐，只能报告测量值，不能判定预算通过。出现 OOM、数值异常、资源无法稳定、reset 延迟失控或吞吐不达预设门槛则停止并重新审查环境数/地图布局；不自动改地图配比或 sampler。

## 4. 总体通过与停止规则

并行运行资格只有在 G0–G4 全部通过、输入及代码哈希可复现、reset 比例与 transition 比例分开报告后才可标为 `parallel_runtime_qualified`，并且只对该代码、几何、环境数和设备组合有效。任何一个门未通过都保留为 `blocked` 或 `not_run`，不把局部 smoke 改写成整体通过。

资格检查过程中任何非有限状态、重复/错误 spawn、跨环境接触、未预期边界穿越、异步 reset 改动未结束环境、CUDA/PhysX 异常、资源越线或审计数据缺失，都立即停止并保留 trace。禁止自动重试、增加预算、启动 PPO、更新策略权重或转入实机。

## 5. 当前未解决项

- 120×15 地图的实际建图时间、mesh 数量、内存和吞吐没有测量。
- 唯一 row/origin 只证明索引分配；运动过程中是否越界、碰撞或读取邻接地形仍未验证。
- G0 的静态 map/guard 契约已通过；G3 仍须测量动态刚体位移与加速度，验证线速度 lookahead 足够保守。
- 2 环境运行已固定到干净提交 `12ab9e0`，运行清单记录代码与模型哈希；后续运行仍需重新冻结输入。
- 新 guard 的 2 环境复测已通过；8/16 环境启动检查也已通过；32/60/120 环境尚未运行。
- reset 调度状态没有 checkpoint 保存/恢复协议；训练恢复语义尚未资格化。
- 尚未锁定用于运动中隔离测试的冻结策略、命令范围和预设吞吐目标。
- 已有 2 环境运行出现 Lite3 足部视觉 USD 引用缺失警告；本次 smoke 未据此确认视觉资产完整性。

**2026-10-28 G1 进展：** 当前冻结代码的 2 环境 smoke 通过，全部 17 个刚体被 guard 覆盖，首步无终止，传感器有限，单环境 reset 不影响另一环境。完整证据见 [G1 两环境记录](lite3-source-g1-2env-20261028/README.md)。此前关于未提交工作树和待复测的描述为运行前历史；本次使用干净提交 `12ab9e0`。G1 更大规模以及 G2–G4 仍未通过。

**2026-10-28 G1 八环境进展：** 干净提交 `1b3244d` 上的 8 环境启动检查通过，环境源码与两环境运行一致；8×15 origin grid、全刚体 guard、首步观测和扫描均通过，退出码 0。环境构建约 27.02 秒，采样显存峰值 3444 MiB。证据见 [八环境记录](lite3-source-g1-8env-20261028/README.md)。这不代表完整 G1 或 G2–G4 通过。

**2026-10-28 G1 十六环境进展：** 干净提交 `bdbf0c7` 上的 16 环境启动检查通过，环境源码与八环境一致；16×15 origin grid、全刚体 guard、首步状态/观测和高度扫描通过。环境构建约 54.57 秒，采样显存峰值 4102 MiB，退出码 0。证据见 [十六环境记录](lite3-source-g1-16env-20261028/README.md)。G1 后续规模和 G2–G4 仍未通过。

**下一步：** 用户同意后进入 G1 的 32 环境启动检查；保持当前几何、guard、seed、超时和失败停止规则。本轮停在 16 环境，不自动放大，不启动 PPO。
