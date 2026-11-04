# 120 环境 G2：第一批 reset 缓存边界检查未通过

状态：`rejected_at_first_reset_contact_cache_boundary`，实际运行日期 2026-11-04。

用户单独同意后，从干净且实时远端一致的
`48ed6680d16c8fe090ede7db73407db9a135116a` 执行唯一一次 GPU0 G2。第一批 `[0]`
的原生 reset 返回后，观察器在 before/after 比较中发现非选中接触缓存变化，立即
停止。监督器和独立验收均拒绝，未重试、续跑、修改验收条件或进入训练。

冻结提案为 11 月 3 日创建的
`/tmp/lite3_source_g2_120env_20261103.proposal.json`，SHA 为
`3bc4c8d71beeb78ff7dda3f09b710b978b7aa1a7a47a680ed53cce7c6a1bc9dc`。
批准副本只修改 status/user_approved，原输出路径中的 20261103 也按冻结范围保留。
本次调用实际 UTC 时间为 2026-11-04 01:04:55.339310 至 01:06:23.517561。
Git 远端读取沿用进程环境内的同仓库 SSH 映射，没有改变仓库配置。

## 实际执行到的位置

| 项目 | 实际结果 |
|---|---|
| 设备/场景 | GPU0 RTX 3090，120 环境、1 行 shared bank，六家族/15 条件 |
| 初始化 | 场景与观测构建完成，初始 120 张票和 sampler 一致 |
| 接触 view | 已建立 2040 刚体×2064 目标的拓扑，capacity 1048576 |
| 运行时证明 | 1 次原生 cuda:0 stage/backend 检查，尚无逐物理子步采样 |
| reset 尝试 | 仅第一批 `[0]`，有 attempt 和 after 快照，缺少 checked 通过记录 |
| 完成的合格 reset 批次 | 0；原生 reset 返回不代表 G2 边界验收通过 |
| 恢复控制步/物理子步 | 0 / 0；steps 文件为空，没有接触或运动子步帧 |
| 诊断/训练/PPO | 实际完成 0 / 0 / 0；计划的 480 条诊断没有执行 |
| 其余三批 | 未执行；304 张票、两完整块和 64 张尾块未完成验收 |
| 退出 | 监督器返回 1，子进程关闭退出码 0，无超时，进程组清空 |

G2 未取得通过资格。没有恢复首步数据，不能据此判断 G2 运动隔离、承重、接触
正控制或吞吐。此前 G1 的限定资格不变，G3/G4、技能及正式训练仍未取得资格。

## 原始失败和独立核对

实际断言为：

`reset changed untouched buffer: sensor/contact_forces/_is_outdated`

日志包含一次 Python traceback。before/after 都有 42 个字段，第一批 reset IDs
为 `[0]`，两份快照的 sim_step 均为 0。CPU 重新执行原边界检查，得到相同断言。
对非选中环境逐项比较得到：

| 字段 | 变化 |
|---|---|
| contact_forces._is_outdated | env1–119，共 119 行，由 True 变 False |
| contact_forces.net_forces_w | 26 个非选中环境的缓存由零填入数据 |
| contact_forces.net_forces_w_history | 同样的 26 个环境出现缓存更新 |
| 其余 39 个已记录字段 | 非选中行全部保持相等 |

26 个环境为 36、40、41、42、44、45、49、50、51、54、55、56、59、60、61、64、
65、66、70、71、72、73、76、77、78、79。非选中 root、关节、origin、动作、命令
及记录到的运行时状态没有变化；这不构成对未采集物理量或连续运动的保证。
选中 env0 的 column/origin/root 按 reset 改变，两路扫描过期标志被置为 True。
实际 observation_history_terms 为空；本轮没有真实观测历史项可检查。

parent acceptance 中首条理由是 `G2 report mismatch: reset_batches`。失败发生在
采集完成之前，report 中该字段还保留计划列表，而完整验收要求完成数量 4，因此
首先触发了报告一致性拒绝。原始错误保存在 report.error、runtime.log 和
contact_error 事件中。该通用拒绝理由不是导致运行停止的原始断言，不能只看子进程
退出码或该理由就判定成功/根因。

## 直接机制及 CPU 重放的范围

源码检查确认实际调用链：PreResetCaptureManagerBasedRLEnv 先执行预 reset 回调；
真实 Lite3IsaacLabAdapter 注册的 capture_terminal_contact 会调用
_nonfoot_contact_peak，读取 contact_sensor.data.net_forces_w_history。这个 data
getter 调用 SensorBase._update_outdated_buffers，更新全部过期行并将对应标志置 False。
之后才进入原生 reset 清理选中环境。观察器的 before 快照刷新了扫描，但未先刷新
接触缓存，因此比较了旧缓存与回调刷新后的缓存。

相关原生文件已在 51 项依赖指纹中锁定：ContactSensor.data 位于已安装
contact_sensor.py 的 105 行，SensorBase._update_outdated_buffers 位于 sensor_base.py
的 354 行。项目回调和观察器源码指纹也一并保存。本轮没有修改它们。

CPU 复核执行了这些原生传感器的 getter、reset 和更新簿记方法，使用原始 after
缓存作为假后端输入，复现了记录中的接触标志、净力缓存和历史缓存变化。在同一
重放中先刷新再拍基准快照，随后读取与单环境 reset 保持非选中缓存相等。
这只核对延迟刷新/清理机制，不重新计算物理力，也不能证明真实 GPU 修复后必然
通过。缓存中的非零力没有对应本次稀疏物理帧，不能授予支持力阳性资格。

上一阶段 CPU 合成环境没有包含真实适配器的这条预 reset 接触读取回调，导致覆盖
缺口。929 项 CPU 测试通过不能替代这次实际 G2。本阶段只做失败分析和归档，没有
忽略变化字段、修改容差或调整采集源码；真实 G2 仍按原冻结规则判为失败。

## 资源、关闭与预算

| 口径 | 结果 |
|---|---:|
| 外层调用，含父进程预检 | 88.1779889 秒 |
| 监督器计时 | 75.8604219 秒 |
| 子程序关闭前 | 64.5314932 秒 |
| AppLauncher 启动 | 30.4599419 秒 |
| 场景构建 | 7.5115515 秒 |
| GPU0 前/后整卡占用 | 6077 / 6077MiB |
| GPU0 整卡采样峰值/最小可用 | 8948 / 15177MiB |
| 本次进程采样峰值 | 2834MiB，PID 1236164 |
| GPU0 温度采样范围 | 47–55 摄氏度 |
| events 大小 | 8166699bytes |

共 15 次资源采样，本次进程出现 9 次，不能当作精确峰值；卡上已有任务，整卡占用
不能全部归于本次。RSS 峰值和恢复步计时只在正常采集完成路径写入，本轮未记录，
不能补成 0 或据显存估算。未触发 120/480/15 秒限制；没有稳定期吞吐或 reset 延迟
资格数据。进程组 1236163 独立检查为空，本次 worker 已从 GPU 表和 /proc 消失。
环境 close 返回，application 记录 closing 后正常退出，不声称 Python 层 app.close
已返回；persistence_errors 为空，未清理其他任务。

控制台 500 条 Warning，其中 480 条 visual 引用、6 条 GPU 枚举跳过；runtime.log
有 1 次断言回溯。Kit/Isaac Lab 日志未检出 Python traceback，三份日志均未检出
[Error] 标记。这些计数不抹去业务断言失败，也不确认其他 GPU 健康或视觉资源完整性。

243 项生产输入、51 项依赖、G1 120→60→32→16 前置归档、原模型均保持一致。
checkpoint SHA 仍为 `87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`，
模型子模块仍为 b452a1f。生产台账逐字节不变，SHA 仍为
`b1d224ba20a93c93c99189d700aeb145d506259f56462eb506def64f8f888189`，
仅保留 run1 accepted 384 条/2 次真实 PPO 更新，run2/run3 未消费，无退款操作。

## 归档和下一阶段提案

provenance.json 保留原件路径、字节数和 SHA，原始数据和日志不改写。
SHA256SUMS 绑定原始文件、文档、CPU 复核器和 provenance，review.json 单独保存。
review.py 在 CPU 上核对原冻结拒绝、失败边界、缓存变化、机制重放、预算及清理，
不启动 GPU。archive_review_passed 只表示失败归档通过复核，G2_accepted 始终为 false。
Git 空白检查仅排除保留原始尾空格的 runtime.log/kit.log，其余文件正常检查。

下一阶段只建议 CPU 修复与回归：将真实适配器的预 reset 接触读取链纳入样例，
在可比较的新鲜缓存基线上拍快照，保留非选中状态逐项不变性检查；增加本次真实
快照回放和污染故障用例，改进失败报告中计划批次与完成计数的区分、原始原因的
呈现。完成测试、独立复核和新的未批准提案后汇报。旧失败原件和批准清单保留。
本阶段没有进行该源码修复，也没有创建或启动新的 GPU 尝试；重测仍须另行同意。
