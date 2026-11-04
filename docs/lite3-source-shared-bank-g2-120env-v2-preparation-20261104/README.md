# 120 环境 G2 v2：接触缓存基准修复与 CPU 准备

用户同意的范围为 CPU 修复、回归、归档和新的未批准 GPU 提案。本阶段未查询 GPU，
未启动 GPU 任务或 Isaac Sim，未新增真实训练交互或 PPO 更新。真实 G2 仍未通过，
G3/G4、承重正控制、技能和正式训练资格保持未取得。

对应的原始失败见 [11 月 4 日 GPU G2 归档](../lite3-source-shared-bank-g2-120env-runtime-20261104/README.md)。
该次运行在第一批 [0] reset 后发现非选中接触缓存变化。旧批准清单、输出和归档均
保留，未重试或改写。旧 CPU 准备和 GPU 失败复核器使用修改前的原始源码快照重新
执行，结果与各自 review.json 一致；这些旧复核器继续对应原来的源码版本。

本次修复修改 243 项生产输入中的 5 项，全部 51 项已安装依赖保持原样：

| 文件 | 改动 |
|---|---|
| scale_diagnostic.py | 在 reset 前快照之前读取 contact_forces.data，刷新过期缓存；单独记录 reset 尝试次数 |
| source_scale_reset_collection.py | 在 finally 中保留已通过 reset、运动帧、票和延迟；控制步返回时累计实际完成交互 |
| source_scale_reset_probe.py | 从零初始化实际计数；失败路径也记录事件文件哈希和 RSS 峰值 |
| source_scale_reset_validation.py | 优先呈现原始失败；检查新鲜接触缓存基准、接触历史和实际计数 |
| scale_reset.py | 注册独立的 v2 schema/attempt_id，将计划批次命名为 planned_reset_batches |

边界比较仍使用原有逐项相等检查。没有删除接触缓存或过期标志，没有放宽容差。
被选中环境仍须清空动作、历史、接触缓存及运行状态；未选中环境的已记录状态仍须
逐项相等。新验收额外要求每批 before 的接触过期标志全部为 False，传感器时间与
上次更新时间相等，并保留三帧净力历史的完整字段。

CPU 样例现在实例化真实 Lite3IsaacLabAdapter，让它注册 capture_terminal_contact，
执行真实预 reset 环境方法、适配器接触读取，以及由原生文件提取的 SensorBase
update/reset 和 ContactSensor data/reset 方法。方法源码由旧运行清单的依赖指纹
核对，未导入 Isaac/Omni 模块。底层后端只供应 CPU 张量，安全输入沿用合成样例；
因此该测试核对回调、缓存与采集逻辑，不重新验证 GPU 物理力。

修复前先运行新增回归，得到 1 passed / 1 failed，失败断言与真实运行一致：
`reset changed untouched buffer: sensor/contact_forces/_is_outdated`。修复后首轮
120 项定向测试全部通过，pytest 492.90 秒，外层 495.50 秒，68 条依赖 warning。
随后补充的验收故障用例一并纳入最终登记回归：**951 passed，0 failed，0 skipped**，
覆盖 27 个登记测试文件，比此前新增 22 项，G2 相关检查共 127 项。pytest 耗时
902.87 秒，外层 905.98 秒。568 条 warning 均来自提取已安装 PhysX API 方法时
解析其文档中的旧转义写法。测试关闭 CUDA 可见性和 pytest 插件自动加载，CPU
单线程执行；没有安装或升级依赖。

新增测试覆盖真实适配器回调、原始 GPU 缓存张量重放、8 类未选中状态变化，以及
第 1/2/3 批失败时的部分计数。原始 GPU before/after 快照仍被严格边界检查拒绝。
独立验收还检查接触历史缺失、过期基准、时钟不一致、错误完成计数和计划批次改变。
子进程退出码 0、非零或清理文件缺失，都不能覆盖已经保存的原始失败原因。

归档中的两个完整 CPU 样例可独立复核：

| 合成 CPU 样例 | reset 尝试/通过 | 返回的控制步 | 运动子步 | 合成交互 | 验收 |
|---|---:|---:|---:|---:|---|
| cpu_example | 4 / 4 | 4 | 16 | 480 | 通过 CPU 证据检查 |
| cpu_partial_failure | 2 / 1 | 1 | 4 | 120 | 第二批未选中 root 改变，被拒绝 |

其中 reset_batches 仅表示通过边界检查的批数，reset_attempts 表示已发起的批数，
planned_reset_batches 保存计划，simulation_control_steps 记录 wrapped.step 正常
返回的数量，completed_diagnostic_transitions 为该数量乘 120。发生中途错误时，
这些实际值不再被清单中的计划值代替。失败样例的原始断言在验收理由中优先显示。
这些合成交互不计入真实诊断或生产训练预算。

原模型 SHA 仍为
`87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c`。
生产台账保持逐字节一致，SHA 仍为
`b1d224ba20a93c93c99189d700aeb145d506259f56462eb506def64f8f888189`，
保留此前 run1 的 384 条交互和 2 次真实 PPO 更新，run2/run3 未消费。本阶段没有
安装依赖、改变原模型或旧 G1/G2 证据。

新的 GPU 提案继续固定原来规模与验收范围：

| 项目 | v2 待批准范围 |
|---|---|
| 设备/环境 | GPU0 cuda:0，120 环境，1 行 shared bank，六家族/15 条件 |
| 四批 reset | [0]；[1,17,119]；0–118 的 60 个偶数 ID；全部 120 个 ID |
| 恢复采集 | 每批 1 个零动作控制步，共 4 控制步、16 个 5ms 子步、480 条只读诊断 |
| 累计分配 | 初始 120 + reset 184 = 304 张票，2 个完整块和 64 张尾块 |
| 策略/训练/PPO | 不加载策略，无训练，无 PPO 更新 |
| 资源门槛 | GPU0 可用显存至少 12288MiB，磁盘至少 4GiB，events 至多 256MiB |
| 时间限制 | 启动 120 秒，总计 480 秒，清理 15 秒 |
| 接触范围 | capacity 1048576；require_ground_positive=False，不授予承重正控制资格 |
| 失败处理 | 立即停止，不重试、不续跑、不自动扩容 |

新 attempt_id 为 `lite3-source-shared-g2-120env-20261104-v2`。拟冻结清单路径为
`/tmp/lite3_source_g2_120env_20261104_v2.proposal.json`，新的运行输出为
`/tmp/lite3_source_shared_g2_120env_20261104_v2`；该输出目录未创建。
旧 v1 批准清单无法满足新 schema、源码指纹与尝试标识的要求。

raw/g2.v2.proposal.draft.json 是开发工作草稿，remote_revision_verified=None，
status=proposed、user_approved=False。提交并确认远端一致后，才从干净版本生成
上述外部冻结提案。冻结过程不启动 GPU；真实运行须用户再次单独同意，批准副本
届时只允许修改 status/user_approved。启动命令保存在 raw/launch_command.proposal.json，
本阶段未执行它。

原始 CPU 日志、JUnit、预算前后副本、输入/依赖指纹、旧归档指纹和合成样例均已
保存。events 使用无损 gzip，provenance.json 同时绑定解压后的原始字节数和 SHA。
SHA256SUMS 覆盖原始证据、文档、复核器与 provenance，review.json 单独保存。
review.py 只在 CPU 上复核，并在临时目录解压样例，不改写旧证据或启动仿真。
Git 空白检查仅排除原始回溯文件 initial_red.log 和 initial_red.junit.xml 中保留的
尾空格，其余文件通过检查；两份原始文件仍由字节数和 SHA 完整约束。
