# 八环境运动/reset 检查准备：CPU 通过，真实运行待同意

状态：`cpu_prepared_eight_env_runtime_pending`，2026-11-02。

本阶段按用户批准仅实现诊断、验收与 CPU 验证，**没有启动 Isaac Sim 或 GPU 任务**。
实际 Source 训练 transitions / PPO 更新为 0。既有回归含一次合成数据 CPU PPO 更新，
新运动探针与完整 CPU 替身都不构造 PPO/optimizer。

## 历史证据的复用范围

旧八环境 G2 绑定 `fba100c5ee15eb069f4a35ebd141bc2b0ecc50f3`、seed130196，
证明其配置下的 20 步零动作接触、120 次直接 reset 和一次自动 timeout 路径。
当前接触张量、接触时序、reset 快照、地面 ray caster、共享分配/碰撞组五个关键文件
与该归档哈希相同；旧 G2 的 34 个安装依赖也逐项相同，机器人子模块仍为 `b452a1f`。
具体比较见 `reuse_audit.json`。

这些证据支持复用接口和工具，不能授予新配置的运动资格。新探针使用 geometry30000、
sampler131001、actor 均值加 std=1 噪声，并覆盖实际运动、普通失败及自动 reset。
因此新建独立八环境入口/验收器，原两环境 24 步验收规则保持限定范围。

## 本阶段实现

### 接触、运动和扫描的时序

- 每次 physics simulate/fetch 后，复用 `TensorContactCapture`，在 scene.update 前
  读取完整 136 个机器人刚体与所有地面 collider 的 Cartesian pair 张量。
  保存详细接触点、count、pair/net 力矩阵；原有容量/范围/有限性/重建校验保持不变。
- 用可恢复的实例钩子在 scene.update 后采集所有 17 刚体的 link-origin 位置、
  link 线速度和四元数，保存逐 5 ms 位移与有限差分线性加速度。
  不混用 COM 速度与 link 位置；reset 之后先重建运动历史，再记录恢复首步，
  防止把跨 tile 的 reset 位移误记为物理运动或加速度。
- 基座扫描每个物理子步采集，前视与基座扫描在每个控制步后采集。
  核对共享 ground Warp target、mesh IDs、finite hits、缓存年龄、命中所属 tile；
  基座九条射线还需跟随当前 root 的 XY（最大 0.08 m）。
- 当前碰撞包络已经离开所属 tile、无效状态、错误扫描归属或跨环境接触会停止。
  只有预测越界仍交给现有控制周期 termination，作为各环境失败反馈；不放宽 guard。

### reset 边界与普通失败

对 `_reset_idx` 本身前后做比较，排除正常物理积分；比较前预读 lazy sensors，
避免把终止观测触发的合法刷新误判成 reset 跨行污染。

保存完整 before/after tensor 快照、ticket、显式 root pose 和默认 pose，核对：

- 未 reset 环境的 root/joint/action/command/history/sensor caches、filter/lifecycle
  记忆和分配票据完全不变；
- 选中环境 episode/action/history、contact/timestamp 及 filter/lifecycle 按既有契约清理；
- 选中环境 origin 与独立 sampler replay 一致，root 返回 default pose + origin；
- 每批从环境重新读取 allocator，兼容事务式 adapter 替换 allocator 实例；
- 下一物理步明确记录 recovery IDs，运动差分按 reset 后状态起算。

普通安全、接触、姿态或预测边界终止是对应环境的 episode 结果；诊断继续记录。
强制 timeout 与普通失败同一步出现时允许 compound termination，如实保存两个掩码，
纯 timeout bootstrap mask 仍为 `truncated & ~terminated`。
不要求旧平地 actor 先通过六类地形，也不删除任何家族。

### 落盘与独立验收

复用关闭前原子/fsync 结果写盘和逐控制步 JSONL。八环境使用独立 schema，
不把两环境验收规则改成“任意环境数均通过”。另存 events JSONL，逐物理子步成批
fsync，异常时保留已编码证据；事件文件有 1 GiB 上限，超额停止，不扩容重试。
NaN/Inf 故障用显式字符串记录原值，不能作为正常有限值验收。

CPU 验收器按事件顺序独立重建详细接触点到 pair/net 力、全部物理步的运动差分、
扫描归属/年龄、reset 前后快照、sampler 票据、恢复首步和 reset/transition 台账。
缺记录、记录矛盾、采集错误、退出非零、清理或资源监控失败均拒绝。

接触矩阵原文可能超过 Git 单文件限制，验收器支持无损 gzip 归档：SHA-256 始终
核对解压后的原始事件字节，不以压缩替代或重建数据。运行时仍写原始 JSONL。

独立监督器固定八环境/96 步、GPU0 显存门、120 s 启动/240 s 总时限，并复用既有
进程组清理。源码、checkpoint、35 项安装依赖和已核对远端的干净提交均须匹配。
没有用户批准或请求扩大预算时，在 GPU 查询/启动前拒绝执行。

## CPU 验证

最终整组 **410 passed，0 failed，0 skipped**，详见 `regression.junit.xml`。
其中新运动诊断 26 项，既有训练、持久化、安全、共享接触、扫描与分配/reset 回归384项。
359 条 warning 来自已安装 PhysX tensor API docstring 的转义提示，未修改安装文件。

完整 CPU 回放执行本机 pinned `ManagerBasedRLEnv.step` 的实际函数体，使用明确的
CPU physics/backend 替身，调用生产 SourceTrainingRuntime/VecEnv、安全桥、
contact capture 和新运动/reset 钩子。它验证：

- 96 控制步、384 个物理子步，接触读取发生在 scene.update 前、运动采集在其后；
- 单环境及两批不连续组的强制 timeout；另外在第 13 步注入 env3 普通失败，
  整个诊断继续而非要求旧策略先过关；
- 共 9 次 episode 结束、4 批 reset、768 个合成 transitions，独立票据/台账一致；
- reset 恢复首步与选中 spawn 校验，不变性破坏会保存边界证据后拒绝；
- 外来/缺失扫描、NaN、过期基座扫描、实际包络越界、缺失/矛盾控制记录均拒绝；
- 故障后恢复实例钩子，字节预算拒绝且不覆盖，实际入口的失败报告先于 fake close
  直接 `os._exit(0)` 保存；
- 监督器拒绝未批准 manifest，安装依赖哈希不符即拒绝，不修改安装文件。

`cpu_example/` 保存的是本轮 pytest 产生的**合成 CPU 数据**与通过结果，绝非新仿真。
其中 backend 的 `cuda:0` 和 `gpu_pair_tensor` 是为了覆盖既有张量契约而提供的合成
描述符，没有分配或运行 CUDA/PhysX。完整事件原文约 168 MB，gzip 约 1.2 MB，
原始字节 SHA 和复制来源均写入 `cpu_audit.json`。

开发过程中补齐了旧 stepping fixture 缺少的 reset recorder 空回调与最终 contact summary；
验收器正确拒绝缺失 summary。最终增加了选中 root spawn 的显式校验后重新运行整组，
上述 410 项为最终结果。没有用失败测试的中间产物作为通过证据。

## 已冻结的拟运行范围，尚未执行

| 项目 | 规格 |
| --- | --- |
| 场景 | 8 环境、1 行 shared bank、六类/15 个 L1 cell 全部保留 |
| 动作 | pinned actor 均值 + std=1 Gaussian，policy 45 / critic 393，命令 `(0.3,0,0)` |
| Seeds | PPO根130001、geometry30000、sampler131001、noise132001；reset标识40000不随机化 |
| 交互上限 | 96 全局控制步 ×8 =768诊断 transitions；4子步/控制步，共384物理子步 |
| 注入超时 | 控制步24：`[0]`；48：`[1,4,7]`；72：`[0,2,5,6]`；预置episode counter=1999 |
| 恢复窗口 | 最后一批注入后仍有24控制步；自然最后一步reset若无恢复数据，单独报告，不补步数 |
| 设备 | 物理GPU0，可共享，启动前至少12288MiB可用；可见设备映射只允许未设置或`0` |
| 时限 | initial-ready前120s；子进程总时限240s；清理宽限15s；不延时、不换卡 |
| 接触容量 / 证据上限 | 65536详细接触点容量；events JSONL上限1GiB |
| 学习 | 不构造PPO、critic或optimizer；零更新，不保存新策略 |
| 失败 | 完整性/记录错误停止；普通各环境失败记录并reset；失败不重试，不扩大到120环境 |

这至多证明指定八环境、短时运动/reset 的完整性。1.92 s/环境通常仍处于入口区域，
不要求覆盖所有15个cell，不评估9m完成、技能成功率或几何泛化；不授予完整G3/G4。
5ms离散采样也不是连续时间越界证明，GPU共享运行不提供G4独占吞吐结论。
自然在最后一步结束的回合没有恢复首步，验收会如实报告 pending IDs。

仅在用户另行明确同意后，创建全新 run manifest，填本阶段最终提交和远端SHA，
从 `input_inventory.json` 核对输入/资产/依赖，执行：

```bash
PYTHONPATH=source/rl_training \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/source_eight_env_moving_supervisor.py run \
  --manifest /tmp/lite3_source_eight_env_moving_manifest_20261102.json \
  --output-dir /tmp/lite3_source_eight_env_moving_20261102 \
  --checkpoint /home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt
```

本阶段未执行该命令。具体提案 `next_stage.json` 的批准字段保持 false。
获批运行结束后，无论结果如何，都归档详细汇报并等待下一阶段同意。
