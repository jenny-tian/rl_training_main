# 2 环境训练接线检查：运行结束，结构化证据不完整

状态：`incomplete_runtime_artifacts`，2026-11-02。
本轮唯一一次仿真已结束；**不能标记为完整验收通过**。没有重跑、扩大规模或训练。

用户对上一阶段提案回复“同意”，本次仅执行 2 环境、24 全局控制步、
最多 48 个诊断 transitions、零 PPO 更新的接线检查。每阶段完成后须等待
用户明确同意下一阶段。批准记录和运行范围在 `raw/run_manifest.json`。

## 实际执行及现有证据

- 运行代码为 `ae19a40015c81e1c772168530761124188abfe24`，启动前工作树干净，
  与远端 SHA 一致；已登记代码及 checkpoint 的 SHA-256 核对通过。
- 运行 1：PPO 根 seed 130001、geometry 30000、sampler 131001、
  reset 状态标识 40000、独立噪声 seed 132001。状态标识不是已消费的随机流。
- GPU0 启动门实测可用 **23725 MiB**，满足 12288 MiB 下限。
  仿真日志确认 `cuda:0`、2 个环境、0.005 s physics / 0.02 s control，
  三个训练专用 termination 项成功注册。地形构建约 0.279 s，场景约 1.164 s，
  simulation start 约 1.731 s；这些不是训练吞吐结果。
- 09:20:34.892–09:21:00.551（北京时间），监督器计时约 **25.66 s**，
  进程退出码 0，未触发 240 s wall timeout。
- 日志含唯一的 `wiring probe passed; PPO updates=0` 标记，且无 Python traceback。
  按 pinned 脚本的控制流，该标记位于 24 步循环和最终 timeout/reset 断言之后。
  因而日志支持“到达通过标记”，但不是逐步原始证据。
- 没有 PPO/optimizer 构造或更新，未启动 Source 训练，未访问验证/开发/封存 split。

按代码通过标记可推断：24 次全局控制步、48 个诊断 transitions，
以及注入 timeout 后 done=`[1,0]`、episode counters=`[0,24]` 的断言已经执行。
**上述是日志与锁定代码的推断，不是留存张量的独立复核结果。**

## 为什么本轮未通过完整验收

`raw/result.json` 仅保留启动时写入的 235 字节初始快照：
`status=starting`、`simulation_control_steps=0`。其中 0 是过期占位值，
不能解释为实际没有推进仿真。

逐步 raw/executed action、projection、terminal IDs、活动结束项、
terminal steps 与 exposure 台账只存在于子进程内存，最终未写盘。
因此无法从归档独立复核动作数值、逐行终止快照和采样计数；不补造缺失记录，
不据此宣称完整通过、全部安全指标通过、G3/G4 通过或 120 环境可用。

已确认脚本的最终 JSON 写入位于 `app.close()` 之后。
`raw/kit_runtime.log` 记录：09:20:54 左右进入 `SimulationApp.close`、
关闭 stage、随后执行 framework shutdown/release。进程结束后文件仍是初始快照。
结合脚本和本机库源码，归因为关闭流程结束进程，使后续写入没有执行。
本阶段只做只读定位，**未修改运行脚本、未进行复现或重跑**。
此前的 CPU 测试没有覆盖真实 Kit 关闭后的文件持久化，这也是本次暴露的验证缺口。

监督器同样只按子进程退出码返回成功，没有将 `status=starting` 识别为归档失败；
本轮独立 reviewer 已拒绝接受这个退出码作为完整成功证据。

## 资源与清理

5 秒间隔的 GPU0 采样中，总占用峰值 6176 MiB、最小可用 17949 MiB。
这是所有用户的卡级采样，不是本进程独占峰值，也不能换算训练吞吐。
Kit 初始化枚举了四张卡并运行 P2P 检查；日志中的 IOMMU/P2P 性能提示及足端
visual USD 引用警告完整保留，不以这些日志替代物理隔离或碰撞验收。

本次 supervisor PID 2026441、timeout PID/进程组 2026484、仿真 PID 2026486
在检查时均已退出，该进程组无剩余进程。运行后 GPU compute 列表中没有本次
仿真进程。其他用户任务未被终止或修改；GPU 总占用的前后差异含其他任务变化。
原始快照在 `raw/gpu_*` / `raw/compute_*`，独立清理检查在 `cleanup_check.json`。

## 归档及离线复核

`raw/` 保留本次九个原始监督/运行文件，以及 Isaac Lab 和 Kit 完整日志。
`supervisor.py` 为本次实际执行的外部监督脚本副本。
`review.py` 只读核对原始证据，不启动仿真；`review.json` 判定为
`incomplete_runtime_artifacts`，并明确分开直接证据、推断与缺失数据。
`provenance.json` 记录代码、依赖源码和原始产物哈希，`SHA256SUMS` 可校验归档。

## 下一阶段提案：仅修复报告持久化，等待同意

下一步建议只做代码修改和 CPU 验证，不启动 GPU/Isaac Sim：

1. 将完整结果写盘放到环境/应用关闭之前，用 flush/fsync 和原子替换确保落盘。
2. 每个控制步追加独立动作/终止记录，保留成功退出之前的进度，避免仅内存缓存。
3. 将环境清理和应用关闭结果独立记录；外部监督器核验最终状态、24 条记录及
   必填数据，而非仅看退出码。退出码为 0 但结果仍为 `starting` 应判失败。
4. 用 CPU 子进程模拟关闭时直接退出、异常中止及报告缺失，验证退出前证据保留，
   并回归现有训练接线检查。
5. 提交同步并汇报修复结果后暂停。**修复同意不包含仿真重跑**；是否以新目录
   再执行同等范围检查，必须在修复阶段结束后再次得到用户同意。

具体边界见 `next_stage.json`。当前阶段已停止，等待用户选择。
