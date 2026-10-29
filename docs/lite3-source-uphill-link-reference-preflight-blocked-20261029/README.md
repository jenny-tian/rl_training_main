# 上坡参考点复验：GPU0 占用，停在预检

状态：`blocked_gpu0_occupied_runtime_not_started`。用户已同意[一次同条件复验方案](../lite3-source-link-reference-offline-correction-20261029/next_stage.json)。本次在干净提交 `00a422f56ec377a4c2b51adc4b62dce8829cee81` 完成输入核对，但 GPU0 资源检查未通过，**没有启动监督运行器、worker 或仿真，没有执行任何 episode**。

237 项冻结输入及 checkpoint SHA-256 全部一致，模型子模块干净，待用目录 `/tmp/lite3_source_policy_uphill_link_reference_20261029` 没有创建。工况仍为 5° 上坡 L1，geometry/state seeds 10001/20060，difficulty 0.3384848514104325。

`_gpu_preflight(0, minimum_available_mib=12288)` 实际报错：`GPU 0 already has a compute process`。随后资源快照显示 GPU0 有 4 个计算进程（PID 920157–920160），进程属主为另一用户 `mfy`；其启动时间为 2026-10-29 18:20:21，不是本次任务创建的进程。显存余量高于准入线，但既定规则同时要求目标 GPU 无计算进程。本次没有向这些进程发信号，也没有改用其他 GPU、降低准入条件或自动重试。

证据见 [预检记录](preflight_receipt.json)、[GPU 快照](gpu_snapshot.csv)、[计算进程](gpu_compute_snapshot.csv)和[进程属主](process_owners.txt)。这些诊断快照采于首次预检异常之后，不代表首次查询的精确瞬时状态；冻结的预检函数对归档快照离线回放，得到同一拒绝原因。

本次不计为第三次上坡失败：[基线账本](baseline_progress.json)逐字节保留，两次已执行的同 seed 上坡仍按历史失败记录，其余 12 个条件仍未运行。没有新的安全通过率、参考点运行验证、G3/G4 或训练结论；生产代码、策略、奖励和阈值均未变动。

用户对原单次运行的授权保留，[后续继续规格](continuation.json)没有扩大范围。待后续继续时，重新核对干净提交、原冻结输入和 GPU0 空闲条件，然后才可执行尚未使用的单次 episode；无需重复申请同一范围的授权。本轮已按预检失败停止，没有启动后台轮询。原离线准备中的 `runtime_authorized=false` 是当时的历史快照，本次新授权记录在预检记录中，不追写旧文件。
