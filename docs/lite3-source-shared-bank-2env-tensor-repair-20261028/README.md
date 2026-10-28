# GPU 成对接触两环境实测与恢复核验

2026-10-28，运行版本 `93f1cd46278e5a0c0bf7882c0df93a20d6b8bb71`。

**仿真证据通过两环境 shared-bank smoke；监督器最终汇总未完成。** 用户已批准一次 GPU0 复测，原会话于 22:25 启动并生成结果，本次恢复只核验及归档，没有再次仿真、修改运行源码或安装依赖。

## 实测结果

- 子进程退出码 0，监督运行耗时 18.601 秒，预算 120/240 秒，采样显存峰值 2,820 MiB。
- 两环境初始位置重叠检查、垂直遮挡扫描探针通过；20 控制步、连续 80 个物理子步采集完整，81 次运行状态检查一致。
- 地面接触点累计 env0=276、env1=268，跨环境接触=0。容量 16,384，最大使用 8；成对力残差最大 7.6294e-6 N，净力残差 0。
- 实测回读抑制 setting 和状态均为 true，tensor 设备 cuda:0。在该状态下已获得真实成对 GPU 接触证据。
- 五批直接 reset 检查全部完成，连同初始化共 8 张分配票、6 次直接 reset。原始 journal 最后记录为 complete。
- 142 项源码、32 项依赖、6 项资产哈希一致。console、Kit、IsaacLab 日志未发现显式 Error、traceback、cooking 或接触容量丢弃标记。保留原有视觉 prim 引用、P2P 等 warning，未将它们描述为零警告。
- 恢复核查时 GPU0 18 MiB、利用率 0、无计算进程；无残留本次测试进程。GPU3 的其他任务未修改。

## 汇总缺失与离线复核

原目录含 result.json、execution.json 和完整 journal，但没有 outcome.json。execution.json 中退出码 0 指的是仿真子进程，不能当作监督器退出码；原监督器最终 stderr 和退出码没有保留，不能追认。

当前监督器在子进程结束后 `from source_shared_terrain_diagnostic import shared_report_passed`。同一解释器、不启动 Kit 的离线复现稳定报 `ModuleNotFoundError: No module named 'pxr'`，完整堆栈见 [复现日志](supervisor_validator_import_reproduction.log)。导入还会解析到环境中安装的另一工作树 `/home/tjx/rl_training`。这是已复现的汇总路径缺陷，与本次缺失位置吻合；由于原 stderr 未留存，不宣称已直接取得原监督器异常堆栈。

[review.py](review.py) 从当前已校验源码提取原 shared_report_passed 函数 AST，使用原 tensor_summary_passed，对保存结果执行相同验收条件，结果为 true。该操作不加载仿真、不改变验收函数、不回填原 outcome.json。另检查全部哈希、原日志、journal 和资源释放文件。详细结果见 [post_run_review.json](post_run_review.json)。

可复核命令（仅离线）：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python docs/lite3-source-shared-bank-2env-tensor-repair-20261028/review.py
```

原始目录 `/tmp/lite3_shared_bank_2env_tensor_repair_20261028` 保持原样。归档原文件逐字节一致，见 SHA256SUMS.json；新增审核文件明确属于事后复核。

## 范围与下一阶段

接触采集修复已获得本次真实 GPU 支持，但结论仅限两环境、20 零动作步的重叠/扫描/直接 reset smoke。未验证全地形重叠、单边动作运动隔离、自动终止 reset、非空策略历史、G3/G4、120 环境或 PPO。没有加载策略或训练。

下一阶段建议先做离线收尾修复：把共享结果验收函数放到不依赖 Kit/任务包的模块，监督器和诊断脚本共同引用；覆盖实际独立 Python 导入与成功/失败结果汇总，保证异常也留下最终 outcome。使用本次已归档结果测试，无需重跑 GPU。按此前逐阶段确认约定，尚未开始该代码修复；更大规模仿真仍需另行确认。
