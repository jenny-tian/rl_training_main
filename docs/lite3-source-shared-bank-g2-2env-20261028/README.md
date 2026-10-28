# 共享 bank 两环境 G2 实测通过

2026-10-28。状态：`shared_bank_g2_2env_passed`。

用户批准后，从干净提交 `2134d87b9db03263a6aef7335088c8ef98bc0f94` 执行一次 GPU0 检查。**仿真子进程和监督器均退出 0，最终 outcome 正常生成，汇总异常为空。** 没有自动重试、运行中修复、加载策略或训练。

## 执行与结果

原命令：

```bash
/home/tjx/anaconda3/envs/env_isaaclab/bin/python scripts/tools/source_parallel_terrain_startup_supervisor.py \
  --shared-bank --reset-lifecycle --num-envs 2 --seed 130196 --gpu-index 0 \
  --startup-timeout-s 120 --total-timeout-s 240 \
  --run-dir /tmp/lite3_shared_bank_g2_2env_20261028
```

配置、时限与 [准备方案](../lite3-source-shared-bank-g2-preparation-20261028/README.md) 一致。143 项源码/审计、34 项依赖、6 项资产核对一致。模型版本 `b452a1f404d37155400242f18e454fed8bcc51b8`。

- 两环境／一个共享 bank 行／15 列，初始机器人重叠及扫描遮挡探针通过。
- 前 20 个零动作控制步完整采集 80 物理子步；地面接触点累计 276/268、跨环境接触 0。接触容量 16,384，使用峰值 8；接触点与成对力最大残差 7.6294e-6 N，净力残差 0。
- 90 批、120 次直接 reset 全部通过，输入包含单环境及乱序整批。随后仅环境 0 触发注入的 timeout，恰好一批自动 reset；再后的首个物理步检查通过。
- 91 个 reset 紧邻边界的未 reset 状态检查通过，分配逐票等于独立 oracle。123 张票据序号连续：一个完整 120-ticket block 的全部 15 类计数一致，另有 3 张尾票单独报告。
- 两套扫描每次直接 reset 后刷新通过；自动 reset 和后续物理步也核对地面命中。当前配置 observation history 清单为空，不宣称真实非空策略历史已验证。
- 监督运行耗时 **21.351 秒**；环境构建 2.595 秒；每秒采样 GPU 占用峰值 **2,820 MiB**；仿真进程 CPU 峰值 RSS 4,061,444 KiB。均在原 120/240 秒预算内。
- 退出后 GPU0 **18 MiB、0% 利用率、无计算进程**；本次测试进程无残留。GPU3 原有任务未操作。

## 证据核验

[结果](result.json)、[监督器 outcome](outcome.json)、[执行记录](execution.json)、[运行 manifest](manifest.json)、[接触 journal](result.shared.jsonl)、[reset journal](result.shared.resets.jsonl)、[事后复核](post_run_review.json)。

复核不仅检查通过字段，还核对 91 次 actual/expected assignment、123 连续票号、完整块配比和尾块、90 次扫描刷新、唯一 `[0]` timeout 及最终 post-reset-step 记录。控制台、Kit 与 IsaacLab 日志无显式 Error、traceback、cooking 或接触丢弃标记；控制台/Kit 各保留 16 条已有视觉引用、P2P 等 warning，未宣称无警告。

[review.py](review.py) 可在运行源码版本上离线重复核验，输出单独 post_run_review，不启动仿真。原始 `/tmp` 文件保留，归档副本逐字节相同；SHA256SUMS.json 列出归档哈希。本次未改变运行代码。

## 结论边界与下一步

本次通过的是 **GPU0、两环境、单行 shared bank 的有界 G2 reset 检查**，也确认此前修复的监督器汇总能够在真实运行后正常完成。

接触证据只覆盖前 20 个控制步；G2 末尾两步没有成对接触采集，不能把“跨环境接触 0”扩展到整个运行。注入 timeout 不覆盖全部自然终止路径；两个环境不覆盖较大规模离散 ID/大批 reset；没有冻结策略行走、全地形运动隔离、G3/G4、吞吐或 PPO 资格。

建议下一阶段按逐级扩容顺序，先做 **共享 bank 8 环境启动/reset 检查的离线准备**：显式扩展当前仅两环境的参数/判据、全部 8 环境刚体成对过滤与阳性对照、单个/不连续/乱序/较大 reset 批次及完整票据块；冻结诊断种子、资源/接触容量预算和失败停止规则。不能直接把现有命令的 num-envs 改为 8。本阶段没有据此修改代码或启动扩容，下一阶段等待用户确认。
