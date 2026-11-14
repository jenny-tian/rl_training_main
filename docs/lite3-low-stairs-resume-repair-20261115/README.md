# 续训课程与奖励倍率恢复修复

本阶段执行用户批准的第一项修复：恢复地形课程和奖励倍率，验证启动重置、后续原生课程推进以及检查点保存和重新载入。没有开始抬脚奖励或转向奖励修改，也没有进行 PPO 更新。

## 已确认的问题与本次改动

旧 A 续训入口恢复 actor、critic、Adam 和学习率，但重新初始化地形课程。参考代码根据地形平均等级更新全局 `gait_level`，因而旧 A 初期多项奖励的倍率约为 0.12–0.13，经过约 180 次更新才恢复为 1。模型参数恢复正确并不能保证实际奖励系数保持一致。

新增独立恢复组件 [reference_low_stairs_resume.py](../../scripts/tools/reference_low_stairs_resume.py)，保留既有训练源码及历史证据的哈希。后续获准的新训练入口应通过 `make_resumed_runner` 创建环境包装器和训练器；已归档的旧 `reference_low_stairs_trial.py` 不再作为下一轮训练入口。

恢复组件完成以下工作：

1. 在首次原生环境重置前恢复地形等级、地形列、出生点和奖励倍率。启动时的重置不用于评估上一回合的表现，因此不推进课程；之后立即恢复参考项目的正常升级、降级逻辑。
2. 继续严格恢复 actor、critic、Adam 和自适应学习率，下一次训练编号从 10000 开始。
3. 新检查点内保存逐环境等级、地形列、地形原点、出生点、奖励倍率和固定步态参考足位。原生定期保存即使不传入 `infos`，也会自动包含这些状态。
4. 对模型与课程状态的绑定、地形配置、环境数量、地形原点、索引范围和奖励倍率进行检查。缺失或不匹配时直接报错，不静默退回低等级初始化。
5. 提供仅用于冻结验证的 [reference_low_stairs_resume_validate.py](../../scripts/tools/reference_low_stairs_resume_validate.py)，只有 `capture` 和 `reload` 模式，不提供训练模式。

## 旧检查点的恢复限制

原低台阶 10000 次模型只保存了各等级总数，没有逐环境等级、各地形分别的等级分布或当时的固定步态参考足位。这些历史信息无法补回。

本次对这个指定旧模型采用明确标记的迁移方式：保留六类地形原生列分配，用独立、固定种子的随机排列分配已记录的等级，等级总数和全局平均值精确匹配；固定步态参考足位在新回合重新初始化。不会称为逐环境精确恢复。

| 等级 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 环境数 | 113 | 163 | 250 | 310 | 386 | 452 | 570 | 605 | 666 | 581 |

总计 4096 个环境，平均等级为 5.763916015625，对应奖励倍率 1。

之后由新组件保存的检查点支持课程和固定步态参考的精确恢复。两种恢复方式都从新回合开始，不恢复上一回合的物理状态、接触历史、命令采样状态或随机数流；不能宣称中断轨迹逐步复现。

## 验证方法

CPU 检查覆盖启动重置不改变等级、之后课程恢复工作、配置不符、越界等级、出生点不符、错误倍率、模型与状态混用、异常退出后解除临时重置保护、定期保存包含课程状态，以及旧模型等级总数保持。连同既有相关检查共 62 项通过。

GPU 1 验证使用原低台阶模型、4096 环境、参考训练随机化、六类地形和 2–5 cm 台阶；没有加入 A 或 B 奖励。最终验收分两个独立进程：

- `final_capture`：从旧模型按上述规则迁移，检查首次重置，冻结运行 64 步，逐项核对 10 个受倍率影响的奖励，并显式验证一次原生升级和降级，然后保存仅用于验证的检查点。
- `final_reload`：在全新进程中载入该检查点，逐张量检查等级、列、地形原点、出生点和步态参考；再次冻结运行 64 步，检查倍率及原生升降级。

两个最终进程均通过，总耗时约 120 秒。两次初始化及连续 64 步的倍率均为 1；原生升级和降级检查通过；actor、critic、Adam 参数完全不变，Adam 步数仍为 200000。重新载入时逐环境课程状态及固定步态参考均与保存值一致。

10 项奖励在同一物理状态下的倍率检查均通过，最大比例误差低于 1e-6。以机身高度惩罚为例，旧 A 初始实际系数约 −6.705272，修复后为配置值 −50；动作变化惩罚从约 −0.013410544 恢复为 −0.1。这表示恢复了一致的奖励系数，不是已经训练出了更好的动作。

完整结果见 [verification.json](verification.json)、[首次恢复](capture_report.json) 和 [独立进程重新载入](reload_report.json)。检查点、日志和配置保存在 `logs/reference_six/low_stairs_resume_validation_20261115/`；27 个原始文件的哈希记录在 [raw_manifest.json](raw_manifest.json)。

复现命令，从本工作树根目录执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=source/rl_training:scripts/tools PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
tests/source_curriculum/test_reference_low_stairs_resume.py \
tests/source_curriculum/test_reference_low_stairs_trial.py \
tests/source_curriculum/test_reference_low_stairs.py \
tests/source_curriculum/test_reference_low_stairs_repair.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/reference_low_stairs_resume_validate.py --mode capture \
--output logs/reference_six/resume_recheck_capture

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/reference_low_stairs_resume_validate.py --mode reload \
--checkpoint logs/reference_six/low_stairs_resume_validation_20261115/final_capture/validation_checkpoint.pt \
--checkpoint-sha256 07f23875f1f4a115c33827ceaf96cb167c247ad7170baeaf1df3c7fc8e5528e7 \
--output logs/reference_six/resume_recheck_reload
```

输出目录必须不存在。上面的重新载入命令使用本次已验收的检查点；如使用另一次 `capture` 的输出，应填写其报告中记录的路径和 SHA256。

验证期间修复了脚本本身的三个问题：参考模块加载顺序、推理模式下的资产缓存写入、未调用训练时原生日志器尚未初始化。这三次未完成尝试的日志保留，并列入原始证据清单，不计入最终通过结果。`capture_v3` 首次完整通过后，补充旧模型地形配置绑定检查，再运行最终的两个进程验收。

## 下一阶段，等待用户确认

下一阶段只修复和验证抬脚奖励：让足底低于目标净空时，抬高过程具有连续、可区分的得分；检查当前停止向前摆脚和接触后奖励消失的盲区；重点覆盖后腿顶阶及左右支撑。保留原有地形、PPO 和转向奖励，并使用本阶段恢复组件。

先完成受控函数测试与冻结模型下的奖励计分验证，确认新奖励在接近、抬脚、碰阶、落脚各阶段的行为，以及平地和纯转向的适用范围。本阶段的恢复检查无法证明上阶能力已经改善；第二阶段也需要先报告实现和验证结果，再申请有限预算训练。当前不启动新的训练，也不把抬脚和转向修改合并执行。
