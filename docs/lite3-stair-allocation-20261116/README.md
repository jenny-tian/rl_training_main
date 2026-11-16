# 台阶环境份额对照：代码准备完成，首次启动检查停止

2026-11-16。本阶段尚未完成训练或行为验收。A 组首次集成在环境首次重置前的高度扫描检查处停止，
正式环境控制步 0、PPO 更新 0、冻结评估回合 0，未生成新策略。父模型 zero10100 的 SHA256 保持不变。
失败启动耗时 44.19 s，GPU 1 总占用采样峰值 7,926 MiB；该数值包含其他驻留占用。
没有重启仿真，没有启动 B 组或训练。

## 已完成内容

- 新增确定性的环境分配与课程迁移：A 保持原分配；B 将 205 个平地环境转入上阶、205 个粗糙环境转入下阶。
- 使用目标台阶的原难度比例按等级分层选取迁移对象，保留所有 4096 个环境的原难度等级、完整地图及步态相位缓存。
- 恢复路径先核验父模型身份和原课程合同，再仅允许修改类别、对应环境原点及包含迁移清单的新合同。
- 两组训练入口均限定为 64 步零更新检查、最多 100 次 PPO 更新，并校验初始/50/100 更新检查点的 Adam 步数。
- 冻结评测入口及额外非台阶回归协议已编写。收益与退化按固定起点配对，保留低速姿态失败的统计。
- 生成 [环境分配图](allocation.png) 和 [PDF](allocation.pdf)。图来自父模型课程快照与拟迁移清单，**不是新仿真截图或训练结果**。

完整范围和停止条件见 [PLAN.md](PLAN.md)，逐环境清单见 [allocation_A.json](allocation_A.json) 与
[allocation_B.json](allocation_B.json)。这次仍只改变环境分配，不改净空裕量或落脚门槛。

## 首次启动失败与修复

新增训练入口在 `ManagerBasedRLEnv` 构造完成后，立即调用了原项目的 `audit_simulation`。
此时尚未执行 `make_resumed_runner` 包含的首次重置，原点、机器人状态及射线数据尚未按恢复课程完成同步。
该调用触发 `ValueError: nonfinite initial rays: height_scanner`。原历史训练入口在 runner 初始化后调用审计；
此次是新增入口的检查顺序问题。报错发生在迁移、runner 恢复和正式 rollout 之前。

已修正为：构造环境 → 检查原课程并迁移分配 → 恢复 runner 并完成唯一一次初始重置 → 检查传感器与碰撞网格。
新的审计保留 5,000 条地图扫描/碰撞射线比对、摩擦绑定、45/393 维观测和非有限值检查，
允许按已审核的 A/B 分配核验类别；没有关闭报错检查，也没有额外重置来绕过课程恢复。
另外增加未初始化环境拒绝进入审计的测试。

原失败目录保留于 `logs/reference_six/stair_allocation_20261116/A/integration/`。
修正后的零更新检查使用新的 `integration_v2` 目录，尚未运行。
[preparation.json](preparation.json) 保存原始执行记录、错误栈和修正后源码哈希。
**修复只通过了代码和 CPU 检查，尚未经过仿真验证，不能认定集成已通过。**

## 验证

52 项相关 CPU 测试通过，覆盖确定性分配、等级/原点/算法状态保护、保存恢复、原有恢复逻辑、
未初始化审计拒绝、预算计算、逐起点退化判定及非台阶通过条件。新增脚本通过 Python 编译及差异空白检查。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/tjx/anaconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/source_curriculum/test_reference_stair_allocation.py \
  tests/source_curriculum/test_reference_stair_allocation_evaluate.py \
  tests/source_curriculum/test_reference_low_stairs_resume.py \
  tests/source_curriculum/test_reference_low_stairs_zero_preservation.py
```

首次直接执行 pytest 时，环境的无关自动加载插件 `typeguard` 与 `typing_extensions` 存在导入冲突；
禁用插件自动加载后使用同一 Python 环境完成上述测试。未安装、升级或改动环境依赖。

## 待确认后的具体顺序

本计划约定异常后不自动重启，因此修复完成后停在这里。待用户允许重跑后：

1. GPU 1 顺序运行修正后的 A/B 零更新检查，各 4096×64 步；比对完整扫描网格、碰撞网格及材料哈希，
   检查类别迁移、出生点、碰撞隔离调用和奖励适用范围。
2. 全部通过后按原预算依次运行 A/B 各 100 次更新，仍从同一 zero10100 初始化。
3. 进行最多 6,080 回合冻结验证，遇到预先定义的硬性退化停止剩余矩阵；交付实际完成分母、逐起点结果和正式轨迹对比图。
4. 完成后详细汇报，等待下一阶段批准，不部署、不追加长期训练。

本次失败没有消耗已批准的正式 rollout 或优化更新预算，也没有得到任何新的台阶能力结论。
