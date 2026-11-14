# 净空奖励 500 次限额试训及配对评估

2026-11-15 用户已批准执行：原低台阶 10000 次父模型续训 500 次，并完成后续 1600 回合配对评估和两段固定起点录像。训练已在 GPU 1 启动，前 20 次更新检查通过，倍率为 1，平均每次约 7.85 秒。训练与评估守卫及相关回归共 139 项 CPU 检查通过。最终结果完成后写入本目录，当前没有新的能力验收结论。

## 固定范围

- GPU 1，4096 环境，种子 42，最多 500 次 PPO 更新、49,152,000 次环境交互，不自动延长或失败重启。
- 父检查点 SHA256 为 `419921855e24b40af058ac6642083eca8538f639e596606c51c2150ae7549a8f`，不使用 A500 或平地 1998。
- 六类分块地形、2–5 cm 台阶、原观测和 PPO 保持，只用已验收的 `persistent_clearance` 替换净空项，权重 −1。转向奖励保持。
- 使用已验收的课程恢复组件：旧模型全局等级数量精确恢复，逐环境分配按固定种子重建，首次重置不推进课程，初始倍率为 1；不宣称历史逐环境状态精确恢复。
- 保存启动、100、250、500 次检查点，嵌入课程状态。第一批 20 次检查包含在总预算内。
- 最终 500 次模型用于正式验收；不事后改选中间模型。训练回报与独立行为结果分别报告。

## 验收

复用父模型的固定参数和训练随机化两种评估条件。0/2/3/5 cm 上下阶，0.32/0.4/0.6 m/s，以及站立、平地直行、左右转向，每个条件 20 回合，合计 1600 回合。逐字段核对初始状态和公共配置；两段录像的轨迹必须与正式评估对应轨迹完全一致。

主要改进要求：固定参数 5 cm 上阶，0.32 m/s 超过 0/20，0.4 m/s 超过 4/20，0.6 m/s 不低于 14/20。单独报告 2/3 cm、下阶、姿态终止及所有转向窗口的变化；任一主要退化都不以训练回报增长抵消。左右后腿碰阶诊断使用几何和接触力代理，明确区别于精确训练奖励回放。

在得到本轮结果前固定自动筛查：其他固定参数地形的整段成功数不低于父模型，固定参数各高度姿态终止数不增加；沿用已有转向筛查，配对时间窗平均偏航误差增加超过 10%、新增配对姿态失败或缺失完整时间窗均判失败。随机化任一速度组成功数下降或姿态终止增加，即使前述检查通过也标为需要审查。这个工程筛查不等于统计显著性或部署资格。

本轮包含课程恢复与净空修改，未增加同预算旧净空对照，不能分别证明两项修改各自贡献。完成本阶段后汇报，等待新的阶段批准。

## 执行入口

从工作树根目录执行；输出目录必须不存在：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/reference_low_stairs_clearance_train.py --updates 500 --gpu 1
```

训练输出固定默认位置为 `logs/reference_six/low_stairs_clearance_trial_20261115/clearance_seed42_500/`。保留监督进程和日志，不停止其他项目；正式评估仅在训练退出状态、检查点来源、更新数和哈希验证全部通过后启动。

后续评估与归档使用以下顺序，等待器不运行训练、不延长预算：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/reference_low_stairs_clearance_evaluate.py --wait-for-training

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/tjx/anaconda3/envs/env_isaaclab/bin/python \
scripts/tools/reference_low_stairs_clearance_analyze.py
```

只有第一条成功完成才执行第二条。评估入口的固定输出为 `logs/reference_six/low_stairs_clearance_evaluation_20261115/`；已存在的结果只有通过身份、来源、协议和完成状态检查后才复用，不覆盖失败目录。
