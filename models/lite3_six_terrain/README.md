# Lite3：Isaac Lab 六类地形训练与权重

六类地形为：平地、多尺度粗糙地面、普通坡面（上/下坡）、台阶（上/下阶）、粗糙坡面、摩擦变化。本目录发布覆盖这些地形的统一策略，台阶高度为 **2–5 cm**。

## 权重及对应配置

| 权重 | 用途 | 配套配置 |
|---|---|---|
| [B10200.pt](weights/B10200.pt) | 当前基础策略候选，台阶环境占比约 30% | [完整配置](configs/B10200/configuration.json)、[环境分配迁移](configs/B10200/allocation_migration.json) |
| [zero10100.pt](weights/zero10100.pt) | B 的父模型与能力对照，台阶环境占比约 20% | [完整配置](configs/zero10100/configuration.json) |
| [lowstairs10000.pt](weights/lowstairs10000.pt) | 六类低台阶从头训练 10000 次 PPO 更新的检查点 | [环境 YAML](configs/lowstairs10000/params/env.yaml)、[PPO YAML](configs/lowstairs10000/params/agent.yaml)、[运行参数](configs/lowstairs10000/runtime_overrides.json) |

均为原始 PyTorch/RSL-RL 检查点，包含 actor、critic、优化器及保存的训练信息。B 和 zero 还包含课程状态。actor 输入 45 维、输出 12 维，critic 输入 393 维；策略采用 ELU MLP，隐藏层为 512/256/128。使用 `actor_state_dict` 和 `critic_state_dict`，不是旧版 RSL-RL 的 `model_state_dict` 格式。

训练使用 4096 个环境、种子 42、每次 PPO 更新 24 个控制步；Isaac Lab 策略周期为 **20 ms**。初始环境份额为平地/粗糙/坡面/台阶各 20%，粗糙坡面/摩擦变化各 10%。B 保持地图不变，将部分平地和粗糙环境迁给台阶；实际环境数以 `allocation_migration.json` 为准。

配置文件是训练时的原始快照，其中本机路径用于记录来源；在新机器运行时由代码重新解析路径。`manifest.json` 记录模型来源提交、SHA256 和打包环境版本。

## 下载与依赖

```bash
git clone --recurse-submodules --branch codex/lite3-ea-lpl-20261106 \
  https://github.com/jenny-tian/rl_training_main.git
cd rl_training_main
git clone https://github.com/DeepRoboticsLab/rl_training.git ../rl_training_reference
git -C ../rl_training_reference checkout 127b4ff1081b1c7a2f08842298d06dcc5e9460e4
```

本项目的 `deep_robotics_model` 子模块提供 Lite3 USD 资源；参考项目无需初始化机器人子模块。参考加载器核对机器人资源与参考版本对应的 Git blob。

在已安装 Isaac Sim / Isaac Lab 的 Python 环境中运行。打包时使用 Python 3.11、Isaac Sim 5.1.0.0、`isaaclab==0.54.3`、`isaaclab_rl==0.5.0`、`rsl-rl-lib==5.0.1`、`torch==2.7.0+cu128`；其余版本见 `manifest.json`。Isaac Lab 的 Python 包版本与其发行标签采用不同编号。

```bash
python -m pip install -e source/rl_training
cd models/lite3_six_terrain
sha256sum -c SHA256SUMS
cd ../..
```

## 从头训练六类低台阶策略

从仓库根目录执行；`--reference-root` 可以指向任意位置的参考 checkout。`--gpu` 支持物理 GPU 0–3，启动器检查至少 12288 MiB 空闲显存。输出目录必须尚不存在。

先执行 120 环境、100 控制步、零优化更新的启动检查：

```bash
python scripts/tools/train_lite3_six_terrain.py \
  --reference-root ../rl_training_reference \
  --mode smoke --gpu 0 --output logs/six_terrain_smoke
```

执行有明确预算的训练：

```bash
python scripts/tools/train_lite3_six_terrain.py \
  --reference-root ../rl_training_reference \
  --mode train --updates 10000 --seed 42 --gpu 0 \
  --output logs/six_terrain_train
```

入口复用原始 [reference_six_train.py](../../scripts/tools/reference_six_train.py) 的 `low_stairs` 分支，自动保存 `params/env.yaml`、`params/agent.yaml`、执行记录和权重；完整训练输出 `model_updates_10000.pt`。`--mode validate` 为 4096 环境、3 次 PPO 更新的数值检查。

以上命令对应 lowstairs10000 的从头训练方案。zero10100 与 B10200 是随后修订奖励、命令采样和环境分配后的检查点，不能把从头训练命令称为直接复现 B10200。各阶段原始训练代码与证据已随此分支保留：

| 内容 | 源码 / 记录 |
|---|---|
| 六类地形与摩擦区域 | [reference_six_terrain_cfg.py](../../scripts/tools/reference_six_terrain_cfg.py) |
| 低台阶与奖励 | [reference_low_stairs_cfg.py](../../scripts/tools/reference_low_stairs_cfg.py)、[净空奖励](../../scripts/tools/reference_low_stairs_clearance_cfg.py) |
| zero/B 的最终环境配置 | [reference_low_stairs_guarded_cfg.py](../../scripts/tools/reference_low_stairs_guarded_cfg.py) |
| zero10100 训练 | [reference_low_stairs_zero_preservation_train.py](../../scripts/tools/reference_low_stairs_zero_preservation_train.py) |
| B10200 环境分配和训练 | [reference_stair_allocation.py](../../scripts/tools/reference_stair_allocation.py)、[训练入口](../../scripts/tools/reference_stair_allocation_train.py) |
| 课程状态与优化器恢复 | [reference_low_stairs_resume.py](../../scripts/tools/reference_low_stairs_resume.py) |
| Isaac Lab 完整冻结评测 | [RESULTS.md](../../docs/lite3-stair-allocation-completion-20261116/RESULTS.md) |

历史阶段入口会检查原始实验目录、父模型与审计哈希，属于实验复现记录；本发布包提供三个关键检查点和配置，不包含所有中间检查点、保持样本库和原始日志。新训练使用上面的便携入口。

## 发布检查

在仓库根目录运行 `python scripts/tools/verify_lite3_six_terrain.py`，检查文件哈希、三个检查点的 actor/critic、优化器状态与 CPU 推理。该检查只需 PyTorch，不启动 Isaac Sim。

仿真视频和截图继续保留本地。发布内容包括源码、权重、配置和文字/JSON 验证记录。
