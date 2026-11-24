# Lite3 六类地形训练与使用说明

这个项目使用 **Isaac Lab + PPO**，训练 Lite3 四足机器人在六类地形上运动，并提供已经训练好的权重和对应配置。

第一次接触本项目，建议按这个顺序操作：**下载代码 → 准备 Python 环境 → 检查现成权重 → 检查仿真启动 → 做 3 次训练更新 → 正式训练**。每一步的命令、输出和通过条件都写在下面。

## 1. 先了解项目里有什么

### 六类地形

| 地形 | 可以怎样理解 |
|---|---|
| 平地 | 普通水平地面 |
| 多尺度粗糙地面 | 不同高度起伏的地面 |
| 普通坡面 | 上坡、下坡 |
| 台阶 | 上阶、下阶，本发布版本的高度为 **2–5 cm** |
| 粗糙坡面 | 在坡面上叠加地面起伏 |
| 摩擦变化 | 地面不同区域的摩擦系数不同 |

**一份策略权重同时用于这六类地形。** 不需要给六种地形分别下载六个模型。项目中另外出现的 MuJoCo 六场景演示，是另一组评估场景；本文讲的是 Isaac Lab 训练。

### 几个必要概念

| 名称 | 在本项目中的作用 |
|---|---|
| Isaac Sim | 提供机器人与地形的物理仿真 |
| Isaac Lab | 在仿真上组织观测、动作、奖励和并行环境 |
| PPO / RSL-RL | PPO 是训练算法；RSL-RL 是执行 PPO 的代码库 |
| actor / 策略 | 根据机器人观测输出 12 个关节动作 |
| critic | 训练时估计状态价值，帮助更新策略 |
| checkpoint / 权重文件 | `.pt` 文件，保存网络参数、优化器和部分训练状态 |
| 环境配置 | 定义机器人、地形、观测、动作、奖励等 |
| PPO 配置 | 定义网络层数、学习率、每轮采样步数等 |

### 三份权重怎么选

| 文件 | 含义 | 什么时候用 |
|---|---|---|
| [B10200.pt](weights/B10200.pt) | 当前基础策略候选；台阶环境占比约 30% | 研究当前基础策略时优先查看 |
| [zero10100.pt](weights/zero10100.pt) | B 的父模型；台阶环境占比约 20% | 与 B 对比能力变化 |
| [lowstairs10000.pt](weights/lowstairs10000.pt) | 六类低台阶从头训练 10000 次 PPO 更新的结果 | 学习本文的从头训练流程 |

三份文件已经随 Git 仓库上传，合计约 20 MB。`zero10100` 是历史实验名称，其中 `zero` 指那一阶段动作保持损失的系数为 0，并不表示模型没有训练。

本文的从头训练命令对应 **lowstairs10000 的训练方案**。zero10100 和 B10200 还经历了后续奖励、命令采样和环境分配调整；运行一次从头训练命令不会直接得到同一个 B10200 文件。

## 2. 准备机器与软件环境

### 机器要求

- 使用 Linux，建议 Ubuntu 22.04；下面命令按 Bash 终端编写。
- 训练需要 NVIDIA GPU 和可用的驱动。先运行 `nvidia-smi`，确认能看到显卡。
- 当前启动器要求所选 GPU 至少有 **12288 MiB（12 GiB）空闲显存**。这是空闲显存，不是显卡标称总显存；达到门槛后仍要看实际运行结果。
- 完整训练固定使用 4096 个并行环境。CPU 权重检查不需要 GPU，也不需要启动 Isaac Sim。
- 私有仓库需要先获得 GitHub 访问权限。下载、安装依赖还需要能访问 GitHub、NVIDIA 和 PyTorch 的下载源。

### 使用已有环境

如果机器已有本项目的 Conda 环境，先激活它，不必重复安装。例如本机环境名为 `env_isaaclab`：

```bash
conda activate env_isaaclab
python --version
python -m pip show isaacsim isaaclab isaaclab_rl rsl-rl-lib torch
```

环境名以你所在机器为准。不要在他人正在使用的共享环境里直接升级依赖。版本核对后可从第 3 节开始。

### 新机器安装

本次打包机器的软件版本如下。Isaac Lab 的 Git 版本与 Python 包版本编号不同。

| 软件 | 版本 |
|---|---|
| Python | 3.11 |
| Isaac Sim | 5.1.0（包元数据可能显示 5.1.0.0） |
| Isaac Lab 源码 | `f4aa17f87e2e5db5484f0b5974918573e8918ce2`，位于 v2.3.2 之后 |
| `isaaclab` / `isaaclab_rl` | 0.54.3 / 0.5.0 |
| RSL-RL（安装包名 `rsl-rl-lib`） | 5.0.1 |
| PyTorch | 2.7.0+cu128 |

完整环境版本还可查看 [manifest.json](manifest.json)。下面采用 Conda + pip 安装，先安装好 Conda 和 Git。Isaac Sim 5.1 的 pip 安装需要 GLIBC 2.35 或以上，可以用 `ldd --version` 查看；驱动要求及系统依赖参见 [Isaac Lab 安装说明](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)。官网最新版本可能变化，本项目使用下列固定版本。

在你希望存放项目的目录执行：

```bash
conda create -n lite3_six python=3.11 -y
conda activate lite3_six
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout f4aa17f87e2e5db5484f0b5974918573e8918ce2
cd IsaacLab
./isaaclab.sh --install rsl_rl
python -m pip install rsl-rl-lib==5.0.1
cd ..
```

Isaac Sim 体积较大，安装需要一定时间和磁盘空间；首次启动按终端提示处理 NVIDIA 许可条款。以上步骤根据本机版本和对应源码安装文档整理，本次发布没有另建干净环境重装验证。

## 3. 下载项目和参考代码

下面命令会创建两个并列目录。请使用 Git 克隆，保留 `.git` 元数据，加载器需要它来核对源码与机器人资源版本。

```bash
git clone --recurse-submodules --branch codex/lite3-ea-lpl-20261106 \
  https://github.com/jenny-tian/rl_training_main.git
git clone https://github.com/DeepRoboticsLab/rl_training.git rl_training_reference
git -C rl_training_reference checkout 127b4ff1081b1c7a2f08842298d06dcc5e9460e4
cd rl_training_main
python -m pip install -e source/rl_training
```

为什么有两个仓库：`rl_training_main` 提供六类地形、阶段训练代码和权重；`rl_training_reference` 提供锁定版本的原始 Lite3 环境和 PPO 配置，训练入口会组合两者。只需安装主项目，参考项目通过路径加载。参考项目保持该提交和源码不变，其机器人子模块不需要初始化。

`--recurse-submodules` 会下载主项目所需的机器人资源。如果克隆时漏了这个选项，在 `rl_training_main` 内执行：

```bash
git submodule update --init --recursive
```

预期目录结构：

```text
你的工作目录/
├── IsaacLab/                         # 如果你自行安装了 Isaac Lab
├── rl_training_reference/            # 原始 Lite3 环境，锁定版本
└── rl_training_main/                 # 后面的命令都在这里执行
    ├── deep_robotics_model/          # Git 子模块：机器人资源
    ├── scripts/tools/               # 六类地形训练入口
    ├── source/rl_training/          # 本项目 Python 源码
    └── models/lite3_six_terrain/
        ├── weights/                 # 三份 .pt 权重
        ├── configs/                 # 各权重对应的原始配置
        ├── manifest.json            # 来源、版本和文件清单
        └── SHA256SUMS               # 下载完整性校验
```

**除非特别说明，下文所有命令均在 `rl_training_main` 仓库根目录执行，并保持 Conda 环境已激活。** `../rl_training_reference` 表示与主项目并列的参考目录。如果放在其他位置，将命令里的这个路径替换成你的实际路径即可。

安装项目后再次核对环境，确认安装依赖没有把关键版本换掉：

```bash
python -m pip show isaacsim isaaclab isaaclab_rl rsl-rl-lib torch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

训练机器上最后一项应为 `True`。

## 4. 第一次运行：先检查下载的权重

这一步只运行 CPU 推理，不会训练，也不会弹出机器人窗口。

```bash
python scripts/tools/verify_lite3_six_terrain.py
```

**通过条件：** 命令正常结束，输出 JSON 中有 `"status": "passed"`，并列出三个模型。脚本会检查文件 SHA256、网络参数是否有限、优化器状态和 256 个输入的 actor 推理结果。

如果只想检查下载文件有没有损坏，也可以执行：

```bash
(cd models/lite3_six_terrain && sha256sum -c SHA256SUMS)
```

每行应显示 `OK`。括号表示在临时子 shell 中切换目录，执行完后你仍在仓库根目录。

如果只想读取权重内容，在已安装 PyTorch 的环境中执行：

```bash
python - <<'PY'
import torch
checkpoint = torch.load(
    "models/lite3_six_terrain/weights/B10200.pt",
    map_location="cpu",
    weights_only=True,
)
print("保存的内容：", list(checkpoint))
print("actor 第一层形状：", checkpoint["actor_state_dict"]["mlp.0.weight"].shape)
print("迭代编号：", checkpoint["iter"])
PY
```

B 的第一层形状应为 `[512, 45]`，迭代编号为 `10199`，对应完成 10200 次更新。文件使用 `actor_state_dict` / `critic_state_dict`，不能按旧版 `model_state_dict` 键读取。

## 5. 第二次运行：检查仿真能否启动

先查看 GPU 的空闲显存：

```bash
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv
```

选择空闲显存至少 12288 MiB 的卡。下面以 GPU 0 为例；如果使用 GPU 2，就把 `--gpu 0` 改为 `--gpu 2`。启动器会自动设置 GPU 可见性，不需要另写 `CUDA_VISIBLE_DEVICES`。

```bash
python scripts/tools/train_lite3_six_terrain.py \
  --reference-root ../rl_training_reference \
  --mode smoke --gpu 0 --output logs/six_terrain_smoke
```

`smoke` 是启动检查：使用 120 个环境运行 100 个控制步，输入零动作，**不做 PPO 更新**。它检查机器人资源、地形、碰撞与观测能否正常创建，不用于判断机器人走得好不好。当前入口固定为 headless 模式，没有图形窗口是正常的。

终端先打印进程号和日志路径。另开一个终端，进入同一仓库，可查看运行日志：

```bash
tail -f logs/six_terrain_smoke/runtime.log
```

在这个查看日志的终端按 `Ctrl+C`，只会退出 `tail`。在启动训练的终端按 `Ctrl+C`，则会中断任务。

**通过条件：** 主命令正常结束，`execution.json` 的 `status` 为 `completed`、`exit_code` 为 `0`；`report.json` 的 `status` 为 `passed`：

```bash
python - <<'PY'
import json
from pathlib import Path
directory = Path("logs/six_terrain_smoke")
for filename in ("execution.json", "report.json"):
    record = json.loads((directory / filename).read_text())
    print(filename, record["status"], record.get("exit_code"))
PY
```

`report.json` 没有 `exit_code` 字段，因此第二行最后显示 `None` 属于正常情况。

## 6. 第三次运行：检查 PPO 是否能更新

启动检查通过后执行：

```bash
python scripts/tools/train_lite3_six_terrain.py \
  --reference-root ../rl_training_reference \
  --mode validate --gpu 0 --output logs/six_terrain_validate
```

这次会使用 **4096 个环境、执行 3 次 PPO 更新**，检查数值是否正常以及权重是否实际更新。通过条件与上一步相同，另需 `report.json` 中 `completed_updates` 为 `3`。此时会有检查点输出，但 3 次更新的模型还没有学会走路。

## 7. 正式从头训练

前两项仿真检查通过后，再启动完整训练：

```bash
python scripts/tools/train_lite3_six_terrain.py \
  --reference-root ../rl_training_reference \
  --mode train --updates 10000 --seed 42 --gpu 0 \
  --output logs/six_terrain_train
```

| 参数 | 含义 |
|---|---|
| `--reference-root` | 指定参考代码目录 |
| `--mode train` | 正式训练模式，从随机初始化开始 |
| `--updates 10000` | 最多执行 10000 次 PPO 更新；允许范围为 1–10000 |
| `--seed 42` | 随机种子，用于控制随机初始化与采样 |
| `--gpu 0` | 物理显卡编号，支持 0、1、2、3 |
| `--output` | 本次运行的输出目录，必须尚不存在 |

每轮更新采集 4096 × 24 = 98304 次环境交互；10000 轮合计 983040000 次。这里“10000 轮”指 PPO 更新次数，不是 10000 个机器人回合。完整训练需要持续运行较长时间，应根据 GPU 和实际日志判断进度；入口在预算完成或达到 48 小时监督时限时停止。

可以先用 `--updates 100` 熟悉流程，但这会从头训练一个新模型。当前便携入口没有 `--resume` 参数；再次执行不会自动接着上一次训练。每次尝试用新的输出目录，例如 `logs/six_terrain_train_002`。

训练期间可查看文本进度：

```bash
tail -f logs/six_terrain_train/runtime.log
```

可另开终端，在同一 Python 环境下查看 TensorBoard：

```bash
tensorboard --logdir logs/six_terrain_train/training --host 127.0.0.1 --port 6006
```

在同一台电脑浏览器打开 `http://127.0.0.1:6006`。如果运行在远程服务器，可在自己电脑执行 `ssh -L 6006:localhost:6006 用户名@服务器地址`，保持连接后再打开本地浏览器；用户名和地址需替换。

### 训练结果保存在哪里

```text
logs/six_terrain_train/
├── runtime.log                   # 训练输出、异常信息
├── execution.json                # 是否正常结束、退出码、耗时
├── report.json                   # 更新次数、损失、地形等级等
├── params/
│   ├── env.yaml                  # 本次实际使用的环境配置
│   └── agent.yaml                # 本次实际使用的 PPO 配置
├── training/                     # TensorBoard 与定期检查点
└── model_updates_10000.pt         # 完成 10000 次更新后的最终权重
```

**完成条件：** `execution.json` 显示 `completed` 且退出码为 0，`report.json` 显示 `passed` 且 `completed_updates` 为 10000，同时最终权重存在。若使用其他更新预算，最终文件名会随预算变化。只看到一个中途保存的 `.pt` 文件，不代表完整训练已经结束。

## 8. 配置和源码应该从哪里看

先看 [lowstairs10000 环境 YAML](configs/lowstairs10000/params/env.yaml) 和 [PPO YAML](configs/lowstairs10000/params/agent.yaml)，了解参数；再沿下面的顺序看实现。

| 顺序 | 文件 | 重点 |
|---|---|---|
| 1 | [train_lite3_six_terrain.py](../../scripts/tools/train_lite3_six_terrain.py) | 命令行参数、参考路径、运行预算 |
| 2 | [reference_six_train.py](../../scripts/tools/reference_six_train.py) | 环境创建、PPO 更新、保存日志和权重 |
| 3 | [reference_lite3_loader.py](../../scripts/tools/reference_lite3_loader.py) | 加载锁定版本的 Lite3 环境 |
| 4 | [reference_six_terrain_cfg.py](../../scripts/tools/reference_six_terrain_cfg.py) | 六类地形、摩擦区域和地形份额 |
| 5 | [reference_low_stairs_cfg.py](../../scripts/tools/reference_low_stairs_cfg.py) | 2–5 cm 台阶与低台阶奖励 |
| 6 | [reference_low_stairs_rewards.py](../../scripts/tools/reference_low_stairs_rewards.py) | 低台阶奖励的具体计算 |

几个常用参数：actor 为 `45 → 512 → 256 → 128 → 12` 的 ELU 网络；critic 输入 393 维；控制周期为 20 ms。actor 的 45 维输入依次包含机身角速度、投影重力、速度命令、关节位置偏差、关节速度和上一步动作。每组量的顺序、缩放和关节排列均由环境配置定义，不能直接把任意 45 个数用于机器人控制。

**修改 YAML 快照不会改变训练。** 本入口使用 Python 配置构造环境，`configs/` 和运行目录下的 YAML/JSON 是导出的记录。要改变训练，应先找到相应 Python 配置，在主项目中修改，再用新输出目录执行检查；参考仓库源码会被完整性检查，保持不变。

### 研究 zero10100 / B10200 时再看

| 内容 | 入口 |
|---|---|
| zero/B 的最终环境与命令配置 | [reference_low_stairs_guarded_cfg.py](../../scripts/tools/reference_low_stairs_guarded_cfg.py) |
| 后续净空奖励 | [reference_low_stairs_clearance_cfg.py](../../scripts/tools/reference_low_stairs_clearance_cfg.py) |
| zero10100 原始训练入口 | [reference_low_stairs_zero_preservation_train.py](../../scripts/tools/reference_low_stairs_zero_preservation_train.py) |
| B 的环境分配与原始训练入口 | [reference_stair_allocation.py](../../scripts/tools/reference_stair_allocation.py)、[reference_stair_allocation_train.py](../../scripts/tools/reference_stair_allocation_train.py) |
| 课程状态与优化器恢复 | [reference_low_stairs_resume.py](../../scripts/tools/reference_low_stairs_resume.py) |
| B 完整配置与分配记录 | [configuration.json](configs/B10200/configuration.json)、[allocation_migration.json](configs/B10200/allocation_migration.json) |
| zero 完整配置 | [configuration.json](configs/zero10100/configuration.json) |
| 已有 Isaac Lab 冻结评测 | [RESULTS.md](../../docs/lite3-stair-allocation-completion-20261116/RESULTS.md) |

历史阶段入口依赖原始实验目录、父模型、保持样本库和审计记录。本发布包包含三个关键检查点及对应配置，不包含全部历史运行产物；因此这些入口供追溯实验实现，不是新机器上一条命令复现 B 的入口。

## 9. 常见问题

| 现象 | 处理方法 |
|---|---|
| GitHub 提示 `Repository not found` / 无权限 | 确认已获得私有仓库权限，并使用自己的 GitHub 凭据 |
| 找不到脚本或 `source/rl_training` | 先确认当前终端在 `rl_training_main` 根目录 |
| 找不到 `isaaclab` / `torch` | 检查 `which python`，确认已激活安装依赖的 Conda 环境 |
| `nvidia-smi` 无法连接驱动，或 CUDA 为 `False` | 检查驱动与当前 Python 的 CUDA 版 PyTorch，先解决环境问题 |
| `require >=12288` / 显存不足 | 选择其他空闲 GPU 或等待已有任务结束；不要停止他人的进程 |
| `FileExistsError` | `--output` 目录已存在，换一个新目录名，保留旧日志 |
| 提示参考版本不匹配 | 在参考仓库执行第 3 节的固定提交 checkout，确认没有改动源码 |
| 找不到 `Lite3.usd` | 在主仓库执行 `git submodule update --init --recursive` |
| SHA256 校验失败 | 下载文件可能不完整或被改动，先核对 `git status`，再重新下载对应文件 |
| 没有弹出机器人窗口 | 该训练入口固定为 headless，无窗口属正常现象，用日志和 TensorBoard 看进度 |
| 运行失败但终端没有详细错误 | 查看本次输出目录的 `runtime.log`、`execution.json`，以及存在时的 `error.json` |
| 想直接给普通 `play.py` 传 B10200 | 本文尚未提供便携的六类地形回放入口；普通 Rough 任务不会自动重建这套训练配置。先按第 4 节检查模型，已有能力结果见第 8 节报告 |

反馈问题时，请附上：你执行的完整命令、Python 环境名、GPU 编号、输出目录，以及 `runtime.log` 末尾的错误内容。这样可以直接定位问题。

## 10. 已验证内容

本发布的三份权重哈希和 CPU 推理通过，六类地形、低台阶与分配相关的 34 项测试通过。发布时未重新运行仿真，详细检查见 [VALIDATION.md](VALIDATION.md)；已有完整训练与评测记录见第 8 节。视频和截图保留本地，GitHub 提供代码、权重、配置和文字/JSON 记录。
