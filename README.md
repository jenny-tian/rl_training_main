# rl_training

## Lite3 六类地形：新同学从这里开始

本分支提供 Lite3 在平地、粗糙地面、坡面、台阶、粗糙坡面和摩擦变化地形上的 Isaac Lab 训练代码，以及 B10200、zero10100、lowstairs10000 三份权重和配套配置。一份策略用于六类地形，当前发布版本的台阶高度为 2–5 cm。

**请先阅读 [六类地形新生上手指南](models/lite3_six_terrain/README.md)**，按“下载与安装 → 权重检查 → 仿真启动检查 → 3 次更新检查 → 正式训练”的顺序操作。指南包含每一步的命令、通过条件、输出位置和常见错误。

- 只查看现成权重：从指南第 4 节开始，需要已安装 PyTorch。
- 从头训练：按指南第 2–7 节准备 Isaac Lab 环境并运行。
- 阅读模型和实验实现：查看指南第 8 节的源码索引。

下面保留上游项目的通用介绍和 Rough 任务示例。**本分支六类地形训练使用指南中的 `scripts/tools/train_lite3_six_terrain.py` 入口**；下方通用 Rough 任务命令不对应这套六类地形配置。

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-5.0.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/gdM9mQutC8)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause)

## Tutorial Videos
We've released the following tutorials for training and deploying a reinforcement learning policy. Please check it out on [Bilibili](https://b23.tv/UoIqsFn) or [YouTube](https://youtube.com/playlist?list=PLy9YHJvMnjO0X4tx_NTWugTUMJXUrOgFH&si=pjUGF5PbFf3tGLFz)! 

## Overview

**rl_training** is a RL training library for deeprobotics robots, based on IsaacLab. The table below lists all available environments:

| Robot Model         | Environment Name (ID)                                      | Screenshot |
|---------------------|------------------------------------------------------------|------------|
| [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | Rough-Deeprobotics-Lite3-v0 | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="300">
| [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html) | Rough-Deeprobotics-M20-v0 | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="300">

> [!NOTE]
> If you want to deploy policies in mujoco or real robots, please use the corresponding deploy repo in [Deep Robotics Github Center](https://github.com/DeepRoboticsLab).
## Contribution 

Everyone is welcome to contribute to this repo. If you discover a bug or optimize our training config, just submit a pull request and we will look into it.
## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone --recurse-submodules https://github.com/DeepRoboticsLab/rl_training.git
  ```

- Using a python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e source/rl_training
  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

  ```bash
  python scripts/tools/list_envs.py
  ```

<details>

<summary>Setup as Omniverse Extension (Optional, click to expand)</summary>

We provide an example UI extension that will load upon enabling your extension defined in `source/rl_training/rl_training/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `rl_trainingb/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

</details>

## Try examples

Deeprobotics Lite3:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-Lite3-v0 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-Lite3-v0 --num_envs=10
```

Deeprobotics M20:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-M20-v0 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-M20-v0 --num_envs=10
```

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> ====================== ========================= ========================
> Command                Key (+ve axis)            Key (-ve axis)
> ====================== ========================= ========================
> Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
> Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
> Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
> ====================== ========================= ========================
> ```
* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint model.pt`

## Lite3 online terrain adaptation (simulation only)

The current project documentation is maintained in
[`docs/lite3-project-documentation-20260923.md`](docs/lite3-project-documentation-20260923.md)
and the implementation baseline is summarized in [`design.md`](design.md).

> [!WARNING]
> This workflow is for Isaac Lab simulation research. It does not authorize
> unprotected online exploration on a real Lite3. Real-robot use requires an
> independently validated hardware safety controller, emergency stop, operator
> supervision, timing watchdogs, and staged low-energy tests.

The adaptation stack keeps a flat PPO locomotion policy as the nominal controller
and supports two independently evaluated routes:

- **Direct LoRA-PPO:** freeze the base Actor and update only low-rank adapter
  matrices plus a separate Critic.
- **Residual SAC:** freeze PPO and learn a bounded residual under novelty and
  safety gates.

Both routes distinguish the policy request action (`a_req`) from the projected
executed action (`a_exec`). The replay/rollout contract, terminal handling,
candidate rollback, flat-retention probe and sealed-test isolation are part of
the current implementation. The latent residual, height-scan teacher and
steering-inversion utilities are available for the next RGLS experiments, but
are not yet an end-to-end validated result. FlowDAgger is not the training
algorithm used by this repository; only its latent action-inversion idea is
considered as an optional initialization reference.

Run component tests with:

```bash
pytest -q tests/online_adaptation
```

The exact simulation commands, checkpoint requirements and result-reporting
rules are documented in the project guide linked above. Use the sealed evaluator
only after a checkpoint has been selected on the development protocol:

```bash
python scripts/reinforcement_learning/online_adaptation/run_lora_ppo_sealed_test.py \
  --base-checkpoint /path/to/flat/model.pt \
  --checkpoint /path/to/selected/latest.pt \
  --output-dir artifacts/final_test/run
```

## Multi-gpu acceleration
* To train with multiple GPUs, use the following command, where --nproc_per_node represents the number of available GPUs:
    ```bash
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless 
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-Lite3-v0 --headless --distributed --num_envs=2048
    ```
* Note: each gpu will have the same number of envs specified in the config, to use the previous total number of envs, devide it by the number of gpus.
* To scale up training beyond multiple GPUs on a single machine, it is also possible to train across multiple nodes. To train across multiple nodes/machines, it is required to launch an individual process on each node.

    For the master node, use the following command, where --nproc_per_node represents the number of available GPUs, and --nnodes represents the number of nodes:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless --distributed
    ```
    Note that the port (`5555`) can be replaced with any other available port.
    For non-master nodes, use the following command, replacing `--node_rank` with the index of each machine:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless --distributed
    ```

## Tensorboard

To view tensorboard, run:

```bash
tensorboard --logdir=logs
```

## Export Policy to ONNX (without Isaac Sim)

Export a trained checkpoint to ONNX directly from the `.pt` file — no Isaac Sim or environment setup required:

```bash
# Lite3
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path logs/rsl_rl/deeprobotics_lite3_rough/<run>/model_5000.pt \
    --robot lite3 \
    --output_path exported/lite3_policy.onnx

# M20
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path logs/rsl_rl/deeprobotics_m20_rough/<run>/model_5000.pt \
    --robot m20 \
    --output_path exported/m20_policy.onnx
```

Robot metadata (joint names, stiffness/damping, default positions, action scales) is embedded in the ONNX file as model properties. Add `--no_metadata` to skip this.

## Compare Training Runs

Diff the `agent.yaml` and `env.yaml` configs between two runs (saved automatically to `params/` by `train.py`):

```bash
python scripts/tools/compare_runs.py \
    logs/rsl_rl/deeprobotics_lite3_rough/<run1> \
    logs/rsl_rl/deeprobotics_lite3_rough/<run2>
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

**Note: Replace `<path-to-isaac-lab>` with your own IsaacLab path.**

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/rl_training",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks",
    ]
}
```

### Clean USD Caches

Temporary USD files are generated in `/tmp/IsaacLab/usd_{date}_{time}_{random}` during simulation runs. These files can consume significant disk space and can be cleaned by:

```bash
rm -rf /tmp/IsaacLab/usd_*
```

## Acknowledgements

The project uses some code from the following open-source code repositories:

- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)
