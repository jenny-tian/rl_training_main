# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

_PRE_RESET_CAPTURE_ENV = (
    "rl_training.online_adaptation.pre_reset_capture_env:"
    "PreResetCaptureManagerBasedRLEnv"
)

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsLite3FlatTrainerCfg",
    },
)

gym.register(
    id="MotionImitation-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.motion_imitation_env_cfg:DeeproboticsLite3MotionImitationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3MotionImitationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="FlatOnline-Deeprobotics-Lite3-v0",
    entry_point=_PRE_RESET_CAPTURE_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_online_env_cfg:DeeproboticsLite3FlatOnlineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsLite3RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsLite3RoughTrainerCfg",
    },
)

gym.register(
    id="ContinuousRough-Deeprobotics-Lite3-v0",
    entry_point=_PRE_RESET_CAPTURE_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.continuous_rough_env_cfg:DeeproboticsLite3ContinuousRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
    },
)

gym.register(
    id="Grass-Deeprobotics-Lite3-v0",
    entry_point=_PRE_RESET_CAPTURE_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.continuous_rough_env_cfg:DeeproboticsLite3GrassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
    },
)

gym.register(
    id="RockyRoad-Deeprobotics-Lite3-v0",
    entry_point=_PRE_RESET_CAPTURE_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.continuous_rough_env_cfg:DeeproboticsLite3RockyRoadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
    },
)

gym.register(
    id="OnlineAdapt-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.online_adapt_env_cfg:DeeproboticsLite3OnlineAdaptEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
    },
)

gym.register(
    id="HeldoutBoxes-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.heldout_boxes_env_cfg:DeeproboticsLite3HeldoutBoxesEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
    },
)

# Separately budgeted full-policy fine-tuning; not the small-budget adapter task.
gym.register(
    id="PolicyLearning-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.policy_learning_env_cfg:DeeproboticsLite3PolicyLearningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPORunnerCfg",
    },
)
