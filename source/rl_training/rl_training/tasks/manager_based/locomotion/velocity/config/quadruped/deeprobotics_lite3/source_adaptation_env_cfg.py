"""Opt-in reward correction for the next Source experiment; no training is launched here."""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from rl_training.source_scan.adaptation_profile import (
    SOURCE_ADAPTATION_PROFILE,
    build_source_adaptation_rewards,
    validate_source_adaptation_config,
)
from .source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg


@configclass
class DeeproboticsLite3SourceAdaptationEnvCfg(DeeproboticsLite3SourceTrainingEnvCfg):
    """Keep the existing L1 dynamics and safety, with a pinned low-speed objective."""

    source_parallel_sampler_enabled: bool = True
    source_parallel_sampler_num_envs: int = 120
    source_shared_bank_rows: int = 1
    source_reward_profile: str = SOURCE_ADAPTATION_PROFILE
    reward_gait_level: float = 1.0

    def __post_init__(self):
        # configclass.replace() may supply this instance's final reward dict.
        # Rough.__post_init__ expects the full attribute-based configuration.
        self.rewards = RewardsCfg()
        super().__post_init__()
        # The inherited Source constructor removes zero-weight Rough terms.
        # Build a fresh explicit set after it, so removed flat terms are restored.
        self.rewards = build_source_adaptation_rewards(RewardTermCfg, SceneEntityCfg)
        validate_source_adaptation_config(self)
