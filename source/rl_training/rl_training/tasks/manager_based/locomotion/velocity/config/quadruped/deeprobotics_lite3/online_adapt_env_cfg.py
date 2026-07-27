from isaaclab.utils import configclass

from rl_training.online_adaptation.terrain_guards import assert_heldout_terrains_disabled

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3OnlineAdaptEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.observations.policy.height_scan = None
        assert_heldout_terrains_disabled(self.scene.terrain.terrain_generator)
        self.disable_zero_weight_rewards()
