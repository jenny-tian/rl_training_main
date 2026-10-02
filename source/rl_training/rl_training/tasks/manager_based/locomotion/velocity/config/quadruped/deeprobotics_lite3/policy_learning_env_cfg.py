"""Full-policy learning on a fixed flat/rough mix; no reference gait or height input."""
import copy
import math
import numpy as np
import trimesh

from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass

from rl_training import policy_course
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.terrains import RockyRoadTrackCfg
from .flat_env_cfg import DeeproboticsLite3FlatEnvCfg
from .continuous_rough_env_cfg import DeeproboticsLite3RockyRoadEnvCfg


def flat_learning_track(difficulty,cfg):
    length,width=cfg.size
    mesh=trimesh.creation.box(extents=(length,width,.2))
    mesh.apply_translation((length/2,width/2,-.1))
    return [mesh],np.array([1.5,width/2,0.])


@configclass
class FlatLearningTrackCfg(SubTerrainBaseCfg):
    function=flat_learning_track


@configclass
class DeeproboticsLite3PolicyLearningEnvCfg(DeeproboticsLite3FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        reference=DeeproboticsLite3RockyRoadEnvCfg()
        self.scene.num_envs=128
        self.scene.terrain.terrain_type='generator'
        common=dict(flat_length=3.,transition_length=.5,lane_width=4.,rough_amplitude=.04,grid_spacing=.1,
                    rock_count=84,rock_fraction=.52,radius_x=(.16,.38),radius_y=(.13,.32),height_range=(.05,.10))
        # Use fixed column proportions; one row at fixed difficulty has no progression.
        self.scene.terrain.terrain_generator=TerrainGeneratorCfg(
            seed=211,curriculum=True,size=(10.,4.),border_width=0.,num_rows=1,num_cols=8,
            difficulty_range=(1.,1.),use_cache=False,
            sub_terrains={'flat':FlatLearningTrackCfg(proportion=.25),
                'rough101':RockyRoadTrackCfg(proportion=.375,seed=101,**common),
                'rough107':RockyRoadTrackCfg(proportion=.375,seed=107,**common)})
        self.scene.terrain.max_init_terrain_level=0
        self.scene.terrain.physics_material=copy.deepcopy(reference.scene.terrain.physics_material)
        self.scene.height_scanner=None
        self.scene.height_scanner_base=None
        self.commands=copy.deepcopy(reference.commands)
        self.observations.policy.enable_corruption=False
        self.observations.critic.enable_corruption=False
        self.events.randomize_rigid_body_material=copy.deepcopy(reference.events.randomize_rigid_body_material)
        self.events.randomize_reset_base.params=copy.deepcopy(reference.events.randomize_reset_base.params)
        self.events.randomize_reset_base.params['pose_range'].update(x=(-.02,.02),y=(-.02,.02),yaw=(-math.pi/90,math.pi/90))
        self.episode_length_s=40.
        self.is_finite_horizon=True
        # Deadline is a terminal failure, not a bootstrap-able time limit.
        self.terminations.time_out=DoneTerm(func=mdp.time_out,time_out=False)
        self.terminations.terrain_out_of_bounds=None
        self.terminations.illegal_contact=copy.deepcopy(reference.terminations.illegal_contact)
        self.terminations.bad_orientation_2=copy.deepcopy(reference.terminations.bad_orientation_2)
        feet=SceneEntityCfg('robot',body_names=['FL_FOOT','FR_FOOT','HL_FOOT','HR_FOOT'])
        self.terminations.course_success=DoneTerm(func=policy_course.course_success,params={'asset_cfg':feet})
        self.terminations.course_lateral=DoneTerm(func=policy_course.course_lateral_failure,params={'asset_cfg':feet})
        self.rewards.course_lateral=RewTerm(func=policy_course.course_lateral_cost,weight=-1.,params={'asset_cfg':feet})
        self.rewards.course_terminal=RewTerm(func=policy_course.course_terminal_reward,weight=1.)
        self.disable_zero_weight_rewards()
