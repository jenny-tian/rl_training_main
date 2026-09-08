"""Lite3-adapted flat-terrain PPO configuration.

This task deliberately contains no motion reference, gait phase, foot-height
trajectory, mirror reward, or AMP-style prior.  PPO discovers a gait from
velocity tracking, foot air time, and Lite3-specific physical regularization.
The public Unitree Go2 task is used only as a structural reference.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as reward_mdp

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


# Lite3 flat reward scales.  The structure follows velocity-control PPO, but
# weights are calibrated to Lite3's smaller body and actuator model rather
# than copied from Go2.  Terms that do not appear here are disabled.
LITE3_FLAT_REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 4.0,
    "track_ang_vel_z_exp": 1.5,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.25,
    "flat_orientation_l2": -10.0,
    # The USD standing seed is z=0.375 m.  A stronger term prevents the
    # smaller Lite3 from exploiting a crouched, front-leg-dominant posture.
    "base_height_l2": -25.0,
    "joint_torques_l2": -2.5e-4,
    # Unitree's numerical scale is too strong for Lite3's USD actuator model;
    # retain the same term but use the Lite3-proven physical scale.
    "joint_acc_l2": -1.0e-8,
    "feet_air_time": 2.0,
    # Phase-free physical balance: all four Lite3 feet should have comparable
    # air/contact durations, without prescribing a trot sequence or reference.
    "feet_air_time_variance": -5.0,
    "undesired_contacts": -1.0,
    "feet_slide": -0.05,
    "foot_impact_velocity": -0.5,
    "contact_forces": -0.01,
    "action_rate_l2": -0.05,
    "joint_pos_limits": -5.0,
    "stand_still": -0.2,
}


@configclass
class DeeproboticsLite3FlatEnvCfg(DeeproboticsLite3RoughEnvCfg):
    """Flat Lite3 task derived from the Unitree Go2 velocity task."""

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain has no terrain-level curriculum to drive the shared
        # reward gate.  Keep all physical regularizers active from the first
        # rollout; otherwise their terms are silently multiplied by the
        # rough-task default gait_level=0.0.
        reward_mdp.gait_level = 1.0

        # ------------------------------ Scene ------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        self.scene.terrain.physics_material.restitution = 0.0
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.robot.soft_joint_pos_limit_factor = 0.9

        # --------------------------- Observations --------------------------
        # Lite3's deployed runner supplies 45 proprioceptive values and has no
        # base-linear-velocity estimator.  Keep that term disabled for the
        # actor; the critic may still use privileged base velocity below.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.motion_phase = None
        self.observations.policy.motion_reference = None
        self.observations.policy.height_scan = None

        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.motion_phase = None
        self.observations.critic.motion_reference = None
        self.observations.critic.height_scan = None

        # Keep Lite3's smaller HipX scale and real-robot safety clips.  The
        # Go2 scalar 0.25 is not physically interchangeable with Lite3.
        self.actions.joint_pos.scale = {
            ".*_HipX_joint": 0.125,
            "^(?!.*_HipX_joint).*": 0.25,
        }

        # ----------------------------- Commands ----------------------------
        command = self.commands.base_velocity
        command.resampling_time_range = (10.0, 10.0)
        command.rel_standing_envs = 0.1
        command.rel_heading_envs = 1.0
        # Reserve samples for standing and one-axis motions.  This gives PPO
        # explicit low-speed/forward examples while retaining mixed SE(2)
        # commands, instead of relying on a rare draw near zero.
        command.rel_zero_vel_envs = 0.1
        command.rel_only_lin_y_envs = 0.1
        command.rel_only_lin_x_envs = 0.2
        command.rel_only_ang_z_envs = 0.1
        command.planar_velocity_deadband = 0.05
        command.only_lin_x_magnitude_range = (0.08, 0.25)
        command.heading_command = True
        command.heading_control_stiffness = 0.5
        command.debug_vis = False
        command.command_acceleration_limits = None
        # Match the command limits used by Lite3_rl_deploy.
        command.ranges.lin_vel_x = (-0.8, 0.8)
        command.ranges.lin_vel_y = (-0.8, 0.8)
        command.ranges.ang_vel_z = (-0.8, 0.8)
        command.ranges.heading = (-3.14, 3.14)

        # --------------------------- Randomization -------------------------
        # Reproduce the Go2 task: randomized friction, randomized reset state,
        # and a velocity push every 15 seconds.  Do not add the stronger mass,
        # inertia, COM, gain, or external-force randomization used by rough.
        material = self.events.randomize_rigid_body_material
        material.params["static_friction_range"] = (0.7, 1.1)
        material.params["dynamic_friction_range"] = (0.7, 1.1)
        material.params["restitution_range"] = (0.0, 0.0)
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        # Start every rollout from the physical default pose.  Go2 tolerates
        # wide reset scaling; Lite3 needs a stable standing seed before reset
        # randomization is reintroduced.
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (-0.2, 0.2),
                    # Lite3's smaller actuator and body inertia need a gentler
                    # initial angular perturbation than Go2 to avoid every
                    # early rollout terminating before PPO sees a gait.
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.2, 0.2),
            },
        }
        # Pushes are introduced in the later robustness task.  They are not
        # useful while the first flat gait is still learning to stand.
        self.events.randomize_push_robot = None

        # ------------------------------ Rewards ----------------------------
        for name in dir(self.rewards):
            term = getattr(self.rewards, name)
            if hasattr(term, "weight"):
                term.weight = 0.0
        for name, weight in LITE3_FLAT_REWARD_WEIGHTS.items():
            getattr(self.rewards, name).weight = weight

        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["target_height"] = 0.375
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_air_time.params["command_threshold"] = 0.05
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.params["threshold"] = 0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_(THIGH|SHANK)"]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.foot_impact_velocity.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["command_name"] = "base_velocity"
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names

        # -------------------------- Terminations ---------------------------
        self.terminations.terrain_out_of_bounds = None
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
                "threshold": 1.0,
            },
        )
        # The deployment safety contract remains stricter; training allows a
        # little more transient tilt so PPO receives useful recovery rollouts.
        self.terminations.bad_orientation_2.params["max_xy_norm"] = 0.75

        # Unitree's Go2 task has neither terrain/command nor gait curricula.
        self.curriculum.terrain_levels = None
        self.curriculum.gait_level = None
        self.curriculum.command_levels = None

        if self.__class__.__name__ == "DeeproboticsLite3FlatEnvCfg":
            self.disable_zero_weight_rewards()
