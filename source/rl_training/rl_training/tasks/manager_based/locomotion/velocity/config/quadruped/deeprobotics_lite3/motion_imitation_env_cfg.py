"""Dedicated flat low-level motion-imitation task for Lite3."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import DeeproboticsLite3FlatEnvCfg


REFERENCE_REWARD_WEIGHTS = {
    "motion_reference_base_lin_vel": 2.0,
    "motion_reference_base_ang_vel": 0.8,
    "motion_reference_joint_pos": 1.4,
    "motion_reference_projected_gravity": 0.8,
}

# The paper's four state-tracking terms assume a physically consistent robot
# demonstration.  Our animal-mocap retarget is kinematic, so explicit foot and
# contact targets are required to prevent a high-reward skating solution.
FOOT_REFERENCE_REWARD_WEIGHTS = {
    "motion_prior_contact": 1.5,
    "motion_prior_clearance": 2.0,
}

FLAT_MEMORY_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class DeeproboticsLite3MotionImitationEnvCfg(DeeproboticsLite3FlatEnvCfg):
    """Pre-train one flat walking skill before velocity or residual learning."""

    # Kept on this legacy, explicitly selected task only.  The formal flat PPO
    # task has no reference-data or motion-prior interface.
    motion_prior_path: str | None = None
    motion_prior_mode: str = "off"
    motion_prior_weight: float = 1.0
    motion_prior_decay_steps: int = 100_000

    # Reference-state initialization is useful once the controller is stable,
    # but a lower probability keeps the first phase of imitation close to the
    # robot's known default pose.
    reference_state_probability: float = 0.5
    reference_future_steps: tuple[int, ...] = (1, 2, 10, 30)

    def __post_init__(self):
        super().__post_init__()

        # Maximum of the short-to-long episode curriculum: 600 policy steps.
        self.episode_length_s = 12.0
        # Avoid Nucleus USD dependencies for this flat-only pre-training task.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_MEMORY_TERRAIN_CFG
        self.scene.terrain.visual_material = None
        self.scene.sky_light = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.0
        self.commands.base_velocity.rel_only_lin_y_envs = 0.0
        self.commands.base_velocity.rel_only_lin_x_envs = 0.0
        self.commands.base_velocity.rel_only_ang_z_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.resampling_time_range = (12.0, 12.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Motion imitation starts without dynamics perturbations. They are
        # introduced only after all four legs track the reference in rollout.
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_reset_joints = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_base = None
        self.events.randomize_push_robot = None
        self.events.motion_reference_reset = None

        self.curriculum.terrain_levels = None
        self.curriculum.gait_level = None
        self.curriculum.command_levels = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.time_out = DoneTerm(
            func=mdp.motion_imitation_time_out,
            time_out=True,
            params={"start_steps": 20, "end_steps": 600, "curriculum_steps": 7_500},
        )

        # Remove inherited velocity, standing, trot and hand-authored phase
        # objectives. The low-level task is driven only by reference state.
        for name in dir(self.rewards):
            term = getattr(self.rewards, name)
            if hasattr(term, "weight"):
                term.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.applied_torque_limits.weight = -0.2
        self.rewards.applied_torque_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_torques_l2.weight = -2.0e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        # Keep the physical regularizers at the scale used by the stable Lite3
        # velocity task.  The previous -7e-5 term dominated all reference
        # rewards and encouraged the policy to fall/terminate to avoid motion.
        self.rewards.joint_acc_l2.weight = -1.0e-8
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.flat_orientation_l2.weight = -10.0
        self.rewards.tilt_guard_band.weight = -10.0
        self.rewards.tilt_guard_band.params["guard_xy_norm"] = 0.55
        self.rewards.tilt_guard_band.params["max_xy_norm"] = 0.65
        self.rewards.contact_forces.weight = -0.005
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.observations.policy.motion_phase = None
        self.observations.critic.motion_phase = None
        self.observations.policy.motion_reference = None
        self.observations.critic.motion_reference = None
        self.configure_motion_prior_terms()
        self.disable_zero_weight_rewards(
            preserve_zero_weight_terms=tuple(REFERENCE_REWARD_WEIGHTS) + tuple(FOOT_REFERENCE_REWARD_WEIGHTS)
        )

    def configure_motion_prior_terms(self, prior_terms=None):
        """Bind CLI reference settings to all imitation MDP terms."""
        enabled = bool(self.motion_prior_path) and self.motion_prior_mode != "off"
        for name, weight in REFERENCE_REWARD_WEIGHTS.items():
            term = getattr(self.rewards, name)
            term.weight = weight if enabled else 0.0
            term.params["reference_path"] = self.motion_prior_path
            term.params["mode"] = self.motion_prior_mode
            term.params["prior_weight_value"] = self.motion_prior_weight
            term.params["decay_steps"] = self.motion_prior_decay_steps

        for name, weight in FOOT_REFERENCE_REWARD_WEIGHTS.items():
            term = getattr(self.rewards, name)
            term.weight = weight if enabled else 0.0
            term.params["reference_path"] = self.motion_prior_path
            term.params["mode"] = self.motion_prior_mode
            term.params["prior_weight_value"] = self.motion_prior_weight
            term.params["decay_steps"] = self.motion_prior_decay_steps

        foot_weights = (1.0, 1.0, 1.0, 1.0)
        self.rewards.motion_prior_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.motion_prior_contact.params["foot_weights"] = foot_weights
        self.rewards.motion_prior_clearance.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=self.foot_link_name, preserve_order=True
        )
        self.rewards.motion_prior_clearance.params["foot_weights"] = foot_weights
        self.rewards.motion_prior_clearance.params["std"] = 0.05
        self.rewards.motion_prior_clearance.params["relative_height"] = True
        self.rewards.motion_reference_joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # The paper's sum-of-squares exponent is intentionally softened for
        # the retargeted mocap, whose joint offsets are not exact simulator
        # demonstrations.  This keeps the joint imitation signal visible to
        # PPO instead of collapsing to ~0 at initialization.
        self.rewards.motion_reference_base_lin_vel.params["error_scale"] = 0.5
        self.rewards.motion_reference_base_ang_vel.params["error_scale"] = 0.25
        self.rewards.motion_reference_joint_pos.params["error_scale"] = 0.15
        self.rewards.motion_reference_projected_gravity.params["error_scale"] = 1.0

        if not hasattr(self.events, "motion_reference_reset"):
            return
        if not enabled:
            self.events.motion_reference_reset = None
            self.observations.policy.motion_phase = None
            self.observations.critic.motion_phase = None
            self.observations.policy.motion_reference = None
            self.observations.critic.motion_reference = None
            return

        joint_cfg = SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True)
        self.events.motion_reference_reset = EventTerm(
            func=mdp.reset_from_motion_reference,
            mode="reset",
            params={
                "asset_cfg": joint_cfg,
                "reference_path": self.motion_prior_path,
                "reference_state_probability": self.reference_state_probability,
            },
        )
        phase_term = ObsTerm(
            func=mdp.motion_reference_phase,
            params={"reference_path": self.motion_prior_path},
            clip=(-1.0, 1.0),
        )
        target_term = ObsTerm(
            func=mdp.motion_reference_joint_targets,
            params={
                "asset_cfg": joint_cfg,
                "reference_path": self.motion_prior_path,
                "future_steps": self.reference_future_steps,
            },
            clip=(-3.0, 3.0),
        )
        self.observations.policy.motion_phase = phase_term
        self.observations.critic.motion_phase = phase_term
        self.observations.policy.motion_reference = target_term
        self.observations.critic.motion_reference = target_term
