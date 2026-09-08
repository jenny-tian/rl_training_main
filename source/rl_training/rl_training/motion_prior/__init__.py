"""Motion-prior trajectory ingestion, retargeting, and reference sampling."""

from .schema import MotionTrajectory
from .manager import MotionPriorManager, MotionPriorSample, prior_weight
from .retarget import (
    DEFAULT_GO2_JOINT_NAMES,
    LITE3_JOINT_LIMITS,
    Lite3RetargetConfig,
    foot_positions_from_lite3_joints,
    foot_reachability_mask,
    joint_limit_violation_mask,
    lite3_joints_from_foot_positions,
    project_joint_limits,
    retarget_joint_trajectory,
    retarget_motion,
)
from .gait_quality import GaitQuality, derive_contact, derive_world_contact, score_gait
from .sources import (
    AI4ANIMATION_FREQUENCY_HZ,
    AI4ANIMATION_PELVIS_INDEX,
    AI4ANIMATION_TOE_INDICES,
    BVH_FOOT_JOINT_NAMES,
    load_bvh_foot_kinematics,
    load_bvh_foot_positions,
    load_ai4animation_foot_positions,
    retarget_ai4animation,
)

__all__ = [
    "MotionTrajectory",
    "MotionPriorManager",
    "MotionPriorSample",
    "prior_weight",
    "DEFAULT_GO2_JOINT_NAMES",
    "LITE3_JOINT_LIMITS",
    "Lite3RetargetConfig",
    "foot_positions_from_lite3_joints",
    "foot_reachability_mask",
    "joint_limit_violation_mask",
    "lite3_joints_from_foot_positions",
    "project_joint_limits",
    "retarget_joint_trajectory",
    "retarget_motion",
    "GaitQuality",
    "derive_contact",
    "derive_world_contact",
    "score_gait",
    "AI4ANIMATION_FREQUENCY_HZ",
    "AI4ANIMATION_PELVIS_INDEX",
    "AI4ANIMATION_TOE_INDICES",
    "BVH_FOOT_JOINT_NAMES",
    "load_bvh_foot_kinematics",
    "load_bvh_foot_positions",
    "load_ai4animation_foot_positions",
    "retarget_ai4animation",
]
