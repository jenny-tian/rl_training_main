"""External quadruped trajectory normalization and Lite3 retargeting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES

from .schema import MotionTrajectory


DEFAULT_GO2_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "HL_hip_joint", "HL_thigh_joint", "HL_calf_joint",
    "HR_hip_joint", "HR_thigh_joint", "HR_calf_joint",
)
_LEG_ORDER = ("FL", "FR", "HL", "HR")
_ROLE_ORDER = ("HipX", "HipY", "Knee")
_ROLE_ALIASES = {
    "hipx": "HipX", "hiproll": "HipX", "hipabduction": "HipX", "hip": "HipX",
    "hipy": "HipY", "hippitch": "HipY", "thigh": "HipY",
    "knee": "Knee", "calf": "Knee",
}
_HIP_ANCHORS = np.asarray(
    ((0.20, 0.105, 0.0), (0.20, -0.105, 0.0), (-0.20, 0.105, 0.0), (-0.20, -0.105, 0.0)),
    dtype=np.float32,
)

# The policy order is FL, FR, HL, HR and each leg is HipX, HipY, Knee.
# Keep these limits local to the retargeter so diagnostics do not depend on
# the reward/configuration modules being imported first.
LITE3_JOINT_LIMITS = np.asarray(
    ((-0.523, 0.523), (-2.670, 0.314), (0.524, 2.792)) * 4,
    dtype=np.float32,
)


@dataclass(frozen=True)
class Lite3RetargetConfig:
    """Explicit conventions for normalizing another quadruped into Lite3."""

    joint_names: tuple[str, ...] = LITE3_POLICY_JOINT_NAMES
    default_joint_pos: tuple[float, ...] = (0.0, -0.65, 1.3) * 4
    joint_sign: tuple[float, ...] = (1.0, -1.0, 1.0) * 4
    joint_offset: tuple[float, ...] = (0.0,) * 12
    joint_scale: tuple[float, ...] = (1.0,) * 12
    source_neutral: tuple[float, ...] = (0.0,) * 12
    thigh_length: float = 0.20
    shank_length: float = 0.21
    hip_anchors: tuple[tuple[float, float, float], ...] = tuple(map(tuple, _HIP_ANCHORS.tolist()))

    def __post_init__(self) -> None:
        for name, value in (
            ("joint_names", self.joint_names),
            ("default_joint_pos", self.default_joint_pos),
            ("joint_sign", self.joint_sign),
            ("joint_offset", self.joint_offset),
            ("joint_scale", self.joint_scale),
            ("source_neutral", self.source_neutral),
        ):
            if len(value) != 12:
                raise ValueError(f"{name} must contain 12 values")
        if len(self.hip_anchors) != 4:
            raise ValueError("hip_anchors must contain four leg anchors")
        if self.thigh_length <= 0.0 or self.shank_length <= 0.0:
            raise ValueError("leg segment lengths must be positive")


def _semantic_key(name: str) -> tuple[str, str]:
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    leg = next((candidate for candidate in _LEG_ORDER if candidate.lower() in compact), None)
    if leg is None:
        raise ValueError(f"cannot infer quadruped leg from joint name {name!r}")
    role = next(
        (canonical for alias, canonical in sorted(_ROLE_ALIASES.items(), key=lambda item: -len(item[0])) if alias in compact),
        None,
    )
    if role is None:
        raise ValueError(f"cannot infer quadruped joint role from joint name {name!r}")
    return leg, role


def retarget_joint_trajectory(
    source_joint_pos: np.ndarray,
    source_joint_names: Sequence[str] = DEFAULT_GO2_JOINT_NAMES,
    config: Lite3RetargetConfig | None = None,
) -> np.ndarray:
    """Map source absolute joint angles into canonical Lite3 physical angles."""
    cfg = config or Lite3RetargetConfig()
    source = np.asarray(source_joint_pos, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != len(source_joint_names):
        raise ValueError("source_joint_pos must have shape [T, len(source_joint_names)]")
    if not np.isfinite(source).all():
        raise ValueError("source_joint_pos must contain only finite values")
    source_indices: dict[tuple[str, str], int] = {}
    for index, name in enumerate(source_joint_names):
        key = _semantic_key(name)
        if key in source_indices:
            raise ValueError(f"duplicate source joint semantic key {key}")
        source_indices[key] = index

    output = np.empty((source.shape[0], 12), dtype=np.float32)
    default = np.asarray(cfg.default_joint_pos, dtype=np.float32)
    sign = np.asarray(cfg.joint_sign, dtype=np.float32)
    offset = np.asarray(cfg.joint_offset, dtype=np.float32)
    scale = np.asarray(cfg.joint_scale, dtype=np.float32)
    neutral = np.asarray(cfg.source_neutral, dtype=np.float32)
    for target_index, target_name in enumerate(cfg.joint_names):
        key = _semantic_key(target_name)
        if key not in source_indices:
            raise ValueError(f"source trajectory is missing joint {key[0]} {key[1]}")
        source_value = source[:, source_indices[key]]
        output[:, target_index] = default[target_index] + offset[target_index] + sign[target_index] * scale[target_index] * (
            source_value - neutral[target_index]
        )
    return output


def _validate_joint_array(joint_pos: np.ndarray) -> np.ndarray:
    joints = np.asarray(joint_pos, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != 12:
        raise ValueError("Lite3 joint positions must have shape [T, 12]")
    if not np.isfinite(joints).all():
        raise ValueError("Lite3 joint positions must contain only finite values")
    return joints


def joint_limit_violation_mask(
    joint_pos: np.ndarray,
    joint_limits: np.ndarray = LITE3_JOINT_LIMITS,
) -> np.ndarray:
    """Return one boolean per frame when any Lite3 joint exceeds its limits."""
    joints = _validate_joint_array(joint_pos)
    limits = np.asarray(joint_limits, dtype=np.float32)
    if limits.shape != (12, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("joint_limits must have shape [12, 2] with lower <= upper")
    return np.any((joints < limits[:, 0]) | (joints > limits[:, 1]), axis=1)


def project_joint_limits(
    joint_pos: np.ndarray,
    joint_limits: np.ndarray = LITE3_JOINT_LIMITS,
    *,
    mode: str = "report",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an explicit joint-limit policy and return ``(positions, mask)``.

    ``report`` preserves the source motion and returns the offending-frame mask;
    ``clip`` projects each joint to its closed interval.  The default is
    intentionally non-mutating so a caller cannot silently change a reference.
    """
    joints = _validate_joint_array(joint_pos)
    limits = np.asarray(joint_limits, dtype=np.float32)
    if limits.shape != (12, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("joint_limits must have shape [12, 2] with lower <= upper")
    violations = joint_limit_violation_mask(joints, limits)
    if mode == "report":
        return joints.copy(), violations
    if mode == "clip":
        return np.clip(joints, limits[:, 0], limits[:, 1]).astype(np.float32), violations
    raise ValueError("mode must be 'report' or 'clip'")


def foot_positions_from_lite3_joints(
    joint_pos: np.ndarray, config: Lite3RetargetConfig | None = None
) -> np.ndarray:
    """Forward kinematics for the Lite3 three-DoF legs in body coordinates."""
    cfg = config or Lite3RetargetConfig()
    joints = _validate_joint_array(joint_pos).reshape(-1, 4, 3)
    qx, q2, q3 = joints[..., 0], joints[..., 1], joints[..., 2]
    l1, l2 = cfg.thigh_length, cfg.shank_length
    z_sagittal = -(l1 * np.cos(q2) + l2 * np.cos(q2 + q3))
    x = -(l1 * np.sin(q2) + l2 * np.sin(q2 + q3))
    y = -np.sin(qx) * z_sagittal
    z = np.cos(qx) * z_sagittal
    feet = np.stack((x, y, z), axis=-1)
    return feet + np.asarray(cfg.hip_anchors, dtype=np.float32)[None, :, :]


def lite3_joints_from_foot_positions(
    foot_pos_b: np.ndarray,
    config: Lite3RetargetConfig | None = None,
    *,
    allow_unreachable: bool = False,
) -> np.ndarray:
    """Analytically solve Lite3 leg IK; unreachable frames are rejected by default.

    ``allow_unreachable`` is for offline screening only.  It clips the cosine
    used by IK at the workspace boundary so a diagnostic joint trajectory can
    still be produced; callers must use the resulting reachability mask rather
    than treating those frames as valid training data.
    """
    cfg = config or Lite3RetargetConfig()
    feet = np.asarray(foot_pos_b, dtype=np.float32)
    if feet.ndim != 3 or feet.shape[1:] != (4, 3):
        raise ValueError("foot_pos_b must have shape [T, 4, 3]")
    if not np.isfinite(feet).all():
        raise ValueError("foot_pos_b must contain only finite values")
    rel = feet - np.asarray(cfg.hip_anchors, dtype=np.float32)[None, :, :]
    lateral_radius = np.sqrt(np.square(rel[..., 1]) + np.square(rel[..., 2]))
    qx = np.arctan2(rel[..., 1], np.maximum(-rel[..., 2], 1e-8))
    x = rel[..., 0]
    down = lateral_radius
    distance = np.sqrt(np.square(x) + np.square(down))
    l1, l2 = cfg.thigh_length, cfg.shank_length
    cosine = (np.square(distance) - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not allow_unreachable and np.any((cosine < -1.0 - 1e-5) | (cosine > 1.0 + 1e-5)):
        raise ValueError("foot position is unreachable for Lite3 leg lengths")
    q3 = np.arccos(np.clip(cosine, -1.0, 1.0))
    beta = np.arctan2(-x, down)
    delta = np.arctan2(l2 * np.sin(q3), l1 + l2 * np.cos(q3))
    q2 = beta - delta
    return np.stack((qx, q2, q3), axis=-1).reshape(-1, 12).astype(np.float32)


def foot_reachability_mask(
    foot_pos_b: np.ndarray,
    config: Lite3RetargetConfig | None = None,
    *,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Return a per-frame mask for feet inside the analytic Lite3 workspace."""
    cfg = config or Lite3RetargetConfig()
    feet = np.asarray(foot_pos_b, dtype=np.float32)
    if feet.ndim != 3 or feet.shape[1:] != (4, 3):
        raise ValueError("foot_pos_b must have shape [T, 4, 3]")
    if not np.isfinite(feet).all():
        raise ValueError("foot_pos_b must contain only finite values")
    rel = feet - np.asarray(cfg.hip_anchors, dtype=np.float32)[None, :, :]
    lateral_radius = np.sqrt(np.square(rel[..., 1]) + np.square(rel[..., 2]))
    distance = np.sqrt(np.square(rel[..., 0]) + np.square(lateral_radius))
    lower = abs(cfg.thigh_length - cfg.shank_length)
    upper = cfg.thigh_length + cfg.shank_length
    leg_reachable = (distance >= lower - tolerance) & (distance <= upper + tolerance)
    return np.all(leg_reachable, axis=1)


def _derive_contact(foot_pos_b: np.ndarray, time: np.ndarray) -> np.ndarray:
    velocity = np.gradient(foot_pos_b, time, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    return ((speed < 0.15) & (foot_pos_b[..., 2] < -0.15)).astype(np.float32)


def retarget_motion(
    *,
    time: np.ndarray,
    metadata: Mapping[str, object],
    joint_pos: np.ndarray | None = None,
    foot_pos_b: np.ndarray | None = None,
    contact: np.ndarray | None = None,
    command: np.ndarray | None = None,
    source_joint_names: Sequence[str] = DEFAULT_GO2_JOINT_NAMES,
    config: Lite3RetargetConfig | None = None,
    joint_limit_mode: str = "report",
) -> MotionTrajectory:
    """Retarget either source joints or source Lite3-frame feet into an NPZ schema."""
    cfg = config or Lite3RetargetConfig()
    if joint_pos is None and foot_pos_b is None:
        raise ValueError("retarget_motion requires joint_pos or foot_pos_b")
    time_array = np.asarray(time, dtype=np.float32)
    if foot_pos_b is not None:
        feet = np.asarray(foot_pos_b, dtype=np.float32)
        lite3_joints = lite3_joints_from_foot_positions(feet, cfg)
    else:
        lite3_joints = retarget_joint_trajectory(joint_pos, source_joint_names, cfg)  # type: ignore[arg-type]
        feet = foot_positions_from_lite3_joints(lite3_joints, cfg)
    lite3_joints, joint_violations = project_joint_limits(lite3_joints, mode=joint_limit_mode)
    if joint_limit_mode == "clip":
        feet = foot_positions_from_lite3_joints(lite3_joints, cfg)
    contacts = _derive_contact(feet, time_array) if contact is None else np.asarray(contact, dtype=np.float32)
    merged_metadata = dict(metadata)
    merged_metadata.setdefault("retargeter", "lite3-analytic-ik-v1")
    merged_metadata.setdefault("joint_limit_mode", joint_limit_mode)
    merged_metadata.setdefault("joint_limit_violation_frame_ratio", float(joint_violations.mean()))
    return MotionTrajectory(
        time=time_array,
        joint_pos=lite3_joints,
        foot_pos_b=feet,
        contact=contacts,
        command=command,
        metadata=merged_metadata,
    ).validate()
