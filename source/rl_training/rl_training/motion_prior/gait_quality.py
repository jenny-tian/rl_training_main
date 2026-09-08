"""Auditable gait-quality metrics for selecting retargeted quadruped clips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retarget import LITE3_JOINT_LIMITS



@dataclass(frozen=True)
class GaitQuality:
    frame_count: int
    duration_s: float
    contact_ratio: float
    all_contact_ratio: float
    no_contact_ratio: float
    two_contact_ratio: float
    diagonal_contact_ratio: float
    transition_rate: float
    foot_swing_amplitude_m: float
    unreachable_ratio: float
    joint_limit_violation_ratio: float
    joint_limit_violation_frame_ratio: float
    score: float

    @property
    def accepted(self) -> bool:
        return (
            self.duration_s >= 5.0
            and self.two_contact_ratio >= 0.10
            and self.diagonal_contact_ratio >= 0.05
            and self.transition_rate >= 0.01
            and self.foot_swing_amplitude_m >= 0.02
            and self.unreachable_ratio <= 0.02
            and self.joint_limit_violation_ratio <= 0.02
        )


def derive_contact(
    time: np.ndarray,
    foot_pos_b: np.ndarray,
    *,
    speed_threshold: float = 0.15,
    height_threshold: float = -0.15,
) -> np.ndarray:
    """Derive a conservative contact mask using the same rule as retargeting."""
    time_array = np.asarray(time, dtype=np.float32)
    feet = np.asarray(foot_pos_b, dtype=np.float32)
    if time_array.ndim != 1 or feet.ndim != 3 or feet.shape[0] != time_array.shape[0] or feet.shape[1:] != (4, 3):
        raise ValueError("time and foot_pos_b must have shapes [T] and [T, 4, 3]")
    if time_array.shape[0] < 2 or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing with at least two frames")
    velocity = np.gradient(feet, time_array, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    return (speed < float(speed_threshold)) & (feet[..., 2] < float(height_threshold))


def derive_world_contact(
    time: np.ndarray,
    foot_pos_w: np.ndarray,
    *,
    speed_threshold: float = 0.20,
    ground_height_tolerance: float = 0.025,
) -> np.ndarray:
    """Derive flat-ground contact from world-frame foot stationarity and height."""
    time_array = np.asarray(time, dtype=np.float32)
    feet = np.asarray(foot_pos_w, dtype=np.float32)
    if time_array.ndim != 1 or feet.ndim != 3 or feet.shape[0] != time_array.shape[0] or feet.shape[1:] != (4, 3):
        raise ValueError("time and foot_pos_w must have shapes [T] and [T, 4, 3]")
    if time_array.shape[0] < 2 or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing with at least two frames")
    if speed_threshold <= 0.0 or ground_height_tolerance < 0.0:
        raise ValueError("contact thresholds must be non-negative and speed_threshold must be positive")
    velocity = np.gradient(feet, time_array, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    ground_height = float(np.percentile(feet[..., 2], 5.0))
    return (speed < float(speed_threshold)) & (feet[..., 2] <= ground_height + float(ground_height_tolerance))


def _validate_contact(contact: np.ndarray, frame_count: int) -> np.ndarray:
    mask = np.asarray(contact, dtype=bool)
    if mask.shape != (frame_count, 4):
        raise ValueError("contact must have shape [T, 4]")
    return mask


def score_gait(
    time: np.ndarray,
    foot_pos_b: np.ndarray,
    *,
    contact: np.ndarray | None = None,
    unreachable: np.ndarray | None = None,
    joint_pos: np.ndarray | None = None,
    joint_limits: np.ndarray = LITE3_JOINT_LIMITS,
) -> GaitQuality:
    """Score a clip without hiding the component metrics.

    ``diagonal_contact_ratio`` counts frames matching either FL+HR or FR+HL
    exactly. It is deliberately reported separately from ``two_contact_ratio``
    so pace/bound-like motions are not silently called trot.
    """
    time_array = np.asarray(time, dtype=np.float32)
    feet = np.asarray(foot_pos_b, dtype=np.float32)
    if time_array.ndim != 1 or feet.shape != (time_array.shape[0], 4, 3) or time_array.shape[0] < 2:
        raise ValueError("time and foot_pos_b must have shapes [T] and [T, 4, 3]")
    mask = derive_contact(time_array, feet) if contact is None else _validate_contact(contact, len(time_array))
    counts = mask.sum(axis=1)
    diagonal = ((mask[:, 0] & mask[:, 3] & ~mask[:, 1] & ~mask[:, 2]) |
                (mask[:, 1] & mask[:, 2] & ~mask[:, 0] & ~mask[:, 3]))
    if unreachable is None:
        unreachable_ratio = 0.0
    else:
        unreachable_mask = np.asarray(unreachable, dtype=bool)
        if unreachable_mask.shape != (len(time_array),):
            raise ValueError("unreachable must have shape [T]")
        unreachable_ratio = float(unreachable_mask.mean())

    violation_ratio = 0.0
    violation_frame_ratio = 0.0
    if joint_pos is not None:
        joints = np.asarray(joint_pos, dtype=np.float32)
        limits = np.asarray(joint_limits, dtype=np.float32)
        if joints.shape != (len(time_array), 12) or limits.shape != (12, 2):
            raise ValueError("joint_pos and joint_limits must have shapes [T, 12] and [12, 2]")
        violation_mask = (joints < limits[:, 0]) | (joints > limits[:, 1])
        violation_ratio = float(violation_mask.mean())
        violation_frame_ratio = float(np.any(violation_mask, axis=1).mean())

    contact_ratio = float(mask.mean())
    all_contact_ratio = float(np.all(mask, axis=1).mean())
    no_contact_ratio = float(np.all(~mask, axis=1).mean())
    two_contact_ratio = float((counts == 2).mean())
    diagonal_ratio = float(diagonal.mean())
    transition_rate = float(np.sum(mask[1:] != mask[:-1]) / max(1, (len(mask) - 1) * 4))
    swing_amplitude = float(np.ptp(feet, axis=0)[:, 2].mean())
    score = (
        2.0 * diagonal_ratio
        + 0.8 * two_contact_ratio
        + 0.4 * min(transition_rate / 0.05, 1.0)
        + 0.4 * min(swing_amplitude / 0.05, 1.0)
        - 1.5 * all_contact_ratio
        - 1.0 * no_contact_ratio
        - 3.0 * unreachable_ratio
        - 3.0 * violation_ratio
    )
    return GaitQuality(
        frame_count=len(time_array),
        duration_s=float(time_array[-1] - time_array[0]),
        contact_ratio=contact_ratio,
        all_contact_ratio=all_contact_ratio,
        no_contact_ratio=no_contact_ratio,
        two_contact_ratio=two_contact_ratio,
        diagonal_contact_ratio=diagonal_ratio,
        transition_rate=transition_rate,
        foot_swing_amplitude_m=swing_amplitude,
        unreachable_ratio=unreachable_ratio,
        joint_limit_violation_ratio=violation_ratio,
        joint_limit_violation_frame_ratio=violation_frame_ratio,
        score=float(score),
    )
