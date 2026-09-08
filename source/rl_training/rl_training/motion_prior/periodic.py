"""Build a seamless, state-complete periodic reference from a retargeted clip."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schema import MotionTrajectory


def build_periodic_reference(
    source: MotionTrajectory,
    *,
    frame_start: int,
    unique_frame_count: int,
    forward_speed: float,
) -> MotionTrajectory:
    source.validate()
    if frame_start < 0 or unique_frame_count < 3:
        raise ValueError("frame_start must be non-negative and unique_frame_count must be at least 3")
    stop = frame_start + unique_frame_count
    if stop > len(source.time):
        raise ValueError("periodic frame range exceeds the source trajectory")
    if not np.isfinite(forward_speed) or forward_speed <= 0.0:
        raise ValueError("forward_speed must be finite and positive")

    dt = float(np.median(np.diff(source.time)))

    def close_cycle(value: np.ndarray) -> np.ndarray:
        unique = np.asarray(value[frame_start:stop], dtype=np.float32)
        return np.concatenate((unique, unique[:1]), axis=0)

    joint_pos = close_cycle(source.joint_pos)
    foot_pos_b = close_cycle(source.foot_pos_b)
    contact = close_cycle(source.contact)
    unique_joint_pos = joint_pos[:-1]
    unique_joint_vel = (np.roll(unique_joint_pos, -1, axis=0) - np.roll(unique_joint_pos, 1, axis=0)) / (2.0 * dt)
    joint_vel = np.concatenate((unique_joint_vel, unique_joint_vel[:1]), axis=0).astype(np.float32)
    frame_count = unique_frame_count + 1
    time = np.arange(frame_count, dtype=np.float32) * np.float32(dt)
    base_lin_vel_b = np.zeros((frame_count, 3), dtype=np.float32)
    base_lin_vel_b[:, 0] = np.float32(forward_speed)
    base_ang_vel_b = np.zeros((frame_count, 3), dtype=np.float32)
    projected_gravity_b = np.zeros((frame_count, 3), dtype=np.float32)
    projected_gravity_b[:, 2] = -1.0

    metadata = dict(source.metadata)
    metadata.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "periodic_source_id": source.metadata["source_id"],
            "periodic_frame_start": frame_start,
            "periodic_unique_frame_count": unique_frame_count,
            "periodic_cycle_time_s": float(unique_frame_count * dt),
            "reference_forward_speed_mps": float(forward_speed),
            "reference_schema": "lite3-flat-imitation-v2",
            "source_id": f"{source.metadata['source_id']}-cycle-{frame_start}-{stop}",
        }
    )
    command = np.zeros((frame_count, 3), dtype=np.float32)
    command[:, 0] = np.float32(forward_speed)
    return MotionTrajectory(
        time=time,
        joint_pos=joint_pos,
        foot_pos_b=foot_pos_b,
        contact=contact,
        command=command,
        joint_vel=joint_vel,
        base_lin_vel_b=base_lin_vel_b,
        base_ang_vel_b=base_ang_vel_b,
        projected_gravity_b=projected_gravity_b,
        metadata=metadata,
    ).validate()


def build_periodic_reference_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    frame_start: int,
    unique_frame_count: int,
    forward_speed: float,
) -> MotionTrajectory:
    trajectory = build_periodic_reference(
        MotionTrajectory.load(input_path),
        frame_start=frame_start,
        unique_frame_count=unique_frame_count,
        forward_speed=forward_speed,
    )
    trajectory.save(output_path)
    return trajectory
