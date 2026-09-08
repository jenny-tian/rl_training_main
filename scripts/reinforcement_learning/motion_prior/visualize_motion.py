#!/usr/bin/env python3
"""Render a canonical Lite3 motion-prior trajectory as a simple 3-D playback."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from rl_training.motion_prior.retarget import Lite3RetargetConfig, lite3_joints_from_foot_positions
from rl_training.motion_prior.schema import MotionTrajectory


LEG_NAMES = ("FL", "FR", "HL", "HR")
LEG_COLORS = ("#2f6fed", "#ef8a24", "#28a745", "#c43d3d")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="canonical Lite3 motion-prior NPZ")
    parser.add_argument("--output", required=True, type=Path, help=".mp4 or .gif playback output")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="source-frame stride before rendering")
    return parser


def _kinematic_points(joint_pos: np.ndarray, config: Lite3RetargetConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = joint_pos.reshape(-1, 4, 3)
    qx, q2, q3 = joints[..., 0], joints[..., 1], joints[..., 2]
    anchors = np.broadcast_to(
        np.asarray(config.hip_anchors, dtype=np.float32)[None, :, :],
        (joints.shape[0], 4, 3),
    ).copy()
    thigh = np.zeros_like(anchors)
    thigh[..., 0] = -config.thigh_length * np.sin(q2)
    thigh[..., 2] = -config.thigh_length * np.cos(q2)
    thigh[..., 1] = -np.sin(qx) * thigh[..., 2]
    knee = anchors + thigh
    feet = np.asarray(
        (
            -(config.thigh_length * np.sin(q2) + config.shank_length * np.sin(q2 + q3)),
            -np.sin(qx)
            * -(config.thigh_length * np.cos(q2) + config.shank_length * np.cos(q2 + q3)),
            np.cos(qx)
            * -(config.thigh_length * np.cos(q2) + config.shank_length * np.cos(q2 + q3)),
        )
    ).transpose(1, 2, 0)
    feet = feet + anchors
    return anchors, knee, feet


def _frame_indices(time: np.ndarray, start: float, duration: float | None, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if start < 0.0 or start >= float(time[-1]):
        raise ValueError("start-time is outside the trajectory")
    stop = float(time[-1]) if duration is None else min(float(time[-1]), start + duration)
    if stop <= start:
        raise ValueError("duration must be positive")
    first = int(np.searchsorted(time, start, side="left"))
    last = int(np.searchsorted(time, stop, side="right"))
    indices = np.arange(first, max(first + 1, last), stride, dtype=np.int64)
    if indices[-1] != max(first, last - 1):
        indices = np.r_[indices, max(first, last - 1)]
    return indices


def render_motion(
    trajectory: MotionTrajectory,
    output: str | Path,
    *,
    start_time: float = 0.0,
    duration: float | None = None,
    fps: int = 30,
    stride: int = 1,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    indices = _frame_indices(trajectory.time, start_time, duration, stride)
    config = Lite3RetargetConfig()
    anchors, knees, feet = _kinematic_points(trajectory.joint_pos[indices], config)
    contact = trajectory.contact[indices] > 0.5

    all_points = np.concatenate((anchors.reshape(-1, 3), knees.reshape(-1, 3), feet.reshape(-1, 3)))
    low = all_points.min(axis=0) - np.asarray((0.05, 0.05, 0.04), dtype=np.float32)
    high = all_points.max(axis=0) + np.asarray((0.05, 0.05, 0.04), dtype=np.float32)
    center = (low + high) * 0.5
    span = max(float(np.max(high - low)), 0.35)
    low = center - span * 0.5
    high = center + span * 0.5

    figure = plt.figure(figsize=(8, 6), dpi=120)
    axis = figure.add_subplot(111, projection="3d")
    axis.set_xlabel("x forward (m)")
    axis.set_ylabel("y left (m)")
    axis.set_zlabel("z up (m)")
    axis.set_xlim(float(low[0]), float(high[0]))
    axis.set_ylim(float(low[1]), float(high[1]))
    axis.set_zlim(float(low[2]), float(high[2]))
    axis.view_init(elev=22.0, azim=-62.0)
    axis.set_box_aspect((1.0, 1.0, 0.8))

    leg_lines = [axis.plot([], [], [], color=color, linewidth=3.0)[0] for color in LEG_COLORS]
    foot_points = [axis.plot([], [], [], marker="o", linestyle="", color=color, markersize=7)[0] for color in LEG_COLORS]
    body_line = axis.plot([], [], [], color="#333333", linewidth=3.0)[0]
    time_text = axis.text2D(0.03, 0.95, "", transform=axis.transAxes)
    contact_text = axis.text2D(0.03, 0.91, "", transform=axis.transAxes)

    def update(frame_number: int):
        hip = anchors[frame_number]
        knee = knees[frame_number]
        foot = feet[frame_number]
        for leg, line in enumerate(leg_lines):
            points = np.stack((hip[leg], knee[leg], foot[leg]))
            line.set_data(points[:, 0], points[:, 1])
            line.set_3d_properties(points[:, 2])
            foot_points[leg].set_data((foot[leg, 0],), (foot[leg, 1],))
            foot_points[leg].set_3d_properties((foot[leg, 2],))
        body = np.stack((hip[0], hip[1], hip[3], hip[2], hip[0]))
        body_line.set_data(body[:, 0], body[:, 1])
        body_line.set_3d_properties(body[:, 2])
        timestamp = float(trajectory.time[indices[frame_number]])
        pattern = " ".join(name for name, on in zip(LEG_NAMES, contact[frame_number]) if on) or "none"
        time_text.set_text(f"t = {timestamp:.2f} s")
        contact_text.set_text(f"contact: {pattern}")
        return [*leg_lines, *foot_points, body_line, time_text, contact_text]

    animation = FuncAnimation(figure, update, frames=len(indices), interval=1000.0 / fps, blit=False)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps))
    elif output_path.suffix.lower() == ".mp4":
        animation.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=1800))
    else:
        raise ValueError("output suffix must be .mp4 or .gif")
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    trajectory = MotionTrajectory.load(args.input)
    render_motion(
        trajectory,
        args.output,
        start_time=args.start_time,
        duration=args.duration,
        fps=args.fps,
        stride=args.stride,
    )
    print(f"wrote {args.output}: frames={len(_frame_indices(trajectory.time, args.start_time, args.duration, args.stride))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
