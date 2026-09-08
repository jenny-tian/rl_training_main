#!/usr/bin/env python3
"""Create a seamless, state-complete Lite3 flat imitation reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl_training.motion_prior.periodic import build_periodic_reference_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-start", required=True, type=int)
    parser.add_argument("--unique-frame-count", required=True, type=int)
    parser.add_argument("--forward-speed", required=True, type=float)
    args = parser.parse_args()
    trajectory = build_periodic_reference_file(
        args.input,
        args.output,
        frame_start=args.frame_start,
        unique_frame_count=args.unique_frame_count,
        forward_speed=args.forward_speed,
    )
    print(f"saved {args.output}")
    print(f"frames={len(trajectory.time)} cycle_time={trajectory.time[-1]:.6f}s")
    print(f"foot_vertical_range_m={np.ptp(trajectory.foot_pos_b[:, :, 2], axis=0).tolist()}")


if __name__ == "__main__":
    main()
