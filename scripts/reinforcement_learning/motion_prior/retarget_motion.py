#!/usr/bin/env python3
"""Retarget an external quadruped NPZ into the canonical Lite3 motion-prior NPZ."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl_training.motion_prior.retarget import (
    DEFAULT_GO2_JOINT_NAMES,
    retarget_motion,
)
from rl_training.motion_prior.sources import (
    BVH_FOOT_JOINT_NAMES,
    load_ai4animation_foot_positions,
    load_bvh_foot_kinematics,
)
from rl_training.motion_prior.gait_quality import derive_world_contact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="source NPZ, AI4Animation text, or BVH/zip archive"
    )
    parser.add_argument("--output", required=True, type=Path, help="canonical Lite3 reference NPZ")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-format", required=True)
    parser.add_argument("--flat-only-split", required=True, choices=("train", "validation", "test", "heldout"))
    parser.add_argument("--source-joint-names", type=Path, help="JSON file containing the source joint-name list")
    parser.add_argument(
        "--input-format",
        choices=("npz", "ai4animation_joint_pos", "ai4animation_bvh"),
        default="npz",
        help="use ai4animation_joint_pos for a public *_joint_pos.txt skeleton dump",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--bvh-member", type=str, help="BVH member path when --input is a zip archive")
    parser.add_argument(
        "--bvh-position-scale", type=float, default=0.007, help="BVH source-unit to Lite3 metre scale"
    )
    parser.add_argument("--bvh-forward-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument(
        "--joint-limit-mode",
        choices=("report", "clip"),
        default="report",
        help="report raw limit violations or explicitly project joints before saving",
    )
    parser.add_argument(
        "--bvh-foot-joints",
        nargs=4,
        metavar=("FL", "FR", "HL", "HR"),
        default=None,
        help="four BVH joints whose End Sites are used as feet",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.input_format == "ai4animation_joint_pos":
        time, foot_pos_b = load_ai4animation_foot_positions(
            args.input, frame_start=args.frame_start, frame_end=args.frame_end
        )
        joint_pos = None
        contact = None
        command = None
    elif args.input_format == "ai4animation_bvh":
        time, foot_pos_b, foot_pos_w, _ = load_bvh_foot_kinematics(
            args.input,
            member=args.bvh_member,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            position_scale=args.bvh_position_scale,
            forward_sign=args.bvh_forward_sign,
            foot_joint_names=(
                tuple(args.bvh_foot_joints)
                if args.bvh_foot_joints is not None
                else BVH_FOOT_JOINT_NAMES
            ),
        )
        joint_pos = None
        contact = derive_world_contact(time, foot_pos_w).astype(np.float32)
        command = None
    else:
        with np.load(args.input, allow_pickle=False) as data:
            if "time" not in data:
                raise ValueError("input NPZ must contain time")
            time = np.array(data["time"], copy=True)
            joint_pos = np.array(data["joint_pos"], copy=True) if "joint_pos" in data else None
            foot_pos_b = np.array(data["foot_pos_b"], copy=True) if "foot_pos_b" in data else None
            contact = np.array(data["contact"], copy=True) if "contact" in data else None
            command = np.array(data["command"], copy=True) if "command" in data else None

    joint_names = DEFAULT_GO2_JOINT_NAMES
    if args.source_joint_names is not None:
        names = json.loads(args.source_joint_names.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("--source-joint-names must point to a JSON string list")
        joint_names = tuple(names)
    metadata = {
        "source_id": args.source_id,
        "source_format": args.source_format,
        "retargeter": "lite3-analytic-ik-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "flat_only_split": args.flat_only_split,
    }
    if args.input_format == "ai4animation_bvh":
        metadata.update(
            {
                "input_format": args.input_format,
                "bvh_member": args.bvh_member,
                "bvh_position_scale": args.bvh_position_scale,
                "bvh_forward_sign": args.bvh_forward_sign,
                "contact_source": "bvh-world-foot-stationarity-v1",
                "bvh_foot_joint_names": (
                    list(args.bvh_foot_joints)
                    if args.bvh_foot_joints
                    else list(BVH_FOOT_JOINT_NAMES)
                ),
            }
        )
    result = retarget_motion(
        time=time,
        joint_pos=joint_pos,
        foot_pos_b=foot_pos_b,
        contact=contact,
        command=command,
        source_joint_names=joint_names,
        metadata=metadata,
        joint_limit_mode=args.joint_limit_mode,
    )
    result.save(args.output)
    duration = float(result.time[-1] - result.time[0])
    print(
        f"wrote {args.output}: frames={result.time.shape[0]} duration_s={duration:.3f} "
        f"contact_ratio={float(result.contact.mean()):.3f} source_id={args.source_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
