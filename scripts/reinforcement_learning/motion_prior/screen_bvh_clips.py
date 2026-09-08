#!/usr/bin/env python3
"""Rank BVH clips for use as a flat Lite3 motion prior.

The report keeps source-scale, contact, workspace, and joint-limit diagnostics
visible.  A clip is never silently clipped or accepted because its filename
looks like a walk.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from rl_training.motion_prior.gait_quality import derive_world_contact, score_gait
from rl_training.motion_prior.retarget import (
    foot_reachability_mask,
    lite3_joints_from_foot_positions,
)
from rl_training.motion_prior.sources import BVH_FOOT_JOINT_NAMES, load_bvh_foot_kinematics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="BVH file or zip archive")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=(0.004, 0.005, 0.006, 0.007),
        help="source-unit to metre scales to evaluate",
    )
    parser.add_argument("--forward-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--bvh-foot-joints", nargs=4, default=BVH_FOOT_JOINT_NAMES)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def _members(path: Path) -> list[str | None]:
    if path.suffix.lower() != ".zip":
        return [None]
    with zipfile.ZipFile(path) as archive:
        return [name for name in archive.namelist() if name.lower().endswith(".bvh")]


def _screen(
    kinematics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    member_name: str,
    scale: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    row: dict[str, object] = {
        "member": member_name,
        "scale": float(scale),
        "forward_sign": float(args.forward_sign),
        "accepted": False,
    }
    try:
        time, source_feet, source_world_feet, source_root_pos_w = kinematics
        feet = source_feet * np.float32(scale)
        world_feet = source_world_feet * np.float32(scale)
        root_pos_w = source_root_pos_w * np.float32(scale)
        reachable = foot_reachability_mask(feet)
        joints = lite3_joints_from_foot_positions(feet, allow_unreachable=True)
        contact = derive_world_contact(time, world_feet)
        root_velocity = np.gradient(root_pos_w[:, :2], time, axis=0)
        root_speed = np.linalg.norm(root_velocity, axis=-1)
        root_path_length = float(np.linalg.norm(np.diff(root_pos_w[:, :2], axis=0), axis=-1).sum())
        root_net_displacement = float(np.linalg.norm(root_pos_w[-1, :2] - root_pos_w[0, :2]))
        mean_root_speed = float(root_speed.mean())
        quality = score_gait(
            time,
            feet,
            contact=contact,
            unreachable=~reachable,
            joint_pos=joints,
        )
        accepted = quality.accepted and mean_root_speed >= 0.08 and root_path_length >= 0.5
        screening_score = quality.score + 0.5 * min(mean_root_speed / 0.5, 1.0)
        row.update(
            {
                "frame_count": quality.frame_count,
                "duration_s": quality.duration_s,
                "contact_ratio": quality.contact_ratio,
                "all_contact_ratio": quality.all_contact_ratio,
                "no_contact_ratio": quality.no_contact_ratio,
                "two_contact_ratio": quality.two_contact_ratio,
                "diagonal_contact_ratio": quality.diagonal_contact_ratio,
                "transition_rate": quality.transition_rate,
                "foot_swing_amplitude_m": quality.foot_swing_amplitude_m,
                "unreachable_ratio": quality.unreachable_ratio,
                "joint_limit_violation_ratio": quality.joint_limit_violation_ratio,
                "joint_limit_violation_frame_ratio": quality.joint_limit_violation_frame_ratio,
                "mean_root_speed_mps": mean_root_speed,
                "root_path_length_m": root_path_length,
                "root_net_displacement_m": root_net_displacement,
                "gait_score": quality.score,
                "score": screening_score,
                "accepted": accepted,
            }
        )
    except Exception as exc:  # keep one malformed clip from hiding the rest
        row.update({"error": f"{type(exc).__name__}: {exc}", "score": float("-inf")})
    return row


def main() -> int:
    args = _parser().parse_args()
    if not args.scales or any(scale <= 0.0 for scale in args.scales):
        raise SystemExit("all --scales values must be positive")
    members = _members(args.input)
    if not members:
        raise SystemExit("input archive contains no .bvh members")
    rows = []
    for index, member in enumerate(members, start=1):
        member_name = member or args.input.name
        try:
            kinematics = load_bvh_foot_kinematics(
                args.input,
                member=member,
                position_scale=1.0,
                forward_sign=args.forward_sign,
                foot_joint_names=tuple(args.bvh_foot_joints),
            )
            for scale in args.scales:
                rows.append(_screen(kinematics, member_name, scale, args))
        except Exception as exc:
            for scale in args.scales:
                rows.append(
                    {
                        "member": member_name,
                        "scale": float(scale),
                        "forward_sign": float(args.forward_sign),
                        "accepted": False,
                        "score": float("-inf"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if index == 1 or index == len(members) or index % 10 == 0:
            print(f"screened {index}/{len(members)} clips", file=sys.stderr)

    rows.sort(key=lambda row: (bool(row.get("accepted", False)), float(row.get("score", -np.inf))), reverse=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    print("rank member scale accepted score diagonal two_contact unreachable limits")
    for rank, row in enumerate(rows[: max(0, args.top_k)], start=1):
        print(
            rank,
            row.get("member"),
            f"{float(row['scale']):.4f}",
            row.get("accepted", False),
            f"{float(row.get('score', -np.inf)):.3f}",
            f"{float(row.get('diagonal_contact_ratio', 0.0)):.3f}",
            f"{float(row.get('two_contact_ratio', 0.0)):.3f}",
            f"{float(row.get('unreachable_ratio', 1.0)):.3f}",
            f"{float(row.get('joint_limit_violation_ratio', 1.0)):.3f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
