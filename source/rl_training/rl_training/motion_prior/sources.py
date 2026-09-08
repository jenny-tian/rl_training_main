"""Load source motion formats that require skeleton-level retargeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import zipfile
from typing import Mapping

import numpy as np

from .retarget import Lite3RetargetConfig, retarget_motion
from .schema import MotionTrajectory


AI4ANIMATION_PELVIS_INDEX = 0
AI4ANIMATION_NECK_INDEX = 3
AI4ANIMATION_TOE_INDICES = (10, 19, 15, 23)  # FL, FR, HL, HR in the public loader
AI4ANIMATION_FREQUENCY_HZ = 60.0
BVH_FOOT_JOINT_NAMES = ("LeftHand", "RightHand", "LeftFoot", "RightFoot")


@dataclass
class _BVHNode:
    name: str
    offset: np.ndarray
    channels: tuple[str, ...]
    children: list["_BVHNode"] = field(default_factory=list)
    end_offset: np.ndarray | None = None
    channel_start: int = 0


def _read_bvh_text(path: str | Path, member: str | None = None) -> str:
    input_path = Path(path)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".bvh")]
            if member is None:
                if len(names) != 1:
                    raise ValueError("BVH zip input requires --bvh-member when it contains multiple clips")
                member = names[0]
            if member not in archive.namelist():
                raise ValueError(f"BVH member not found in archive: {member}")
            return archive.read(member).decode("utf-8-sig")
    if member is not None:
        raise ValueError("member is only valid when the BVH input is a zip archive")
    return input_path.read_text(encoding="utf-8-sig")


def _parse_bvh(text: str) -> tuple[_BVHNode, int, float, np.ndarray]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0].upper() != "HIERARCHY":
        raise ValueError("BVH input must start with HIERARCHY")
    cursor = 1
    channel_count = 0

    def parse_end_site() -> np.ndarray:
        nonlocal cursor
        if cursor >= len(lines) or lines[cursor].upper() != "END SITE":
            raise ValueError("malformed BVH End Site")
        cursor += 1
        if cursor >= len(lines) or lines[cursor] != "{":
            raise ValueError("malformed BVH End Site block")
        cursor += 1
        if cursor >= len(lines) or not lines[cursor].upper().startswith("OFFSET "):
            raise ValueError("BVH End Site is missing OFFSET")
        values = lines[cursor].split()[1:]
        cursor += 1
        if len(values) != 3:
            raise ValueError("BVH OFFSET must contain three values")
        if cursor >= len(lines) or lines[cursor] != "}":
            raise ValueError("malformed BVH End Site closing brace")
        cursor += 1
        return np.asarray([float(value) for value in values], dtype=np.float32)

    def parse_node() -> _BVHNode:
        nonlocal cursor, channel_count
        header = lines[cursor].split()
        if len(header) != 2 or header[0].upper() not in ("ROOT", "JOINT"):
            raise ValueError(f"expected ROOT/JOINT, got: {lines[cursor]}")
        cursor += 1
        if cursor >= len(lines) or lines[cursor] != "{":
            raise ValueError("BVH node is missing opening brace")
        cursor += 1
        if cursor >= len(lines) or not lines[cursor].upper().startswith("OFFSET "):
            raise ValueError("BVH node is missing OFFSET")
        offset_values = lines[cursor].split()[1:]
        cursor += 1
        if len(offset_values) != 3:
            raise ValueError("BVH OFFSET must contain three values")
        offset = np.asarray([float(value) for value in offset_values], dtype=np.float32)
        if cursor >= len(lines) or not lines[cursor].upper().startswith("CHANNELS "):
            raise ValueError("BVH node is missing CHANNELS")
        channel_values = lines[cursor].split()[2:]
        cursor += 1
        node = _BVHNode(
            name=header[1],
            offset=offset,
            channels=tuple(channel_values),
            channel_start=channel_count,
        )
        channel_count += len(node.channels)
        while cursor < len(lines) and lines[cursor] != "}":
            line = lines[cursor]
            if line.upper().startswith("JOINT "):
                node.children.append(parse_node())
            elif line.upper() == "END SITE":
                node.end_offset = parse_end_site()
            else:
                raise ValueError(f"unexpected BVH hierarchy line: {line}")
        if cursor >= len(lines):
            raise ValueError("unterminated BVH node")
        cursor += 1
        return node

    root = parse_node()
    if cursor >= len(lines) or lines[cursor].upper() != "MOTION":
        raise ValueError("BVH hierarchy must be followed by MOTION")
    cursor += 1
    if cursor >= len(lines):
        raise ValueError("BVH motion section is missing Frames")
    frame_match = re.match(r"Frames:\s*(\d+)", lines[cursor], re.IGNORECASE)
    if frame_match is None:
        raise ValueError("BVH motion section is missing Frames count")
    frame_count = int(frame_match.group(1))
    cursor += 1
    if cursor >= len(lines):
        raise ValueError("BVH motion section is missing Frame Time")
    time_match = re.match(r"Frame Time:\s*([0-9.eE+-]+)", lines[cursor], re.IGNORECASE)
    if time_match is None:
        raise ValueError("BVH motion section is missing Frame Time")
    frame_time = float(time_match.group(1))
    cursor += 1
    motion = np.fromstring(" ".join(lines[cursor:]), sep=" ", dtype=np.float32)
    expected = frame_count * channel_count
    if motion.size != expected:
        raise ValueError(f"BVH motion has {motion.size} values, expected {expected}")
    return root, frame_count, frame_time, motion.reshape(frame_count, channel_count)


def _axis_rotation(axis: str, angle_radians: float) -> np.ndarray:
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    if axis.upper() == "X":
        return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)), dtype=np.float32)
    if axis.upper() == "Y":
        return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)), dtype=np.float32)
    if axis.upper() == "Z":
        return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)
    raise ValueError(f"unsupported BVH rotation axis: {axis}")


def _bvh_frame_end_pose(
    root: _BVHNode,
    channels: np.ndarray,
    target_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: dict[str, np.ndarray] = {}

    def visit(node: _BVHNode, parent_position: np.ndarray | None, parent_rotation: np.ndarray | None) -> None:
        values = channels[node.channel_start : node.channel_start + len(node.channels)]
        rotation = np.eye(3, dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
        for channel, value in zip(node.channels, values):
            lower = channel.lower()
            if lower.endswith("position"):
                translation["xyz".index(lower[0])] = value
            elif lower.endswith("rotation"):
                rotation = rotation @ _axis_rotation(lower[0], np.deg2rad(float(value)))
            else:
                raise ValueError(f"unsupported BVH channel: {channel}")
        if parent_position is None:
            world_position = node.offset + translation
            world_rotation = rotation
        else:
            world_position = parent_position + parent_rotation @ node.offset + parent_rotation @ translation
            world_rotation = parent_rotation @ rotation
        positions[node.name] = world_position
        if node.end_offset is not None:
            positions[f"{node.name}__end"] = world_position + world_rotation @ node.end_offset
        for child in node.children:
            visit(child, world_position, world_rotation)

    visit(root, None, None)
    root_position = positions[root.name]
    # Recompute the root rotation from its channels for body-frame conversion.
    root_rotation = np.eye(3, dtype=np.float32)
    for channel, value in zip(root.channels, channels[: len(root.channels)]):
        if channel.lower().endswith("rotation"):
            root_rotation = root_rotation @ _axis_rotation(channel[0], np.deg2rad(float(value)))
    body_output = []
    world_output = []
    for name in target_names:
        if name not in positions:
            raise ValueError(f"BVH target joint not found: {name}")
        end_key = f"{name}__end"
        point = positions.get(end_key, positions[name])
        body_output.append(root_rotation.T @ (point - root_position))
        world_output.append(point)
    return (
        np.asarray(body_output, dtype=np.float32),
        np.asarray(world_output, dtype=np.float32),
        np.asarray(root_position, dtype=np.float32),
    )


def _map_bvh_positions(points: np.ndarray, forward_sign: float, position_scale: float) -> np.ndarray:
    mapped = np.stack((forward_sign * points[..., 0], points[..., 2], points[..., 1]), axis=-1)
    return (mapped * np.float32(position_scale)).astype(np.float32)


def load_bvh_foot_kinematics(
    path: str | Path,
    *,
    member: str | None = None,
    frame_start: int = 0,
    frame_end: int | None = None,
    position_scale: float = 0.007,
    forward_sign: float = -1.0,
    foot_joint_names: tuple[str, str, str, str] = BVH_FOOT_JOINT_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time, body-frame feet, world-frame feet, and world-frame root."""
    if len(foot_joint_names) != 4 or not all(isinstance(name, str) and name for name in foot_joint_names):
        raise ValueError("foot_joint_names must contain four non-empty joint names")
    if frame_start < 0 or (frame_end is not None and frame_end <= frame_start):
        raise ValueError("invalid BVH frame range")
    if position_scale <= 0.0 or not np.isfinite(position_scale):
        raise ValueError("position_scale must be finite and positive")
    if forward_sign not in (-1.0, 1.0):
        raise ValueError("forward_sign must be either -1.0 or 1.0")
    root, frame_count, frame_time, motion = _parse_bvh(_read_bvh_text(path, member))
    if frame_start >= frame_count:
        raise ValueError("frame_start is outside the BVH clip")
    stop = frame_count if frame_end is None else min(frame_end, frame_count)
    if stop <= frame_start:
        raise ValueError("frame range is empty")
    selected = motion[frame_start:stop]
    poses = [_bvh_frame_end_pose(root, frame, foot_joint_names) for frame in selected]
    source_body_feet = np.stack([pose[0] for pose in poses], axis=0)
    source_world_feet = np.stack([pose[1] for pose in poses], axis=0)
    source_root = np.stack([pose[2] for pose in poses], axis=0)
    time = np.arange(selected.shape[0], dtype=np.float32) * np.float32(frame_time)
    return (
        time,
        _map_bvh_positions(source_body_feet, forward_sign, position_scale),
        _map_bvh_positions(source_world_feet, forward_sign, position_scale),
        _map_bvh_positions(source_root, forward_sign, position_scale),
    )


def load_bvh_foot_positions(
    path: str | Path,
    *,
    member: str | None = None,
    frame_start: int = 0,
    frame_end: int | None = None,
    position_scale: float = 0.007,
    forward_sign: float = -1.0,
    foot_joint_names: tuple[str, str, str, str] = BVH_FOOT_JOINT_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw dog BVH motion as Lite3 body-frame feet.

    The source archive uses centimetres, BVH ``y`` is the vertical axis with
    downwards motion negative, and ``z`` is the dog's left-right axis. The
    returned coordinates are metres in Lite3 body order
    ``[x-forward, y-left, z-up]`` and leg order ``[FL, FR, HL, HR]``.
    """
    time, body_feet, _, _ = load_bvh_foot_kinematics(
        path,
        member=member,
        frame_start=frame_start,
        frame_end=frame_end,
        position_scale=position_scale,
        forward_sign=forward_sign,
        foot_joint_names=foot_joint_names,
    )
    return time, body_feet


def _rotation_matrix_xyz(angles: tuple[float, float, float]) -> np.ndarray:
    x, y, z = angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rz = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def load_ai4animation_foot_positions(
    path: str | Path,
    *,
    frame_start: int = 0,
    frame_end: int | None = None,
    position_scale: float = 0.825,
    coordinate_rotation_xyz: tuple[float, float, float] = (0.5 * np.pi, 0.0, 0.0),
    root_rotation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.47 * np.pi),
    toe_indices: tuple[int, int, int, int] = AI4ANIMATION_TOE_INDICES,
    pelvis_index: int = AI4ANIMATION_PELVIS_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Read an AI4Animation ``*_joint_pos.txt`` file as body-frame feet.

    The rotations, scale, point IDs, and 60 Hz default mirror the public
    ``nomagiclab/kine2go-pipeline`` loader.  Point IDs remain explicit so a
    different dog skeleton cannot be silently interpreted as this one.
    """
    if frame_start < 0 or (frame_end is not None and frame_end <= frame_start):
        raise ValueError("invalid AI4Animation frame range")
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[1] % 3 != 0:
        raise ValueError("AI4Animation joint_pos text must contain [T, 3*N] coordinates")
    points = raw.reshape(raw.shape[0], -1, 3)
    if max((pelvis_index, *toe_indices)) >= points.shape[1]:
        raise ValueError("AI4Animation point index exceeds the skeleton size")
    points = points[frame_start:frame_end].astype(np.float32, copy=False)
    coord_rotation = _rotation_matrix_xyz(coordinate_rotation_xyz).astype(np.float32)
    root_rotation = _rotation_matrix_xyz(root_rotation_xyz).astype(np.float32)
    points = points @ coord_rotation.T
    points = points @ root_rotation.T
    points *= float(position_scale)
    pelvis = points[:, pelvis_index : pelvis_index + 1, :]
    feet = points[:, toe_indices, :] - pelvis
    time = np.arange(feet.shape[0], dtype=np.float32) / AI4ANIMATION_FREQUENCY_HZ
    return time, feet


def retarget_ai4animation(
    path: str | Path,
    *,
    metadata: Mapping[str, object],
    frame_start: int = 0,
    frame_end: int | None = None,
    config: Lite3RetargetConfig | None = None,
) -> MotionTrajectory:
    """Convert an AI4Animation dog clip into a canonical Lite3 reference."""
    time, feet = load_ai4animation_foot_positions(path, frame_start=frame_start, frame_end=frame_end)
    return retarget_motion(time=time, foot_pos_b=feet, metadata=metadata, config=config)
