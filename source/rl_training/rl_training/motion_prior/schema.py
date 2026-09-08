"""Validated, provenance-bearing trajectory format used by the flat motion prior."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_METADATA_KEYS = (
    "source_id",
    "source_format",
    "retargeter",
    "created_utc",
    "flat_only_split",
)


def _as_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} must not be an object array")
    try:
        array = array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class MotionTrajectory:
    """Canonical quadruped reference motion in Lite3's frame conventions.

    ``joint_pos`` uses ``LITE3_POLICY_JOINT_NAMES`` order.  Foot positions are
    body-frame coordinates in metres and contact is a binary float array so the
    file can be consumed by both NumPy preprocessing and Torch rewards.
    """

    time: np.ndarray
    joint_pos: np.ndarray
    foot_pos_b: np.ndarray
    contact: np.ndarray
    command: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    base_lin_vel_b: np.ndarray | None = None
    base_ang_vel_b: np.ndarray | None = None
    projected_gravity_b: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "MotionTrajectory":
        time = _as_float_array(self.time, "time")
        joint_pos = _as_float_array(self.joint_pos, "joint_pos")
        foot_pos_b = _as_float_array(self.foot_pos_b, "foot_pos_b")
        contact = _as_float_array(self.contact, "contact")

        if time.ndim != 1 or time.shape[0] < 2:
            raise ValueError("time must have shape [T] with at least two frames")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing")
        frame_count = time.shape[0]
        if joint_pos.shape != (frame_count, 12):
            raise ValueError(f"joint_pos must have shape ({frame_count}, 12)")
        if foot_pos_b.shape != (frame_count, 4, 3):
            raise ValueError(f"foot_pos_b must have shape ({frame_count}, 4, 3)")
        if contact.shape != (frame_count, 4):
            raise ValueError(f"contact must have shape ({frame_count}, 4)")
        if np.any((contact < 0.0) | (contact > 1.0)):
            raise ValueError("contact values must be in [0, 1]")
        if self.command is not None:
            command = _as_float_array(self.command, "command")
            if command.shape != (frame_count, 3):
                raise ValueError(f"command must have shape ({frame_count}, 3)")
        for name, value, width in (
            ("joint_vel", self.joint_vel, 12),
            ("base_lin_vel_b", self.base_lin_vel_b, 3),
            ("base_ang_vel_b", self.base_ang_vel_b, 3),
            ("projected_gravity_b", self.projected_gravity_b, 3),
        ):
            if value is None:
                continue
            array = _as_float_array(value, name)
            if array.shape != (frame_count, width):
                raise ValueError(f"{name} must have shape ({frame_count}, {width})")

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        missing = [key for key in REQUIRED_METADATA_KEYS if not self.metadata.get(key)]
        if missing:
            raise ValueError(f"metadata missing required keys: {', '.join(missing)}")
        try:
            json.dumps(dict(self.metadata), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        return self

    def save(self, path: str | Path) -> None:
        """Validate and save the trajectory as a compressed, non-pickle NPZ."""
        self.validate()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "time": np.asarray(self.time, dtype=np.float32),
            "joint_pos": np.asarray(self.joint_pos, dtype=np.float32),
            "foot_pos_b": np.asarray(self.foot_pos_b, dtype=np.float32),
            "contact": np.asarray(self.contact, dtype=np.float32),
            "metadata": np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        }
        if self.command is not None:
            arrays["command"] = np.asarray(self.command, dtype=np.float32)
        for name in ("joint_vel", "base_lin_vel_b", "base_ang_vel_b", "projected_gravity_b"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = np.asarray(value, dtype=np.float32)
        np.savez_compressed(output_path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MotionTrajectory":
        """Load and validate a trajectory without allowing NumPy pickle data."""
        input_path = Path(path)
        with np.load(input_path, allow_pickle=False) as data:
            required = ("time", "joint_pos", "foot_pos_b", "contact", "metadata")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"trajectory NPZ missing keys: {', '.join(missing)}")
            metadata_value = np.asarray(data["metadata"])
            if metadata_value.ndim != 0:
                raise ValueError("metadata must be a scalar JSON string")
            try:
                metadata = json.loads(str(metadata_value.item()))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("metadata must contain valid JSON") from exc
            command = np.array(data["command"], copy=True) if "command" in data else None
            trajectory = cls(
                time=np.array(data["time"], copy=True),
                joint_pos=np.array(data["joint_pos"], copy=True),
                foot_pos_b=np.array(data["foot_pos_b"], copy=True),
                contact=np.array(data["contact"], copy=True),
                command=command,
                joint_vel=np.array(data["joint_vel"], copy=True) if "joint_vel" in data else None,
                base_lin_vel_b=(
                    np.array(data["base_lin_vel_b"], copy=True) if "base_lin_vel_b" in data else None
                ),
                base_ang_vel_b=(
                    np.array(data["base_ang_vel_b"], copy=True) if "base_ang_vel_b" in data else None
                ),
                projected_gravity_b=(
                    np.array(data["projected_gravity_b"], copy=True)
                    if "projected_gravity_b" in data
                    else None
                ),
                metadata=metadata,
            )
        return trajectory.validate()
