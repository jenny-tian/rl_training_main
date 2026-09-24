"""Validated offline teacher trajectories for latent steering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .terrain_protocol import default_flat_online_protocol


TEACHER_DATASET_SCHEMA = "rgls_teacher_dataset_v1"
FORMAL_TRAIN_SEEDS = (101, 103, 107)
FORMAL_DEVELOPMENT_SEEDS = (109,)
_TENSOR_KEYS = frozenset(
    {
        "history",
        "current_frame",
        "base_action",
        "teacher_action",
        "terrain_seed",
        "episode_index",
        "step_index",
        "safe_mask",
    }
)
_REQUIRED_METADATA = frozenset(
    {
        "terrain_split",
        "protocol_sha256",
        "base_checkpoint_sha256",
        "teacher_checkpoint_sha256",
        "sealed_seeds",
        "frame_dim",
        "action_dim",
        "history_length",
    }
)


@dataclass(frozen=True)
class TeacherTrajectoryDataset:
    history: Tensor
    current_frame: Tensor
    base_action: Tensor
    teacher_action: Tensor
    terrain_seed: Tensor
    episode_index: Tensor
    step_index: Tensor
    safe_mask: Tensor
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return int(self.history.shape[0])

    @property
    def frame_dim(self) -> int:
        return int(self.current_frame.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.base_action.shape[1])

    def residual(self) -> Tensor:
        return self.teacher_action - self.base_action


def _finite_float_tensor(value: object, name: str, *, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{name} must be a finite floating-point tensor with ndim={ndim}")
    return value


def _integer_tensor(value: object, name: str, *, size: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.shape[0] != size:
        raise ValueError(f"{name} must be a one-dimensional tensor with {size} rows")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must use an integer dtype")
    return value


def validate_teacher_dataset(
    dataset: TeacherTrajectoryDataset,
    *,
    sealed_seeds: tuple[int, ...] = (),
) -> TeacherTrajectoryDataset:
    if not isinstance(dataset, TeacherTrajectoryDataset):
        raise TypeError("dataset must be a TeacherTrajectoryDataset")
    metadata = dataset.metadata
    if not isinstance(metadata, dict) or not _REQUIRED_METADATA.issubset(metadata):
        raise ValueError("teacher dataset metadata is incomplete")
    split = metadata.get("terrain_split")
    if split not in {"train", "development"}:
        raise ValueError("teacher dataset terrain_split must be train or development")
    expected_seeds = (
        FORMAL_TRAIN_SEEDS if split == "train" else FORMAL_DEVELOPMENT_SEEDS
    )
    if any(not isinstance(metadata.get(key), str) or not metadata[key] for key in (
        "protocol_sha256", "base_checkpoint_sha256", "teacher_checkpoint_sha256",
    )):
        raise ValueError("teacher dataset provenance hashes must be nonempty strings")
    declared_sealed = metadata.get("sealed_seeds")
    if (
        not isinstance(declared_sealed, (list, tuple))
        or any(type(seed) is not int for seed in declared_sealed)
    ):
        raise ValueError("teacher dataset metadata must declare integer sealed_seeds")
    protocol = default_flat_online_protocol()
    protocol_sealed = set(protocol.sealed_seeds)
    if metadata["protocol_sha256"] != protocol.sha256():
        raise ValueError("teacher dataset protocol hash does not match the formal protocol")
    if not protocol_sealed.issubset(set(declared_sealed)):
        raise ValueError("teacher dataset sealed_seeds must include every protocol sealed seed")
    history = _finite_float_tensor(dataset.history, "history", ndim=3)
    current_frame = _finite_float_tensor(dataset.current_frame, "current_frame", ndim=2)
    base_action = _finite_float_tensor(dataset.base_action, "base_action", ndim=2)
    teacher_action = _finite_float_tensor(dataset.teacher_action, "teacher_action", ndim=2)
    count = int(history.shape[0])
    if count < 1 or current_frame.shape[0] != count or base_action.shape[0] != count or teacher_action.shape[0] != count:
        raise ValueError("teacher dataset tensors must have the same nonzero batch dimension")
    if history.shape[1] < 1 or current_frame.shape[1] != history.shape[2]:
        raise ValueError("history and current_frame dimensions do not match")
    if base_action.shape != teacher_action.shape:
        raise ValueError("base_action and teacher_action shapes must match")
    if (
        history.device != current_frame.device
        or history.device != base_action.device
        or history.device != teacher_action.device
        or history.dtype != current_frame.dtype
        or history.dtype != base_action.dtype
        or history.dtype != teacher_action.dtype
    ):
        raise ValueError("floating-point teacher tensors must share device and dtype")
    terrain_seed = _integer_tensor(dataset.terrain_seed, "terrain_seed", size=count)
    episode_index = _integer_tensor(dataset.episode_index, "episode_index", size=count)
    step_index = _integer_tensor(dataset.step_index, "step_index", size=count)
    if (
        not isinstance(dataset.safe_mask, Tensor)
        or dataset.safe_mask.ndim != 1
        or dataset.safe_mask.shape[0] != count
        or dataset.safe_mask.dtype != torch.bool
        or dataset.safe_mask.device != history.device
    ):
        raise ValueError("safe_mask must be a boolean vector matching the dataset")
    if bool((episode_index < 0).any().item()) or bool((step_index < 0).any().item()):
        raise ValueError("episode_index and step_index must be nonnegative")
    blocked_seeds = set(sealed_seeds) | protocol_sealed | set(declared_sealed)
    if any(int(seed) in blocked_seeds for seed in terrain_seed.detach().cpu().tolist()):
        raise ValueError("teacher dataset contains sealed terrain seeds")
    unexpected_split_seeds = set(terrain_seed.detach().cpu().tolist()) - set(expected_seeds)
    if unexpected_split_seeds:
        raise ValueError(
            f"teacher dataset contains seeds outside the formal {split} split: "
            f"{sorted(unexpected_split_seeds)}"
        )
    if int(metadata["frame_dim"]) != int(current_frame.shape[1]) or int(metadata["action_dim"]) != int(base_action.shape[1]):
        raise ValueError("teacher dataset metadata dimensions do not match tensors")
    if int(metadata["history_length"]) != int(history.shape[1]):
        raise ValueError("teacher dataset history length metadata does not match tensors")
    if not bool(dataset.safe_mask.any().item()):
        raise ValueError("teacher dataset must contain at least one safe row")
    return dataset


def _payload(dataset: TeacherTrajectoryDataset) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "schema": TEACHER_DATASET_SCHEMA,
        "tensors": {
            "history": dataset.history.detach().cpu(),
            "current_frame": dataset.current_frame.detach().cpu(),
            "base_action": dataset.base_action.detach().cpu(),
            "teacher_action": dataset.teacher_action.detach().cpu(),
            "terrain_seed": dataset.terrain_seed.detach().cpu(),
            "episode_index": dataset.episode_index.detach().cpu(),
            "step_index": dataset.step_index.detach().cpu(),
            "safe_mask": dataset.safe_mask.detach().cpu(),
        },
        "metadata": dict(dataset.metadata),
    }


def save_teacher_dataset(
    path: str | Path,
    dataset: TeacherTrajectoryDataset,
    *,
    sealed_seeds: tuple[int, ...] = (),
) -> None:
    validate_teacher_dataset(dataset, sealed_seeds=sealed_seeds)
    target = Path(path)
    if target.exists():
        raise FileExistsError("teacher datasets are immutable; choose a new path")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_payload(dataset), target)


def load_teacher_dataset(
    path: str | Path,
    *,
    sealed_seeds: tuple[int, ...] = (),
) -> TeacherTrajectoryDataset:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensors = payload.get("tensors") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("schema") != TEACHER_DATASET_SCHEMA
        or not isinstance(tensors, Mapping)
        or set(tensors) != _TENSOR_KEYS
    ):
        raise ValueError("teacher dataset schema is incompatible")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("teacher dataset metadata must be a dictionary")
    dataset = TeacherTrajectoryDataset(
        history=tensors["history"],
        current_frame=tensors["current_frame"],
        base_action=tensors["base_action"],
        teacher_action=tensors["teacher_action"],
        terrain_seed=tensors["terrain_seed"],
        episode_index=tensors["episode_index"],
        step_index=tensors["step_index"],
        safe_mask=tensors["safe_mask"],
        metadata=metadata,
    )
    return validate_teacher_dataset(dataset, sealed_seeds=sealed_seeds)


def select_safe_rows(dataset: TeacherTrajectoryDataset) -> TeacherTrajectoryDataset:
    """Return only safe rows while preserving provenance and tensor alignment."""
    validate_teacher_dataset(dataset)
    mask = dataset.safe_mask
    return TeacherTrajectoryDataset(
        history=dataset.history[mask],
        current_frame=dataset.current_frame[mask],
        base_action=dataset.base_action[mask],
        teacher_action=dataset.teacher_action[mask],
        terrain_seed=dataset.terrain_seed[mask],
        episode_index=dataset.episode_index[mask],
        step_index=dataset.step_index[mask],
        safe_mask=torch.ones(int(mask.sum().item()), dtype=torch.bool, device=mask.device),
        metadata={**dataset.metadata, "selected_safe_rows": int(mask.sum().item())},
    )
