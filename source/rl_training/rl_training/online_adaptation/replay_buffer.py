from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, fields
from typing import Any

import torch

from .types import TransitionBatch


@dataclass(frozen=True)
class _PreparedReplayBatch:
    rows: dict[str, torch.Tensor]
    row_count: int


@dataclass(frozen=True)
class _ReplaySnapshot:
    size: int
    cursor: int
    indices: torch.Tensor
    rows: dict[str, torch.Tensor]


@dataclass(frozen=True)
class StratifiedBatch:
    transitions: TransitionBatch
    source_is_flat: torch.Tensor
    flat_fraction: float


def adapt_replay_state_frame_dim(
    state: Mapping[str, Any], target_frame_dim: int
) -> dict[str, Any]:
    """Pad legacy replay frames when a learner adds observation features."""
    if not isinstance(state, Mapping):
        raise TypeError("replay state must be a mapping")
    if type(target_frame_dim) is not int or target_frame_dim <= 0:
        raise ValueError("target_frame_dim must be a positive int")
    source_frame_dim = state.get("frame_dim")
    if type(source_frame_dim) is not int or source_frame_dim <= 0:
        raise ValueError("replay state frame_dim must be a positive int")
    if source_frame_dim > target_frame_dim:
        raise ValueError(
            "replay state frame_dim cannot be reduced: "
            f"source={source_frame_dim}, target={target_frame_dim}"
        )
    normalized = dict(state)
    if source_frame_dim == target_frame_dim:
        return normalized
    for name in ("history", "current_frame", "next_history", "next_frame"):
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"replay state {name} must be a non-scalar tensor")
        if value.shape[-1] != source_frame_dim:
            raise ValueError(
                f"replay state {name} last dimension must be {source_frame_dim}, "
                f"got {value.shape[-1]}"
            )
        padding = torch.zeros(
            (*value.shape[:-1], target_frame_dim - source_frame_dim),
            dtype=value.dtype,
            device=value.device,
        )
        normalized[name] = torch.cat((value, padding), dim=-1)
    normalized["frame_dim"] = target_frame_dim
    return normalized


def concatenate_transition_batches(
    batches: tuple[TransitionBatch, ...],
) -> TransitionBatch:
    if not batches:
        raise ValueError("at least one transition batch is required")
    values: dict[str, torch.Tensor] = {}
    for field in fields(TransitionBatch):
        tensors = tuple(getattr(batch, field.name) for batch in batches)
        if any(not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 for tensor in tensors):
            raise ValueError(f"{field.name} must be a batched tensor")
        first = tensors[0]
        if any(
            tensor.shape[1:] != first.shape[1:]
            or tensor.dtype != first.dtype
            or tensor.device != first.device
            for tensor in tensors[1:]
        ):
            raise ValueError(f"{field.name} batch tensors are incompatible")
        values[field.name] = torch.cat(tensors, dim=0)
    batch_size = values["history"].shape[0]
    if any(value.shape[0] != batch_size for value in values.values()):
        raise ValueError("concatenated transition fields have inconsistent row counts")
    return TransitionBatch(**values)


def sample_stratified_batch(
    *,
    flat_replay: ReplayBuffer,
    online_replay: ReplayBuffer,
    batch_size: int,
    flat_fraction: float,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> StratifiedBatch:
    if not flat_replay.frozen:
        raise ValueError("flat replay must be frozen before stratified sampling")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive int")
    if isinstance(flat_fraction, bool) or not isinstance(flat_fraction, (int, float)):
        raise ValueError("flat_fraction must be a finite scalar in [0, 1]")
    flat_fraction = float(flat_fraction)
    if not 0.0 <= flat_fraction <= 1.0 or flat_fraction != flat_fraction:
        raise ValueError("flat_fraction must be a finite scalar in [0, 1]")
    flat_rows = round(batch_size * flat_fraction)
    online_rows = batch_size - flat_rows
    batches: list[TransitionBatch] = []
    if flat_rows:
        batches.append(flat_replay.sample(flat_rows, device, generator=generator))
    if online_rows:
        batches.append(online_replay.sample(online_rows, device, generator=generator))
    transitions = concatenate_transition_batches(tuple(batches))
    source_is_flat = torch.cat(
        (
            torch.ones(flat_rows, 1, dtype=torch.bool, device=device),
            torch.zeros(online_rows, 1, dtype=torch.bool, device=device),
        ),
        dim=0,
    )
    return StratifiedBatch(
        transitions=transitions,
        source_is_flat=source_is_flat,
        flat_fraction=flat_rows / batch_size,
    )


class ReplayBuffer:
    _SCHEMA_VERSION = 1
    _METADATA_KEYS = frozenset(
        {
            "schema_version",
            "capacity",
            "history_length",
            "frame_dim",
            "action_dim",
            "storage_dtype",
            "size",
            "cursor",
            "frozen",
        }
    )

    def __init__(
        self,
        capacity: int,
        history_length: int,
        frame_dim: int,
        action_dim: int,
        storage_dtype: torch.dtype = torch.float16,
    ):
        dimensions = {
            "capacity": capacity,
            "history_length": history_length,
            "frame_dim": frame_dim,
            "action_dim": action_dim,
        }
        for name, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if (
            not isinstance(storage_dtype, torch.dtype)
            or not storage_dtype.is_floating_point
            or storage_dtype.is_complex
        ):
            raise ValueError(f"storage_dtype must be a real floating torch.dtype, got {storage_dtype!r}")
        self.capacity = capacity
        self._history_length = history_length
        self._frame_dim = frame_dim
        self._action_dim = action_dim
        self._size = 0
        self._cursor = 0
        self._storage_dtype = storage_dtype
        self._frozen = False
        self._lock = threading.Lock()
        self._history = torch.zeros(capacity, history_length, frame_dim, dtype=storage_dtype, device="cpu")
        self._current_frame = torch.zeros(capacity, frame_dim, dtype=storage_dtype, device="cpu")
        self._base_action = torch.zeros(capacity, action_dim, dtype=storage_dtype, device="cpu")
        self._residual_action = torch.zeros(capacity, action_dim, dtype=storage_dtype, device="cpu")
        self._reward = torch.zeros(capacity, 1, dtype=torch.float32, device="cpu")
        self._next_history = torch.zeros_like(self._history)
        self._next_frame = torch.zeros_like(self._current_frame)
        self._next_base_action = torch.zeros_like(self._base_action)
        self._terminated = torch.zeros(capacity, 1, dtype=torch.float32, device="cpu")
        self._truncated = torch.zeros(capacity, 1, dtype=torch.float32, device="cpu")
        self._bootstrap_mask = torch.zeros(capacity, 1, dtype=torch.float32, device="cpu")
        self._discount = torch.zeros(capacity, 1, dtype=torch.float32, device="cpu")
        self._safety_mask = torch.zeros(capacity, 1, dtype=torch.float32, device="cpu")
        self._curriculum_stage_id = torch.zeros(capacity, 1, dtype=torch.int64, device="cpu")
        self._episode_id = torch.zeros(capacity, 1, dtype=torch.int64, device="cpu")

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def clear(self) -> None:
        with self._lock:
            self._require_mutable_locked()
            self._size = 0
            self._cursor = 0

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            state: dict[str, Any] = {
                "schema_version": self._SCHEMA_VERSION,
                "capacity": self.capacity,
                "history_length": self._history_length,
                "frame_dim": self._frame_dim,
                "action_dim": self._action_dim,
                "storage_dtype": str(self._storage_dtype).removeprefix("torch."),
                "size": self._size,
                "cursor": self._cursor,
                "frozen": self._frozen,
            }
            state.update(
                {
                    name: storage.detach().to(device="cpu").clone()
                    for name, storage in self._destinations().items()
                }
            )
            return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self._require_mutable_locked()
            metadata, prepared = self._validate_and_clone_state(state)
            previous = {
                name: getattr(self, f"_{name}") for name in self._destinations()
            }
            previous_size = self._size
            previous_cursor = self._cursor
            previous_frozen = self._frozen
            try:
                for name, storage in prepared.items():
                    setattr(self, f"_{name}", storage)
                self._size = metadata["size"]
                self._cursor = metadata["cursor"]
                self._frozen = metadata["frozen"]
            except BaseException:
                for name, storage in previous.items():
                    setattr(self, f"_{name}", storage)
                self._size = previous_size
                self._cursor = previous_cursor
                self._frozen = previous_frozen
                raise

    def _validate_and_clone_state(
        self,
        state: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if not isinstance(state, Mapping):
            raise TypeError("replay state must be a mapping")
        expected_keys = self._METADATA_KEYS | frozenset(self._destinations())
        if set(state) != expected_keys:
            missing = sorted(expected_keys.difference(state))
            extra = sorted(set(state).difference(expected_keys))
            raise ValueError(f"replay state keys mismatch: missing={missing}, extra={extra}")

        expected_metadata = {
            "schema_version": self._SCHEMA_VERSION,
            "capacity": self.capacity,
            "history_length": self._history_length,
            "frame_dim": self._frame_dim,
            "action_dim": self._action_dim,
            "storage_dtype": str(self._storage_dtype).removeprefix("torch."),
        }
        for name, expected in expected_metadata.items():
            value = state[name]
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"replay state {name} must equal {expected!r}")

        size = state["size"]
        cursor = state["cursor"]
        frozen = state["frozen"]
        if type(size) is not int or not 0 <= size <= self.capacity:
            raise ValueError(f"replay state size must be between 0 and {self.capacity}")
        if type(cursor) is not int or not 0 <= cursor < self.capacity:
            raise ValueError(f"replay state cursor must be between 0 and {self.capacity - 1}")
        if size < self.capacity and cursor != size:
            raise ValueError("replay state cursor must equal size before the buffer is full")
        if type(frozen) is not bool:
            raise ValueError("replay state frozen must be a bool")

        prepared: dict[str, torch.Tensor] = {}
        destinations = self._destinations()
        for name, destination in destinations.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"replay state {name} must be a tensor")
            if value.device.type != "cpu":
                raise ValueError(f"replay state {name} must be on CPU")
            if value.layout != torch.strided:
                raise ValueError(f"replay state {name} must use strided layout")
            if value.shape != destination.shape:
                raise ValueError(
                    f"replay state {name} must have shape {tuple(destination.shape)}, "
                    f"got {tuple(value.shape)}"
                )
            if value.dtype != destination.dtype:
                raise ValueError(
                    f"replay state {name} must have dtype {destination.dtype}, got {value.dtype}"
                )
            if value.requires_grad:
                raise ValueError(f"replay state {name} must be detached")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"replay state {name} must contain only finite values")
            prepared[name] = value.detach().clone()

        for name in ("terminated", "truncated", "bootstrap_mask", "safety_mask"):
            try:
                self._validate_binary_tensor(name, prepared[name])
            except ValueError as error:
                raise ValueError(f"replay state {name} must contain only binary values") from error
        discount = prepared["discount"]
        if not bool(((discount >= 0) & (discount <= 1)).all().item()):
            raise ValueError("replay state discount must be between 0 and 1")
        for name in ("curriculum_stage_id", "episode_id"):
            if bool((prepared[name] < 0).any().item()):
                raise ValueError(f"replay state {name} must be nonnegative")
        if bool(
            (prepared["terminated"].bool() & prepared["truncated"].bool())
            .any()
            .item()
        ):
            raise ValueError("replay state terminated and truncated cannot both be true")
        if bool(
            (prepared["terminated"].bool() & prepared["bootstrap_mask"].bool())
            .any()
            .item()
        ):
            raise ValueError("replay state bootstrap_mask must be zero for terminated rows")

        return {"size": size, "cursor": cursor, "frozen": frozen}, prepared

    def _require_mutable_locked(self) -> None:
        if self._frozen:
            raise RuntimeError("replay buffer is frozen")

    def add_batch(
        self,
        history: torch.Tensor,
        current_frame: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
        reward: torch.Tensor,
        next_history: torch.Tensor,
        next_frame: torch.Tensor,
        next_base_action: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        bootstrap_mask: torch.Tensor,
        discount: torch.Tensor,
        safety_mask: torch.Tensor,
        curriculum_stage_id: torch.Tensor,
        episode_id: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        with self._lock:
            self._require_mutable_locked()
        prepared = self._prepare_batch(
            history=history,
            current_frame=current_frame,
            base_action=base_action,
            residual_action=residual_action,
            reward=reward,
            next_history=next_history,
            next_frame=next_frame,
            next_base_action=next_base_action,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask,
            discount=discount,
            safety_mask=safety_mask,
            curriculum_stage_id=curriculum_stage_id,
            episode_id=episode_id,
            valid_mask=valid_mask,
        )
        with self._lock:
            self._require_mutable_locked()
            self._commit_prepared(prepared)

    @classmethod
    def add_batch_atomically(
        cls,
        buffers: tuple[ReplayBuffer, ...],
        **transition: torch.Tensor,
    ) -> None:
        if len({id(buffer) for buffer in buffers}) != len(buffers):
            raise ValueError("atomic replay buffers must be distinct")
        lock_order = sorted(buffers, key=id)
        with ExitStack() as stack:
            for buffer in lock_order:
                stack.enter_context(buffer._lock)
            for buffer in buffers:
                buffer._require_mutable_locked()
        prepared = [(buffer, buffer._prepare_batch(**transition)) for buffer in buffers]
        with ExitStack() as stack:
            for buffer in lock_order:
                stack.enter_context(buffer._lock)
            for buffer in buffers:
                buffer._require_mutable_locked()
            snapshots = [
                (buffer, buffer._snapshot_for_commit(batch)) for buffer, batch in prepared
            ]
            try:
                for buffer, batch in prepared:
                    buffer._commit_prepared(batch)
            except BaseException:
                for buffer, snapshot in reversed(snapshots):
                    buffer._restore_snapshot(snapshot)
                raise

    def _prepare_batch(
        self,
        history: torch.Tensor,
        current_frame: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
        reward: torch.Tensor,
        next_history: torch.Tensor,
        next_frame: torch.Tensor,
        next_base_action: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        bootstrap_mask: torch.Tensor,
        discount: torch.Tensor,
        safety_mask: torch.Tensor,
        curriculum_stage_id: torch.Tensor,
        episode_id: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> _PreparedReplayBatch:
        values = {
            "history": history,
            "current_frame": current_frame,
            "base_action": base_action,
            "residual_action": residual_action,
            "reward": reward,
            "next_history": next_history,
            "next_frame": next_frame,
            "next_base_action": next_base_action,
            "terminated": terminated,
            "truncated": truncated,
            "bootstrap_mask": bootstrap_mask,
            "discount": discount,
            "safety_mask": safety_mask,
            "curriculum_stage_id": curriculum_stage_id,
            "episode_id": episode_id,
        }
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                raise ValueError(f"{name} must be a batched tensor")

        batch_size = history.shape[0]
        if any(value.shape[0] != batch_size for value in values.values()):
            raise ValueError("all transition tensors must have the same batch dimension")

        expected_shapes = {
            "history": (batch_size, self._history_length, self._frame_dim),
            "current_frame": (batch_size, self._frame_dim),
            "base_action": (batch_size, self._action_dim),
            "residual_action": (batch_size, self._action_dim),
            "reward": (batch_size, 1),
            "next_history": (batch_size, self._history_length, self._frame_dim),
            "next_frame": (batch_size, self._frame_dim),
            "next_base_action": (batch_size, self._action_dim),
            "terminated": (batch_size, 1),
            "truncated": (batch_size, 1),
            "bootstrap_mask": (batch_size, 1),
            "discount": (batch_size, 1),
            "safety_mask": (batch_size, 1),
            "curriculum_stage_id": (batch_size, 1),
            "episode_id": (batch_size, 1),
        }
        for name, expected_shape in expected_shapes.items():
            if values[name].shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(values[name].shape)}")

        for name in ("terminated", "truncated", "bootstrap_mask", "safety_mask"):
            self._validate_binary_tensor(name, values[name])

        if bool((terminated.bool() & truncated.bool()).any().item()):
            raise ValueError("terminated and truncated cannot both be true")

        # A true termination has no successor value and therefore cannot bootstrap.
        if bool((terminated.bool() & bootstrap_mask.bool()).any().item()):
            raise ValueError("terminated transitions must have bootstrap_mask=0")

        if discount.dtype == torch.bool or discount.is_complex():
            raise ValueError("discount must be real, finite, and between 0 and 1")
        if not bool(torch.isfinite(discount).all().item()) or not bool(
            ((discount >= 0) & (discount <= 1)).all().item()
        ):
            raise ValueError("discount must be real, finite, and between 0 and 1")

        for name in ("curriculum_stage_id", "episode_id"):
            value = values[name]
            if (
                value.dtype == torch.bool
                or value.dtype.is_floating_point
                or value.dtype.is_complex
                or bool((value < 0).any().item())
            ):
                raise ValueError(f"{name} must be a nonnegative integer tensor")

        if valid_mask is None:
            valid_rows = torch.arange(batch_size, device="cpu")
        else:
            if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape not in {
                (batch_size,),
                (batch_size, 1),
            }:
                raise ValueError("valid_mask must contain one value per transition")
            self._validate_binary_tensor("valid_mask", valid_mask)
            valid_rows = torch.nonzero(
                valid_mask.detach().reshape(-1).to(device="cpu", dtype=torch.bool),
                as_tuple=False,
            ).squeeze(-1)

        destinations = self._destinations()
        prepared = {}
        try:
            for name, destination in destinations.items():
                source_rows = valid_rows.to(device=values[name].device)
                prepared[name] = (
                    values[name]
                    .detach()
                    .index_select(0, source_rows)
                    .to(device="cpu", dtype=destination.dtype)
                    .clone()
                )
        except (RuntimeError, TypeError) as error:
            raise ValueError("transition tensors could not be prepared for replay storage") from error
        return _PreparedReplayBatch(rows=prepared, row_count=valid_rows.numel())

    def _destinations(self) -> dict[str, torch.Tensor]:
        return {
            "history": self._history,
            "current_frame": self._current_frame,
            "base_action": self._base_action,
            "residual_action": self._residual_action,
            "reward": self._reward,
            "next_history": self._next_history,
            "next_frame": self._next_frame,
            "next_base_action": self._next_base_action,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "bootstrap_mask": self._bootstrap_mask,
            "discount": self._discount,
            "safety_mask": self._safety_mask,
            "curriculum_stage_id": self._curriculum_stage_id,
            "episode_id": self._episode_id,
        }

    def _commit_prepared(self, prepared: _PreparedReplayBatch) -> None:
        for row in range(prepared.row_count):
            for name, destination in self._destinations().items():
                destination[self._cursor].copy_(prepared.rows[name][row], non_blocking=False)
            self._cursor = (self._cursor + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def _snapshot_for_commit(self, prepared: _PreparedReplayBatch) -> _ReplaySnapshot:
        if prepared.row_count == 0:
            indices = torch.empty(0, dtype=torch.int64, device="cpu")
        else:
            indices = torch.unique(
                (self._cursor + torch.arange(prepared.row_count, device="cpu")) % self.capacity,
                sorted=True,
            )
        rows = {
            name: destination.index_select(0, indices).clone()
            for name, destination in self._destinations().items()
        }
        return _ReplaySnapshot(
            size=self._size,
            cursor=self._cursor,
            indices=indices,
            rows=rows,
        )

    def _restore_snapshot(self, snapshot: _ReplaySnapshot) -> None:
        for name, destination in self._destinations().items():
            destination.index_copy_(0, snapshot.indices, snapshot.rows[name])
        self._size = snapshot.size
        self._cursor = snapshot.cursor

    @staticmethod
    def _validate_binary_tensor(name: str, value: torch.Tensor) -> None:
        if value.is_complex() or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only binary values")
        if not bool(((value == 0) | (value == 1)).all().item()):
            raise ValueError(f"{name} must contain only binary values")

    def sample(
        self,
        batch_size: int,
        device: str | torch.device,
        generator: torch.Generator | None = None,
    ) -> TransitionBatch:
        with self._lock:
            if self._size == 0:
                raise RuntimeError("cannot sample from an empty replay buffer")
            indices = torch.randint(self._size, (batch_size,), generator=generator, device="cpu")

            def select(storage: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
                return storage[indices].to(device=device, dtype=dtype)

            return TransitionBatch(
                history=select(self._history),
                current_frame=select(self._current_frame),
                base_action=select(self._base_action),
                residual_action=select(self._residual_action),
                reward=select(self._reward),
                next_history=select(self._next_history),
                next_frame=select(self._next_frame),
                next_base_action=select(self._next_base_action),
                terminated=select(self._terminated),
                truncated=select(self._truncated),
                bootstrap_mask=select(self._bootstrap_mask),
                discount=select(self._discount),
                safety_mask=select(self._safety_mask),
                curriculum_stage_id=select(self._curriculum_stage_id, torch.int64),
                episode_id=select(self._episode_id, torch.int64),
            )
