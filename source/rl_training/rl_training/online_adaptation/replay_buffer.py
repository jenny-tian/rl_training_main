from __future__ import annotations

import threading
from contextlib import ExitStack
from dataclasses import dataclass

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


class ReplayBuffer:
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
        self._lock = threading.Lock()
        self._history = torch.empty(capacity, history_length, frame_dim, dtype=storage_dtype, device="cpu")
        self._current_frame = torch.empty(capacity, frame_dim, dtype=storage_dtype, device="cpu")
        self._base_action = torch.empty(capacity, action_dim, dtype=storage_dtype, device="cpu")
        self._residual_action = torch.empty(capacity, action_dim, dtype=storage_dtype, device="cpu")
        self._reward = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")
        self._next_history = torch.empty_like(self._history)
        self._next_frame = torch.empty_like(self._current_frame)
        self._next_base_action = torch.empty_like(self._base_action)
        self._terminated = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")
        self._truncated = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")
        self._bootstrap_mask = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")
        self._discount = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")
        self._safety_mask = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")
        self._curriculum_stage_id = torch.empty(capacity, 1, dtype=torch.int64, device="cpu")
        self._episode_id = torch.empty(capacity, 1, dtype=torch.int64, device="cpu")

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def clear(self) -> None:
        with self._lock:
            self._size = 0
            self._cursor = 0

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
            self._commit_prepared(prepared)

    @classmethod
    def add_batch_atomically(
        cls,
        buffers: tuple[ReplayBuffer, ...],
        **transition: torch.Tensor,
    ) -> None:
        if len({id(buffer) for buffer in buffers}) != len(buffers):
            raise ValueError("atomic replay buffers must be distinct")
        prepared = [(buffer, buffer._prepare_batch(**transition)) for buffer in buffers]
        lock_order = sorted(buffers, key=id)
        with ExitStack() as stack:
            for buffer in lock_order:
                stack.enter_context(buffer._lock)
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
