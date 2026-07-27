from __future__ import annotations

import threading
import torch

from .types import TransitionBatch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        history_length: int,
        frame_dim: int,
        action_dim: int,
        storage_dtype: torch.dtype = torch.float16,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
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
        self._done = torch.empty(capacity, 1, dtype=torch.float32, device="cpu")

    def __len__(self) -> int:
        with self._lock:
            return self._size

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
        done: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        values = (
            history,
            current_frame,
            base_action,
            residual_action,
            reward,
            next_history,
            next_frame,
            next_base_action,
            done,
        )
        batch_size = history.shape[0]
        if any(value.shape[0] != batch_size for value in values):
            raise ValueError("all transition tensors must have the same batch dimension")
        if valid_mask is None:
            valid_rows = torch.arange(batch_size, device="cpu")
        else:
            if valid_mask.numel() != batch_size:
                raise ValueError("valid_mask must contain one value per transition")
            valid_rows = torch.nonzero(
                valid_mask.detach().reshape(-1).to(device="cpu", dtype=torch.bool),
                as_tuple=False,
            ).squeeze(-1)

        destinations = (
            self._history,
            self._current_frame,
            self._base_action,
            self._residual_action,
            self._reward,
            self._next_history,
            self._next_frame,
            self._next_base_action,
            self._done,
        )
        with self._lock:
            for row in valid_rows.tolist():
                for destination, value in zip(destinations, values, strict=True):
                    destination[self._cursor].copy_(value[row], non_blocking=False)
                self._cursor = (self._cursor + 1) % self.capacity
                self._size = min(self._size + 1, self.capacity)

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

            def select(storage: torch.Tensor) -> torch.Tensor:
                return storage[indices].to(device=device, dtype=torch.float32)

            return TransitionBatch(
                history=select(self._history),
                current_frame=select(self._current_frame),
                base_action=select(self._base_action),
                residual_action=select(self._residual_action),
                reward=select(self._reward),
                next_history=select(self._next_history),
                next_frame=select(self._next_frame),
                next_base_action=select(self._next_base_action),
                done=select(self._done),
            )
