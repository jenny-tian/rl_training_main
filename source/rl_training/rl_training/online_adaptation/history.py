from __future__ import annotations

import torch


class HistoryBuffer:
    def __init__(self, num_envs: int, history_length: int, frame_dim: int, device: str | torch.device):
        self._data = torch.zeros(num_envs, history_length, frame_dim, device=device)

    def append(self, frame: torch.Tensor) -> None:
        if frame.shape != self._data[:, -1].shape:
            raise ValueError(f"expected frame shape {self._data[:, -1].shape}, got {frame.shape}")
        self._data = torch.roll(self._data, shifts=-1, dims=1)
        self._data[:, -1].copy_(frame)

    def preview(self, frame: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        candidate = self._data.clone()
        if reset_mask is not None:
            candidate[reset_mask] = 0.0
        candidate = torch.roll(candidate, shifts=-1, dims=1)
        candidate[:, -1].copy_(frame)
        return candidate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._data.zero_()
        else:
            self._data[env_ids] = 0.0

    def snapshot(self) -> torch.Tensor:
        return self._data.clone()
