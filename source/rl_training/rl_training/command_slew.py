"""Dependency-light velocity-command slew limiting."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _validated_max_delta(
    max_delta: Sequence[float], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if len(max_delta) != 3:
        raise ValueError("max_delta must contain exactly three components")
    values = tuple(float(value) for value in max_delta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("max_delta components must be finite")
    if not all(value > 0.0 for value in values):
        raise ValueError("max_delta components must be positive")
    return torch.tensor(values, device=device, dtype=dtype)


def slew_toward(
    current: torch.Tensor,
    target: torch.Tensor,
    *,
    max_delta: Sequence[float],
) -> torch.Tensor:
    if current.shape != target.shape:
        raise ValueError("current and target commands must have the same shape")
    if current.ndim == 0 or current.shape[-1] != 3:
        raise ValueError("velocity commands must have a final dimension of three")
    limit = _validated_max_delta(max_delta, device=current.device, dtype=current.dtype)
    delta = torch.clamp(target - current, min=-limit, max=limit)
    return current + delta
