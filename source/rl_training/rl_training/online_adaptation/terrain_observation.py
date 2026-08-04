from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


HEIGHT_SCAN_FEATURE_DIM = 10
HEIGHT_SCAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeightScanConfig:
    rows: int
    cols: int
    height_scale_m: float = 0.10
    clip_m: float = 0.20
    min_valid_fraction: float = 0.90

    def __post_init__(self) -> None:
        if type(self.rows) is not int or self.rows < 2:
            raise ValueError("height scan rows must be an integer >= 2")
        if type(self.cols) is not int or self.cols < 2:
            raise ValueError("height scan cols must be an integer >= 2")
        for name in ("height_scale_m", "clip_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and positive")
            if not isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        value = self.min_valid_fraction
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError("min_valid_fraction must be finite and in (0, 1]")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denominator


def _masked_min(values: torch.Tensor, mask: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, float("inf")))
    result = masked.amin(dim=dim)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def _masked_max(values: torch.Tensor, mask: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, float("-inf")))
    result = masked.amax(dim=dim)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def _masked_std(values: torch.Tensor, mask: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    mean = _masked_mean(values, mask, dim)
    expanded_mean = mean
    for axis in sorted(dim):
        expanded_mean = expanded_mean.unsqueeze(axis)
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    variance = ((values - expanded_mean).square() * weights).sum(dim=dim) / denominator
    return variance.clamp_min(0.0).sqrt()


def summarize_height_scan(
    relative_heights: torch.Tensor,
    config: HeightScanConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ten normalized height features and a per-batch validity mask."""
    if not isinstance(relative_heights, torch.Tensor) or relative_heights.ndim != 3:
        raise ValueError("height scan must be a [batch, rows, cols] tensor")
    if tuple(relative_heights.shape[1:]) != (config.rows, config.cols):
        raise ValueError(
            "height scan grid shape mismatch: "
            f"expected {(config.rows, config.cols)}, got {tuple(relative_heights.shape[1:])}"
        )
    if relative_heights.is_complex():
        raise ValueError("height scan must be real-valued")

    values = relative_heights.to(dtype=torch.float32)
    finite = torch.isfinite(values)
    valid_fraction = finite.to(dtype=values.dtype).mean(dim=(1, 2)).unsqueeze(-1)
    valid = valid_fraction >= float(config.min_valid_fraction)
    normalized = torch.where(finite, values, torch.zeros_like(values))
    normalized = (normalized / float(config.height_scale_m)).clamp(
        -float(config.clip_m) / float(config.height_scale_m),
        float(config.clip_m) / float(config.height_scale_m),
    )

    forward_rows = config.rows // 2
    forward = finite[:, :forward_rows]
    backward = finite[:, forward_rows:]
    left = finite[:, :, : config.cols // 2]
    right = finite[:, :, config.cols // 2 :]
    features = torch.cat(
        (
            valid_fraction,
            _masked_min(normalized, finite, (1, 2)).unsqueeze(-1),
            _masked_max(normalized, finite, (1, 2)).unsqueeze(-1),
            _masked_mean(normalized, finite, (1, 2)).unsqueeze(-1),
            _masked_std(normalized, finite, (1, 2)).unsqueeze(-1),
            _masked_min(normalized[:, :forward_rows], forward, (1, 2)).unsqueeze(-1),
            _masked_mean(normalized[:, :forward_rows], forward, (1, 2)).unsqueeze(-1),
            _masked_max(normalized[:, :forward_rows], forward, (1, 2)).unsqueeze(-1),
            (
                _masked_mean(normalized[:, :, : config.cols // 2], left, (1, 2))
                - _masked_mean(normalized[:, :, config.cols // 2 :], right, (1, 2))
            ).unsqueeze(-1),
            (
                _masked_mean(normalized[:, :forward_rows], forward, (1, 2))
                - _masked_mean(normalized[:, forward_rows:], backward, (1, 2))
            ).unsqueeze(-1),
        ),
        dim=-1,
    )
    features = torch.where(valid, features, torch.zeros_like(features))
    return features.detach(), valid.detach()
