"""Measure foot lift from the most recent contact, without a terrain oracle."""

from __future__ import annotations

import torch


class SwingLiftTracker:
    """A foot is valid only after observing its own support contact.

    This measures lift from the last support height, not clearance above the
    terrain ahead. World-height offsets and rocks underneath stance feet cannot
    masquerade as swing lift. Call reset for environments that physically reset.
    """

    def __init__(self) -> None:
        self._support_height: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None

    def observe(self, height: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
        if height.ndim != 2 or height.shape != contact.shape:
            raise ValueError("foot height/contact must have matching [batch, feet] shapes")
        if self._support_height is None:
            self._support_height = torch.zeros_like(height)
            self._valid = torch.zeros_like(height, dtype=torch.bool)
        if self._support_height.shape != height.shape:
            raise ValueError("foot batch changed without creating a new tracker")
        finite = torch.isfinite(height) & torch.isfinite(contact)
        stance = (contact > 0.5) & finite
        self._valid = (self._valid | stance) & finite
        self._support_height = torch.where(stance, height.detach(), self._support_height)
        return torch.where(
            self._valid & ~stance,
            (height - self._support_height).clamp_min(0.0),
            torch.zeros_like(height),
        )

    def reset(self, done: torch.Tensor) -> None:
        if self._valid is not None:
            self._valid &= ~done.bool().reshape(-1, 1)


def rough_swing_reward_active(*, stage: str, learning_enabled: bool, adapting: bool) -> bool:
    # Use the same online novelty decision that enables adaptation. No exact
    # terrain boundary or held-out geometry is used to enable this reward.
    return stage in {"rough_2cm", "rough_4cm"} and learning_enabled and adapting
