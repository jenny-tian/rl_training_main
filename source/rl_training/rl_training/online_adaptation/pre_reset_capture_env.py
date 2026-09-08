from __future__ import annotations

from collections.abc import Callable, Sequence

from isaaclab.envs import ManagerBasedRLEnv


class PreResetCaptureManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based environment with hooks that run before automatic resets."""

    def __init__(self, *args, **kwargs):
        self._pre_reset_callbacks: list[Callable[[Sequence[int]], None]] = []
        super().__init__(*args, **kwargs)

    def add_pre_reset_callback(self, callback: Callable[[Sequence[int]], None]) -> None:
        if not callable(callback):
            raise TypeError("pre-reset callback must be callable")
        self._pre_reset_callbacks.append(callback)

    def _reset_idx(self, env_ids: Sequence[int]):
        for callback in self._pre_reset_callbacks:
            callback(env_ids)
        contact_counts = getattr(
            self, "_lite3_illegal_contact_consecutive_counts", None
        )
        if contact_counts is not None:
            contact_counts[env_ids] = 0
        return super()._reset_idx(env_ids)
