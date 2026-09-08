"""Reset-local forward-progress novelty detection."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ProgressNoveltyDecision:
    max_x: float
    steps_without_progress: int
    stalled: bool


class ForwardProgressMonitor:
    def __init__(self, *, min_delta_m: float = 0.05, stall_steps: int = 100) -> None:
        if isinstance(min_delta_m, bool) or not isfinite(min_delta_m) or min_delta_m <= 0.0:
            raise ValueError("min_delta_m must be finite and positive")
        if type(stall_steps) is not int or stall_steps <= 0:
            raise ValueError("stall_steps must be a positive integer")
        self.min_delta_m = float(min_delta_m)
        self.stall_steps = stall_steps
        self.reset()

    def reset(self) -> None:
        self._max_x: float | None = None
        self._progress_reference_x: float | None = None
        self._steps_without_progress = 0

    def observe(
        self,
        *,
        base_x: float,
        commanded_forward: bool,
        paused: bool = False,
    ) -> ProgressNoveltyDecision:
        if isinstance(base_x, bool) or not isfinite(base_x):
            raise ValueError("base_x must be finite")
        observed_x = float(base_x)
        if self._max_x is None:
            self._max_x = observed_x
            self._progress_reference_x = observed_x
        else:
            self._max_x = max(self._max_x, observed_x)

        if commanded_forward and not paused:
            reference = self._progress_reference_x
            if reference is None or self._max_x - reference >= self.min_delta_m:
                self._progress_reference_x = self._max_x
                self._steps_without_progress = 0
            else:
                self._steps_without_progress += 1

        return ProgressNoveltyDecision(
            max_x=self._max_x,
            steps_without_progress=self._steps_without_progress,
            stalled=(
                bool(commanded_forward)
                and not paused
                and self._steps_without_progress >= self.stall_steps
            ),
        )
