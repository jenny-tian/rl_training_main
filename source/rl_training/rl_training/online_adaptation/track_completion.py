"""Reset-relative completion and progress cost for formal terrain challenges."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


FULL_TRACK_COMPLETION_MODE = "full_track_v1"
PROGRESS_COST_VERSION = "remaining_plus_safety_penalty_v1"
STEP_COMPLETION_STAGES = frozenset({"step_2cm", "step_4cm", "step_6cm"})
ROUGH_COMPLETION_STAGES = frozenset({"rough_2cm", "rough_4cm"})


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


def _positive(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


@dataclass
class TrackCompletion:
    """Track a single reset-relative formal challenge.

    `max_x` is deliberately monotonic. A robot that backs up after reaching a
    target remains a completed traversal, and the worker can stop before it
    walks off the end of the generated terrain.
    """

    stage_name: str
    initial_x: float
    target_x: float
    target_distance: float
    step_edge_distance: float | None = None
    completion_mode: str = FULL_TRACK_COMPLETION_MODE
    progress_cost_version: str = PROGRESS_COST_VERSION
    max_x: float | None = None

    @classmethod
    def start(
        cls,
        *,
        stage_name: str,
        initial_x: float,
        flat_length: float,
        step_length: float,
        track_length: float,
        step_clearance_margin_m: float,
        track_end_buffer_m: float,
        completion_mode: str = FULL_TRACK_COMPLETION_MODE,
        progress_cost_version: str = PROGRESS_COST_VERSION,
    ) -> "TrackCompletion":
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a nonempty string")
        initial = _finite_real(initial_x, "initial_x")
        flat = _positive(flat_length, "flat_length")
        step = _positive(step_length, "step_length")
        track = _positive(track_length, "track_length")
        clearance = _nonnegative(step_clearance_margin_m, "step_clearance_margin_m")
        buffer = _nonnegative(track_end_buffer_m, "track_end_buffer_m")
        if not isinstance(completion_mode, str) or not completion_mode.strip():
            raise ValueError("completion_mode must be a nonempty string")
        if not isinstance(progress_cost_version, str) or not progress_cost_version.strip():
            raise ValueError("progress_cost_version must be a nonempty string")

        if stage_name in STEP_COMPLETION_STAGES:
            target_distance = flat / 2.0 + step + clearance
            step_edge_distance = flat / 2.0
        elif stage_name in ROUGH_COMPLETION_STAGES:
            target_distance = track - flat / 2.0 - buffer
            step_edge_distance = None
        else:
            raise ValueError(f"unsupported completion stage: {stage_name!r}")
        if not math.isfinite(target_distance) or target_distance <= 0.0:
            raise ValueError("completion target distance must be positive and finite")
        if flat + step >= track:
            raise ValueError("flat and step segments must leave a non-empty track")
        target = initial + target_distance
        if not math.isfinite(target):
            raise ValueError("completion target must be finite")

        # The generated track extends forward from the reset-relative origin.
        # Reject a target that lies at or beyond its physical end.
        if target_distance >= track:
            raise ValueError("completion target lies outside the track interior")
        if stage_name in STEP_COMPLETION_STAGES and flat + step + clearance >= track:
            raise ValueError("step completion target lies outside the track interior")

        return cls(
            stage_name=stage_name,
            initial_x=initial,
            target_x=target,
            target_distance=target_distance,
            step_edge_distance=step_edge_distance,
            completion_mode=completion_mode,
            progress_cost_version=progress_cost_version,
            max_x=initial,
        )

    @property
    def step_edge_x(self) -> float | None:
        if self.step_edge_distance is None:
            return None
        return self.initial_x + self.step_edge_distance

    @property
    def past_step_edge(self) -> bool:
        edge_x = self.step_edge_x
        return edge_x is not None and self.max_x is not None and self.max_x >= edge_x

    def observe(self, base_x: float) -> None:
        observed = _finite_real(base_x, "base_x")
        if self.max_x is None or observed > self.max_x:
            self.max_x = observed

    @property
    def reached(self) -> bool:
        return self.max_x is not None and self.max_x >= self.target_x

    @property
    def remaining_m(self) -> float:
        current = self.initial_x if self.max_x is None else self.max_x
        return max(0.0, self.target_x - current)

    def cost(self, *, fell: bool = False, safety_violation: bool = False) -> float:
        if type(fell) is not bool or type(safety_violation) is not bool:
            raise TypeError("fell and safety_violation must be bools")
        failure_penalty = self.target_distance if fell or safety_violation else 0.0
        result = self.remaining_m + failure_penalty
        if not math.isfinite(result):
            raise ValueError("completion cost must be finite")
        return result

    def evidence(self, *, fell: bool = False, safety_violation: bool = False) -> dict[str, object]:
        return {
            "initial_x": self.initial_x,
            "max_x": self.max_x if self.max_x is not None else self.initial_x,
            "completion_target_x": self.target_x,
            "completion_remaining_m": self.remaining_m,
            "completion_reached": self.reached,
            "completion_mode": self.completion_mode,
            "progress_cost_version": self.progress_cost_version,
            "cost": self.cost(fell=fell, safety_violation=safety_violation),
        }
