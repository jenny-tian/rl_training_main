from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class AdaptationPerformanceDecision:
    cost: float
    improved: bool
    degraded: bool
    should_rollback: bool
    rollback_reason: str | None


class AdaptationPerformance:
    """Evaluates non-overlapping observable-signal windows within one adaptation episode."""

    def __init__(
        self,
        *,
        window: int = 50,
        improvement_ratio: float = 0.05,
        degradation_ratio: float = 0.10,
        patience: int = 2,
    ) -> None:
        if window <= 0 or patience <= 0:
            raise ValueError("window and patience must be positive")
        if not isfinite(improvement_ratio) or not 0.0 <= improvement_ratio < 1.0:
            raise ValueError("improvement_ratio must be finite and in [0, 1)")
        if not isfinite(degradation_ratio) or degradation_ratio < 0.0:
            raise ValueError("degradation_ratio must be finite and non-negative")
        self.window = window
        self.improvement_ratio = improvement_ratio
        self.degradation_ratio = degradation_ratio
        self.patience = patience
        self._records: list[tuple[float, float, float]] = []
        self._best_cost: float | None = None
        self._degradation_count = 0

    @property
    def best_cost(self) -> float | None:
        return self._best_cost

    @property
    def degradation_count(self) -> int:
        return self._degradation_count

    def start_episode(self) -> None:
        self._records.clear()
        self._best_cost = None
        self._degradation_count = 0

    def add(
        self,
        *,
        velocity_error: float,
        tilt_rms: float,
        nonfoot_contact: int | float,
    ) -> AdaptationPerformanceDecision | None:
        values = (float(velocity_error), float(tilt_rms), float(nonfoot_contact))
        if any(not isfinite(value) for value in values):
            raise ValueError("adaptation performance values must be finite")
        self._records.append(values)
        if len(self._records) < self.window:
            return None

        count = len(self._records)
        velocity_mean = sum(record[0] for record in self._records) / count
        tilt_mean = sum(record[1] for record in self._records) / count
        contact_rate = sum(record[2] for record in self._records) / count
        cost = velocity_mean + tilt_mean + 0.25 * contact_rate
        self._records.clear()

        if self._best_cost is None:
            self._best_cost = cost
            return AdaptationPerformanceDecision(cost, False, False, False, None)

        improved = cost <= self._best_cost * (1.0 - self.improvement_ratio)
        degraded = cost > self._best_cost * (1.0 + self.degradation_ratio)
        if improved:
            self._best_cost = cost
            self._degradation_count = 0
        elif degraded:
            self._degradation_count += 1
        else:
            self._degradation_count = 0
        should_rollback = self._degradation_count >= self.patience
        return AdaptationPerformanceDecision(
            cost=cost,
            improved=improved,
            degraded=degraded,
            should_rollback=should_rollback,
            rollback_reason="performance_degradation" if should_rollback else None,
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"metric value is not JSON-serializable: {type(value).__name__}")


class RollingMetrics:
    def __init__(self, window: int = 100):
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window = window
        self._rolling: deque[dict[str, Any]] = deque(maxlen=window)
        self._heldout: list[dict[str, Any]] = []

    def update(
        self,
        *,
        step: int,
        in_heldout: bool,
        velocity_error: float,
        tilt_rms: float,
        nonfoot_contact: int | float,
        **extra: Any,
    ) -> dict[str, Any]:
        record = {
            "step": int(step),
            "in_heldout": bool(in_heldout),
            "velocity_error": float(velocity_error),
            "tilt_rms": float(tilt_rms),
            "nonfoot_contact": float(nonfoot_contact),
            **{key: _json_value(value) for key, value in extra.items()},
        }
        self._rolling.append(record)
        if in_heldout:
            self._heldout.append(record.copy())
        return record.copy()

    @staticmethod
    def _mean(records: list[dict[str, Any]], key: str) -> float | None:
        if not records:
            return None
        return sum(float(record[key]) for record in records) / len(records)

    @staticmethod
    def _sum(records: list[dict[str, Any]], key: str) -> float:
        return sum(float(record[key]) for record in records)

    def heldout_summary(self) -> dict[str, Any]:
        midpoint = len(self._heldout) // 2
        first_half = self._heldout[:midpoint]
        second_half = self._heldout[midpoint:]
        return {
            "heldout_samples": len(self._heldout),
            "first_half_samples": len(first_half),
            "second_half_samples": len(second_half),
            "first_half_velocity_error": self._mean(first_half, "velocity_error"),
            "second_half_velocity_error": self._mean(second_half, "velocity_error"),
            "first_half_tilt_rms": self._mean(first_half, "tilt_rms"),
            "second_half_tilt_rms": self._mean(second_half, "tilt_rms"),
            "first_half_nonfoot_contacts": self._sum(first_half, "nonfoot_contact"),
            "second_half_nonfoot_contacts": self._sum(second_half, "nonfoot_contact"),
        }

    def rolling_records(self) -> list[dict[str, Any]]:
        return [record.copy() for record in self._rolling]

    def heldout_records(self) -> list[dict[str, Any]]:
        return [record.copy() for record in self._heldout]

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "rolling": self.rolling_records(),
            "heldout": self.heldout_records(),
            "heldout_summary": self.heldout_summary(),
        }
