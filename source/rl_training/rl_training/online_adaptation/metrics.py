from __future__ import annotations

from collections import deque
from numbers import Integral, Real
from typing import Any


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
