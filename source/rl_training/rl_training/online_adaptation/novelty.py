"""Pure proprioceptive novelty detection for gating residual actor updates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt


class NoveltyState(str, Enum):
    CALIBRATING = "calibrating"
    MONITORING = "monitoring"
    ADAPTING = "adapting"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class NoveltyConfig:
    calibration_steps: int = 256
    enter_threshold: float = 2.5
    exit_threshold: float = 1.0
    enter_count: int = 10
    enter_window: int = 20
    recovery_steps: int = 100
    cooldown_steps: int = 100
    contact_count: int = 3
    contact_window: int = 10
    velocity_scale_floor: float = 0.05
    tilt_scale_floor: float = 0.02
    reward_scale_floor: float = 0.10

    def __post_init__(self) -> None:
        positive_counts = (
            self.calibration_steps,
            self.enter_count,
            self.enter_window,
            self.recovery_steps,
            self.cooldown_steps,
            self.contact_count,
            self.contact_window,
        )
        if any(value <= 0 for value in positive_counts):
            raise ValueError("all counts and windows must be positive")
        positive_scales = (
            self.velocity_scale_floor,
            self.tilt_scale_floor,
            self.reward_scale_floor,
        )
        thresholds = (self.enter_threshold, self.exit_threshold)
        if any(not isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("novelty thresholds must be finite and non-negative")
        if any(not isfinite(value) or value <= 0.0 for value in positive_scales):
            raise ValueError("all scale floors must be positive")
        if self.enter_count > self.enter_window:
            raise ValueError("enter_count must not exceed enter_window")
        if self.contact_count > self.contact_window:
            raise ValueError("contact_count must not exceed contact_window")
        if self.exit_threshold >= self.enter_threshold:
            raise ValueError("exit_threshold must be less than enter_threshold")


@dataclass(frozen=True)
class NoveltyDecision:
    state: NoveltyState
    novelty_score: float
    velocity_excess: float
    tilt_excess: float
    reward_drop: float
    entered_adapting: bool
    left_adapting: bool
    actor_update_enabled: bool


class NoveltyGate:
    """Tracks persistent departures from calibrated proprioceptive behavior."""

    _EMA_RATE = 0.01

    def __init__(self, config: NoveltyConfig = NoveltyConfig()) -> None:
        self.config = config
        self._state = NoveltyState.CALIBRATING
        self._calibration_count = 0
        self._cooldown_count = 0
        self._recovery_count = 0
        self._means = {"velocity": 0.0, "tilt": 0.0, "reward": 0.0}
        self._variances = {"velocity": 0.0, "tilt": 0.0, "reward": 0.0}
        self._continuous_crossings: deque[bool] = deque(maxlen=config.enter_window)
        self._contacts: deque[bool] = deque(maxlen=config.contact_window)
        self._last_score = 0.0
        self._last_velocity_excess = 0.0
        self._last_tilt_excess = 0.0
        self._last_reward_drop = 0.0

    def observe(
        self,
        velocity_error: float,
        tilt_rms: float,
        nonfoot_contact: bool,
        reward: float,
        *,
        paused: bool = False,
    ) -> NoveltyDecision:
        observations = (velocity_error, tilt_rms, reward)
        if any(not isfinite(value) for value in observations):
            raise ValueError("novelty observations must be finite")
        if paused:
            return self._decision(actor_update_enabled=False)

        entered_adapting = False
        left_adapting = False
        if (
            self._state is NoveltyState.CALIBRATING
            and self._calibration_count >= self.config.calibration_steps
        ):
            self._state = NoveltyState.MONITORING
        if self._state is NoveltyState.CALIBRATING:
            self._update_baseline(velocity_error, tilt_rms, reward)
            self._calibration_count += 1
        elif self._state is NoveltyState.MONITORING:
            self._set_excesses(velocity_error, tilt_rms, reward)
            self._continuous_crossings.append(self._last_score >= self.config.enter_threshold)
            self._contacts.append(bool(nonfoot_contact))
            if (
                sum(self._continuous_crossings) >= self.config.enter_count
                or sum(self._contacts) >= self.config.contact_count
            ):
                self._state = NoveltyState.ADAPTING
                entered_adapting = True
            else:
                self._update_baseline(velocity_error, tilt_rms, reward)
        elif self._state is NoveltyState.ADAPTING:
            self._set_excesses(velocity_error, tilt_rms, reward)
            self._recovery_count = (
                self._recovery_count + 1
                if self._last_score < self.config.exit_threshold
                else 0
            )
            if self._recovery_count >= self.config.recovery_steps:
                self._enter_cooldown()
                left_adapting = True
        else:
            self._cooldown_count += 1
            if self._cooldown_count >= self.config.cooldown_steps:
                self._state = NoveltyState.MONITORING
                self._reset_episode_counters()
        return self._decision(entered_adapting=entered_adapting, left_adapting=left_adapting)

    def force_cooldown(self) -> NoveltyDecision:
        """Close the actor gate immediately and discard current persistence evidence."""
        if self._state is NoveltyState.CALIBRATING:
            self._reset_episode_counters()
            return self._decision(actor_update_enabled=False)
        left_adapting = self._state is NoveltyState.ADAPTING
        self._enter_cooldown()
        return self._decision(left_adapting=left_adapting)

    def _set_excesses(self, velocity_error: float, tilt_rms: float, reward: float) -> None:
        self._last_velocity_excess = self._positive_zscore(
            velocity_error, "velocity", self.config.velocity_scale_floor
        )
        self._last_tilt_excess = self._positive_zscore(tilt_rms, "tilt", self.config.tilt_scale_floor)
        self._last_reward_drop = self._positive_zscore(
            self._means["reward"] - reward, None, self.config.reward_scale_floor
        )
        self._last_score = max(
            self._last_velocity_excess,
            self._last_tilt_excess,
            self._last_reward_drop,
        )

    def _positive_zscore(self, value: float, key: str | None, scale_floor: float) -> float:
        if key is None:
            scale = max(sqrt(self._variances["reward"]), scale_floor)
            return max(0.0, value / scale)
        scale = max(sqrt(self._variances[key]), scale_floor)
        return max(0.0, (value - self._means[key]) / scale)

    def _update_baseline(self, velocity_error: float, tilt_rms: float, reward: float) -> None:
        for key, value in (("velocity", velocity_error), ("tilt", tilt_rms), ("reward", reward)):
            if self._calibration_count == 0:
                self._means[key] = value
                self._variances[key] = 0.0
                continue
            delta = value - self._means[key]
            self._means[key] += self._EMA_RATE * delta
            self._variances[key] = (1.0 - self._EMA_RATE) * (
                self._variances[key] + self._EMA_RATE * delta * delta
            )

    def _enter_cooldown(self) -> None:
        self._state = NoveltyState.COOLDOWN
        self._cooldown_count = 0
        self._reset_episode_counters()

    def _reset_episode_counters(self) -> None:
        self._continuous_crossings.clear()
        self._contacts.clear()
        self._recovery_count = 0

    def _decision(
        self,
        *,
        entered_adapting: bool = False,
        left_adapting: bool = False,
        actor_update_enabled: bool | None = None,
    ) -> NoveltyDecision:
        return NoveltyDecision(
            state=self._state,
            novelty_score=self._last_score,
            velocity_excess=self._last_velocity_excess,
            tilt_excess=self._last_tilt_excess,
            reward_drop=self._last_reward_drop,
            entered_adapting=entered_adapting,
            left_adapting=left_adapting,
            actor_update_enabled=(
                self._state is NoveltyState.ADAPTING
                if actor_update_enabled is None
                else actor_update_enabled
            ),
        )
