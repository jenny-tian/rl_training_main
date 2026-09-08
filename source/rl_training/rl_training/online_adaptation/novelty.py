"""Pure proprioceptive novelty detection for gating residual actor updates."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Any


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
        if not isfinite(self.enter_threshold) or self.enter_threshold <= 0.0:
            raise ValueError("enter_threshold must be finite and positive")
        if not isfinite(self.exit_threshold) or self.exit_threshold < 0.0:
            raise ValueError("exit_threshold must be finite and non-negative")
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
    progress_excess: float
    progress_stalled: bool
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
        self._last_progress_excess = 0.0
        self._last_progress_stalled = False

    def observe(
        self,
        velocity_error: float,
        tilt_rms: float,
        nonfoot_contact: bool,
        reward: float,
        *,
        paused: bool = False,
        progress_stalled: bool = False,
        update_baseline: bool = True,
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
            self._set_excesses(velocity_error, tilt_rms, reward, progress_stalled)
            self._continuous_crossings.append(self._last_score >= self.config.enter_threshold)
            self._contacts.append(bool(nonfoot_contact))
            if (
                sum(self._continuous_crossings) >= self.config.enter_count
                or sum(self._contacts) >= self.config.contact_count
            ):
                self._state = NoveltyState.ADAPTING
                entered_adapting = True
            else:
                if update_baseline:
                    self._update_baseline(velocity_error, tilt_rms, reward)
        elif self._state is NoveltyState.ADAPTING:
            self._set_excesses(velocity_error, tilt_rms, reward, progress_stalled)
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "state": self._state.value,
            "calibration_count": self._calibration_count,
            "cooldown_count": self._cooldown_count,
            "recovery_count": self._recovery_count,
            "means": dict(self._means),
            "variances": dict(self._variances),
            "continuous_crossings": tuple(self._continuous_crossings),
            "contacts": tuple(self._contacts),
            "last_score": self._last_score,
            "last_velocity_excess": self._last_velocity_excess,
            "last_tilt_excess": self._last_tilt_excess,
            "last_reward_drop": self._last_reward_drop,
            "last_progress_excess": self._last_progress_excess,
            "last_progress_stalled": self._last_progress_stalled,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected_keys = set(self.state_dict())
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("novelty state has an unsupported schema")
        if state["config"] != self.config:
            raise ValueError("novelty configuration differs from checkpoint")
        try:
            novelty_state = NoveltyState(state["state"])
        except (TypeError, ValueError) as error:
            raise ValueError("novelty state value is invalid") from error
        means = state["means"]
        variances = state["variances"]
        expected_metrics = {"velocity", "tilt", "reward"}
        if set(means) != expected_metrics or set(variances) != expected_metrics:
            raise ValueError("novelty baseline state is invalid")
        scalar_values = [
            *means.values(), *variances.values(), state["last_score"],
            state["last_velocity_excess"], state["last_tilt_excess"],
            state["last_reward_drop"],
            state["last_progress_excess"],
        ]
        if any(not isinstance(value, (int, float)) or not isfinite(value) for value in scalar_values):
            raise ValueError("novelty state contains non-finite values")
        counts = (state["calibration_count"], state["cooldown_count"], state["recovery_count"])
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("novelty counters are invalid")
        crossings = tuple(state["continuous_crossings"])
        contacts = tuple(state["contacts"])
        if len(crossings) > self.config.enter_window or len(contacts) > self.config.contact_window:
            raise ValueError("novelty persistence windows are invalid")
        self._state = novelty_state
        self._calibration_count, self._cooldown_count, self._recovery_count = counts
        self._means = dict(means)
        self._variances = dict(variances)
        self._continuous_crossings = deque(
            (bool(value) for value in crossings), maxlen=self.config.enter_window
        )
        self._contacts = deque(
            (bool(value) for value in contacts), maxlen=self.config.contact_window
        )
        self._last_score = float(state["last_score"])
        self._last_velocity_excess = float(state["last_velocity_excess"])
        self._last_tilt_excess = float(state["last_tilt_excess"])
        self._last_reward_drop = float(state["last_reward_drop"])
        self._last_progress_excess = float(state["last_progress_excess"])
        if type(state["last_progress_stalled"]) is not bool:
            raise ValueError("novelty progress state is invalid")
        self._last_progress_stalled = state["last_progress_stalled"]

    def export_baseline(self, *, source: Mapping[str, Any]) -> dict[str, Any]:
        if self._calibration_count < self.config.calibration_steps:
            raise ValueError("novelty baseline calibration is incomplete")
        normalized_source = self._validate_baseline_source(source)
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "observation_count": self._calibration_count,
            "means": dict(self._means),
            "variances": dict(self._variances),
            "source": normalized_source,
        }

    def load_baseline(self, artifact: Mapping[str, Any]) -> None:
        expected = {
            "schema_version", "config", "observation_count", "means", "variances", "source"
        }
        if not isinstance(artifact, Mapping) or set(artifact) != expected:
            raise ValueError("novelty baseline has an unsupported schema")
        if artifact["schema_version"] != 1:
            raise ValueError("novelty baseline schema version mismatch")
        if artifact["config"] != asdict(self.config):
            raise ValueError("novelty baseline configuration differs from runtime")
        count = artifact["observation_count"]
        if type(count) is not int or count < self.config.calibration_steps:
            raise ValueError("novelty baseline observation count is invalid")
        means = artifact["means"]
        variances = artifact["variances"]
        expected_metrics = {"velocity", "tilt", "reward"}
        if not isinstance(means, Mapping) or not isinstance(variances, Mapping):
            raise ValueError("novelty baseline statistics are invalid")
        if set(means) != expected_metrics or set(variances) != expected_metrics:
            raise ValueError("novelty baseline statistics are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            for value in (*means.values(), *variances.values())
        ) or any(float(value) < 0.0 for value in variances.values()):
            raise ValueError("novelty baseline statistics are invalid")
        self._validate_baseline_source(artifact["source"])
        self._state = NoveltyState.MONITORING
        self._calibration_count = count
        self._means = {key: float(value) for key, value in means.items()}
        self._variances = {key: float(value) for key, value in variances.items()}
        self._cooldown_count = 0
        self._last_score = 0.0
        self._last_velocity_excess = 0.0
        self._last_tilt_excess = 0.0
        self._last_reward_drop = 0.0
        self._last_progress_excess = 0.0
        self._last_progress_stalled = False
        self._reset_episode_counters()

    @staticmethod
    def _validate_baseline_source(source: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(source, Mapping) or set(source) != {"seed", "metrics_sha256"}:
            raise ValueError("novelty baseline source is invalid")
        seed = source["seed"]
        digest = source["metrics_sha256"]
        if type(seed) is not int or seed < 0:
            raise ValueError("novelty baseline source seed is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("novelty baseline source hash is invalid")
        return {"seed": seed, "metrics_sha256": digest}

    def _set_excesses(
        self,
        velocity_error: float,
        tilt_rms: float,
        reward: float,
        progress_stalled: bool = False,
    ) -> None:
        self._last_velocity_excess = self._positive_zscore(
            velocity_error, "velocity", self.config.velocity_scale_floor
        )
        self._last_tilt_excess = self._positive_zscore(tilt_rms, "tilt", self.config.tilt_scale_floor)
        self._last_reward_drop = self._positive_zscore(
            self._means["reward"] - reward, None, self.config.reward_scale_floor
        )
        self._last_progress_stalled = bool(progress_stalled)
        self._last_progress_excess = self.config.enter_threshold if progress_stalled else 0.0
        self._last_score = max(
            self._last_velocity_excess,
            self._last_tilt_excess,
            self._last_reward_drop,
            self._last_progress_excess,
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
            progress_excess=self._last_progress_excess,
            progress_stalled=self._last_progress_stalled,
            entered_adapting=entered_adapting,
            left_adapting=left_adapting,
            actor_update_enabled=(
                self._state is NoveltyState.ADAPTING
                if actor_update_enabled is None
                else actor_update_enabled
            ),
        )
