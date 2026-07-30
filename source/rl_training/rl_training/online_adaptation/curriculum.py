from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .terrain_protocol import FlatOnlineProtocol


@dataclass(frozen=True)
class EpisodeOutcome:
    passed: bool
    fell: bool
    steps: int
    cost: float
    flat_retention_passed: bool = True

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or type(self.fell) is not bool:
            raise ValueError("passed and fell must be boolean")
        if type(self.flat_retention_passed) is not bool:
            raise ValueError("flat_retention_passed must be boolean")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive int")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)):
            raise ValueError("cost must be finite and nonnegative")
        if not math.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("cost must be finite and nonnegative")


class ResidualGainScheduler:
    TIERS = (0.0, 0.25, 0.5, 0.75, 1.0)
    SAFE_EPISODES_TO_ADVANCE = 5

    def __init__(self, initial_gain: float = 0.0) -> None:
        if isinstance(initial_gain, bool) or initial_gain not in self.TIERS:
            raise ValueError(f"initial_gain must be one of {self.TIERS}")
        self._tier_index = self.TIERS.index(float(initial_gain))
        self._safe_episode_count = 0

    @property
    def gain(self) -> float:
        return self.TIERS[self._tier_index]

    def open_for_adaptation(self) -> None:
        if self._tier_index == 0:
            self._tier_index = 1
        self._safe_episode_count = 0

    def record_episode(
        self, safe: bool, rolled_back: bool, performance_degraded: bool
    ) -> None:
        if any(type(value) is not bool for value in (safe, rolled_back, performance_degraded)):
            raise ValueError("episode gain outcomes must be boolean")
        if rolled_back:
            self._tier_index = max(0, self._tier_index - 1)
            self._safe_episode_count = 0
            return
        if not safe or performance_degraded:
            self._safe_episode_count = 0
            return
        if self._tier_index == 0:
            return
        self._safe_episode_count += 1
        if self._safe_episode_count >= self.SAFE_EPISODES_TO_ADVANCE:
            self._tier_index = min(len(self.TIERS) - 1, self._tier_index + 1)
            self._safe_episode_count = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "tier_index": self._tier_index,
            "safe_episode_count": self._safe_episode_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "tier_index", "safe_episode_count"
        }:
            raise ValueError("gain scheduler state has an unsupported schema")
        tier_index = state["tier_index"]
        safe_count = state["safe_episode_count"]
        if type(tier_index) is not int or not 0 <= tier_index < len(self.TIERS):
            raise ValueError("gain scheduler tier_index is invalid")
        if (
            type(safe_count) is not int
            or not 0 <= safe_count < self.SAFE_EPISODES_TO_ADVANCE
        ):
            raise ValueError("gain scheduler safe_episode_count is invalid")
        self._tier_index = tier_index
        self._safe_episode_count = safe_count


class CurriculumCoordinator:
    WINDOW_SIZE = 10
    REQUIRED_PASSES = 8
    STOP_REASONS = {
        "episode_budget_exhausted",
        "step_budget_exhausted",
        "no_improvement",
        "stage_budget_exhausted",
        "curriculum_complete",
    }

    def __init__(self, protocol: FlatOnlineProtocol) -> None:
        if not isinstance(protocol, FlatOnlineProtocol):
            raise TypeError("protocol must be a FlatOnlineProtocol")
        self.protocol = protocol
        self._stage_index = 0
        self._stage_episodes = 0
        self._stage_steps = 0
        self._total_episodes = 0
        self._total_steps = 0
        self._outcomes: deque[EpisodeOutcome] = deque(maxlen=self.WINDOW_SIZE)
        self._best_cost: float | None = None
        self._no_improvement_count = 0
        self._seed_index = 0
        self._stop_reason: str | None = None

    @property
    def stage(self):
        return self.protocol.stages[self._stage_index]

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def next_development_seed(self) -> int:
        seed = self.protocol.development_seeds[
            self._seed_index % len(self.protocol.development_seeds)
        ]
        self._seed_index += 1
        return seed

    def record_episode(self, outcome: EpisodeOutcome) -> None:
        if not isinstance(outcome, EpisodeOutcome):
            raise TypeError("outcome must be an EpisodeOutcome")
        if self._stop_reason is not None:
            raise RuntimeError(f"curriculum already stopped: {self._stop_reason}")
        self._total_episodes += 1
        self._total_steps += outcome.steps
        self._stage_episodes += 1
        self._stage_steps += outcome.steps
        self._outcomes.append(outcome)
        if self._best_cost is None or outcome.cost < self._best_cost:
            self._best_cost = float(outcome.cost)
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
        self._update_stop_reason()
        if self._stop_reason is not None:
            return
        if self._window_passes():
            if self._stage_index == len(self.protocol.stages) - 1:
                self._stop_reason = "curriculum_complete"
            else:
                self._stage_index += 1
                self._stage_episodes = 0
                self._stage_steps = 0
                self._outcomes.clear()
                self._best_cost = None
                self._no_improvement_count = 0
        elif self._stage_episodes >= self.stage.max_episodes:
            self._stop_reason = "stage_budget_exhausted"

    def load_counts(self, *, total_episodes: int, total_steps: int) -> None:
        if type(total_episodes) is not int or total_episodes < 0:
            raise ValueError("total_episodes must be a nonnegative int")
        if type(total_steps) is not int or total_steps < 0:
            raise ValueError("total_steps must be a nonnegative int")
        self._total_episodes = total_episodes
        self._total_steps = total_steps
        self._update_stop_reason()

    def _window_passes(self) -> bool:
        return (
            len(self._outcomes) == self.WINDOW_SIZE
            and sum(outcome.passed for outcome in self._outcomes) >= self.REQUIRED_PASSES
            and not any(outcome.fell for outcome in self._outcomes)
            and all(outcome.flat_retention_passed for outcome in self._outcomes)
        )

    def _update_stop_reason(self) -> None:
        if self._total_episodes >= self.protocol.max_total_episodes:
            self._stop_reason = "episode_budget_exhausted"
        elif self._total_steps >= self.protocol.max_total_steps:
            self._stop_reason = "step_budget_exhausted"
        elif self._no_improvement_count >= self.WINDOW_SIZE:
            self._stop_reason = "no_improvement"

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol_sha256": self.protocol.sha256(),
            "stage_index": self._stage_index,
            "stage_episodes": self._stage_episodes,
            "stage_steps": self._stage_steps,
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "outcomes": tuple(self._outcomes),
            "best_cost": self._best_cost,
            "no_improvement_count": self._no_improvement_count,
            "seed_index": self._seed_index,
            "stop_reason": self._stop_reason,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_keys = set(self.state_dict())
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise ValueError("curriculum state has an unsupported schema")
        if state["protocol_sha256"] != self.protocol.sha256():
            raise ValueError("curriculum protocol differs from checkpoint")
        integer_fields = (
            "stage_index", "stage_episodes", "stage_steps", "total_episodes",
            "total_steps", "no_improvement_count", "seed_index",
        )
        if any(type(state[name]) is not int or state[name] < 0 for name in integer_fields):
            raise ValueError("curriculum counters are invalid")
        if state["stage_index"] >= len(self.protocol.stages):
            raise ValueError("curriculum stage_index is invalid")
        outcomes = state["outcomes"]
        if (
            not isinstance(outcomes, tuple)
            or len(outcomes) > self.WINDOW_SIZE
            or any(not isinstance(value, EpisodeOutcome) for value in outcomes)
        ):
            raise ValueError("curriculum outcome window is invalid")
        best_cost = state["best_cost"]
        if best_cost is not None and (
            not isinstance(best_cost, (int, float))
            or not math.isfinite(best_cost)
            or best_cost < 0.0
        ):
            raise ValueError("curriculum best_cost is invalid")
        stop_reason = state["stop_reason"]
        if stop_reason is not None and stop_reason not in self.STOP_REASONS:
            raise ValueError("curriculum stop_reason is invalid")
        self._stage_index = state["stage_index"]
        self._stage_episodes = state["stage_episodes"]
        self._stage_steps = state["stage_steps"]
        self._total_episodes = state["total_episodes"]
        self._total_steps = state["total_steps"]
        self._outcomes = deque(outcomes, maxlen=self.WINDOW_SIZE)
        self._best_cost = None if best_cost is None else float(best_cost)
        self._no_improvement_count = state["no_improvement_count"]
        self._seed_index = state["seed_index"]
        self._stop_reason = stop_reason
