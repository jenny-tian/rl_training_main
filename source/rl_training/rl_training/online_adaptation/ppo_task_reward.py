"""Terminal task rewards for online PPO without changing qualification rules."""

from __future__ import annotations

import math
from numbers import Real


def success_absorbing_reward(
    gamma: float,
    continuing_reward_upper_bound: float = 3.0,
    completion_bonus: float = 5.0,
) -> float:
    """Fold the goal's absorbing-state reward tail into a terminal transition.

    The caller must already have established strict success, and must assign
    this reward after the ordinary reward clamp. Failure and ordinary-step
    rewards are outside this helper's scope. The transition remains terminal:
    this is an explicit goal value, not a critic or post-reset bootstrap.

    For continuing rewards bounded above by C and positive success bonus B,
    return R = B + C / (1 - gamma). This is equivalent to receiving C + B on
    entering the goal and C on every subsequent absorbing-state step. Any
    n-step delay followed by success has return at most
    R - B * (1 - gamma**n), which is strictly smaller than immediate success.
    """
    values = []
    for name, value in (
        ("gamma", gamma),
        ("continuing_reward_upper_bound", continuing_reward_upper_bound),
        ("completion_bonus", completion_bonus),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be finite and positive")
        try:
            normalized = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be finite and positive") from error
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        values.append(normalized)
    discount, upper_bound, bonus = values
    if discount >= 1.0:
        raise ValueError("gamma must satisfy 0 < gamma < 1")
    reward = bonus + upper_bound / (1.0 - discount)
    if not math.isfinite(reward):
        raise ValueError("success absorbing reward must be finite")
    return reward


__all__ = ["success_absorbing_reward"]
