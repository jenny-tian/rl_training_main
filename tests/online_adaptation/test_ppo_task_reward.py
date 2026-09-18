import math
import sys

import pytest

from rl_training.online_adaptation.ppo_task_reward import success_absorbing_reward


def test_default_strict_success_includes_absorbing_goal_value():
    assert success_absorbing_reward(0.99) == pytest.approx(305.0)


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99, 0.999])
@pytest.mark.parametrize("delay_steps", [1, 100])
@pytest.mark.parametrize("upper_bound, bonus", [(3.0, 5.0), (1.5, 0.2)])
def test_any_bounded_continuing_rewards_cannot_make_delayed_success_better(gamma, delay_steps, upper_bound, bonus):
    goal_reward = success_absorbing_reward(gamma, upper_bound, bonus)
    # Every discount weight is positive, so setting every continuing reward
    # to its upper bound maximizes the return of ANY admissible delay sequence.
    best_delayed_return = math.fsum(gamma**step * upper_bound for step in range(delay_steps))
    best_delayed_return += gamma**delay_steps * goal_reward

    assert best_delayed_return < goal_reward
    assert goal_reward - best_delayed_return == pytest.approx(
        bonus * (1.0 - gamma**delay_steps), rel=1e-8, abs=1e-10
    )
    rewards = [upper_bound, -8.0, 0.0, upper_bound / 2.0]
    mixed_delayed_return = math.fsum(
        gamma**step * rewards[step % len(rewards)] for step in range(delay_steps)
    ) + gamma**delay_steps * goal_reward
    assert mixed_delayed_return <= best_delayed_return


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.0, 1.1, float("nan"), float("inf"), -float("inf"), True, "0.99"])
def test_invalid_discount_is_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        success_absorbing_reward(gamma)


@pytest.mark.parametrize("field", ["continuing_reward_upper_bound", "completion_bonus"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf"), True])
def test_reward_parameters_must_be_finite_and_positive(field, value):
    with pytest.raises(ValueError, match=field):
        success_absorbing_reward(0.99, **{field: value})


def test_discount_near_one_returns_a_finite_tail_when_representable():
    gamma = math.nextafter(1.0, 0.0)

    reward = success_absorbing_reward(gamma)

    assert math.isfinite(reward)
    assert reward > success_absorbing_reward(0.99)
    assert reward == pytest.approx(5.0 + 3.0 / (1.0 - gamma))


def test_finite_inputs_cannot_produce_an_infinite_goal_reward():
    with pytest.raises(ValueError, match="absorbing reward must be finite"):
        success_absorbing_reward(0.99, continuing_reward_upper_bound=sys.float_info.max)


def test_unrepresentable_numeric_parameter_is_rejected():
    with pytest.raises(ValueError, match="completion_bonus"):
        success_absorbing_reward(0.99, completion_bonus=10**1000)
