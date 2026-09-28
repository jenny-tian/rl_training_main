from types import SimpleNamespace

import torch

from rl_training.online_adaptation.runner_contract import pin_forward_observation, seeded_initial_reset


def test_seeded_reset_is_independent_of_learner_rng_consumption():
    class Environment:
        def seed(self, seed):
            torch.manual_seed(seed)

        def reset(self):
            return torch.randint(0, 2, (50,)), {}

    env = Environment()
    first, _ = seeded_initial_reset(env, 7)
    torch.randn(3000)  # different SAC/LoRA initialization workloads
    second, _ = seeded_initial_reset(env, 7)
    assert torch.equal(first, second)
    third, _ = seeded_initial_reset(env, 8)
    assert not torch.equal(first, third)


def test_pin_refreshes_stale_command_observation_only_when_needed():
    command = torch.zeros(1, 3)
    term = SimpleNamespace(vel_command_b=command, target_vel_command_b=torch.zeros_like(command),
                           is_standing_env=torch.ones(1, dtype=torch.bool))
    manager = SimpleNamespace(get_command=lambda _: command, get_term=lambda _: term)
    env = SimpleNamespace(command_manager=manager)
    refreshes = []

    def observations():
        refreshes.append(1)
        return {"policy": command.clone()}

    wrapped = SimpleNamespace(unwrapped=env, get_observations=observations)
    initial = {"policy": command.clone()}
    fresh = pin_forward_observation(wrapped, initial, 0.3)
    assert fresh["policy"][0, 0] == 0.3
    assert initial["policy"][0, 0] == 0.
    assert not term.is_standing_env.any()
    assert pin_forward_observation(wrapped, fresh, 0.3) is fresh
    assert len(refreshes) == 1
