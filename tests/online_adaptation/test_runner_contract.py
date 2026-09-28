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


def test_managed_reset_preserves_the_learning_random_stream():
    class Environment:
        def seed(self, seed):
            torch.manual_seed(seed)

        def reset(self):
            return torch.randint(0, 2, (32,)), {}

    torch.manual_seed(91)
    state = torch.get_rng_state().clone()
    first, _ = seeded_initial_reset(Environment(), 7, preserve_learning_rng=True)
    assert torch.equal(torch.get_rng_state(), state)
    second, _ = seeded_initial_reset(Environment(), 7, preserve_learning_rng=True)
    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), state)


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


def test_initial_state_config_preserves_reset_asset_and_bounds_only_pose():
    import math
    from rl_training.online_adaptation.runner_contract import configure_initial_state

    params = {"asset_cfg": object(), "pose_range": {"x": (-1, 1)}}
    asset = params["asset_cfg"]
    env = SimpleNamespace(events=SimpleNamespace(randomize_reset_base=SimpleNamespace(params=params)))
    contract = configure_initial_state(env, "perturbed-v1")
    assert params["asset_cfg"] is asset
    assert params["pose_range"]["x"] == params["pose_range"]["y"] == (-.02, .02)
    assert params["pose_range"]["yaw"] == (-math.radians(2), math.radians(2))
    assert all(v == (0., 0.) for v in params["velocity_range"].values())
    assert contract["mode"] == "perturbed-v1"
    configure_initial_state(env, "fixed")
    assert all(v == (0., 0.) for v in params["pose_range"].values())
