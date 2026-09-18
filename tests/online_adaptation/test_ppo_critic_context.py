import copy

import pytest
import torch
from torch import nn

from rl_training.online_adaptation.online_session import OnlinePPOSession, SessionCounters
from rl_training.online_adaptation.ppo_critic_context import (
    build_critic_observation,
    migrate_critic_context,
)
from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig


POLICY_OBS_DIM = 45


def critic(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1),
    )


def session_state(input_dim=POLICY_OBS_DIM, *, initialize_adam=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        updater = OnlinePPOUpdater(
            nn.Linear(2, 12), critic(input_dim), PPOConfig(target_kl=None), policy_version=5,
        )
        updater.critic_optimizer = torch.optim.Adam(updater.critic.parameters(), lr=3e-4, amsgrad=True)
        observations = torch.randn(8, input_dim)

        def value_step():
            updater.critic_optimizer.zero_grad()
            (updater.critic(observations) - 2.0).square().mean().backward()
            updater.critic_optimizer.step()

        if initialize_adam:
            value_step()
        session = OnlinePPOSession(updater, rollout_capacity=16)
        if initialize_adam:
            value_step()
        updater.policy_version = 6
        session.counters = SessionCounters(
            transitions_collected=37, update_attempts=4,
            accepted_updates=3, rejected_updates=1, discarded_transitions=5,
        )
        result = session.state_dict()
        result["has_unverified_candidate"] = True
        result["audit_metadata"] = {"label": "preserve", "indices": [2, 7]}
        return result


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_tree_equal(value, reference)
    else:
        assert actual == expected


def test_critic_observation_preserves_policy_input_and_appends_velocity_and_time():
    policy_obs = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    velocity = torch.tensor([[0.3, -0.2], [-0.1, 0.4]], requires_grad=True)
    combined = build_critic_observation(policy_obs, velocity, step=25, max_steps=100)

    assert combined.shape == (2, 7)
    assert combined.dtype == policy_obs.dtype
    assert combined.device == policy_obs.device
    torch.testing.assert_close(combined[:, :4], policy_obs, rtol=0, atol=0)
    torch.testing.assert_close(combined[:, 4:6], velocity.double(), rtol=0, atol=0)
    torch.testing.assert_close(combined[:, 6], torch.full((2,), 0.25, dtype=torch.float64))
    combined[:, 4:6].sum().backward()
    torch.testing.assert_close(velocity.grad, torch.ones_like(velocity))
    for step, fraction in ((0, 0.0), (100, 1.0)):
        result = build_critic_observation(policy_obs, velocity, step=step, max_steps=100)
        assert (result[:, -1] == fraction).all()


@pytest.mark.parametrize("step,max_steps", [
    (-1, 100), (101, 100), (0, 0), (0, -1), (float("nan"), 100),
    (float("inf"), 100), (0, float("inf")), (True, 100), (0, True),
])
def test_critic_observation_rejects_invalid_elapsed_fraction(step, max_steps):
    with pytest.raises(ValueError, match="step"):
        build_critic_observation(torch.zeros(2, 45), torch.zeros(2, 2), step=step, max_steps=max_steps)


@pytest.mark.parametrize("policy_obs,velocity", [
    (torch.zeros(45), torch.zeros(1, 2)),
    (torch.zeros(2, 45), torch.zeros(1, 2)),
    (torch.zeros(2, 45), torch.zeros(2, 3)),
    (torch.full((2, 45), float("nan")), torch.zeros(2, 2)),
    (torch.zeros(2, 45), torch.full((2, 2), float("inf"))),
    (torch.zeros(2, 45, dtype=torch.int64), torch.zeros(2, 2)),
    (torch.zeros(2, 45), torch.zeros(2, 2, dtype=torch.complex64)),
    (torch.zeros(2, 45, dtype=torch.float16), torch.full((2, 2), 1e10)),
])
def test_critic_observation_rejects_invalid_context(policy_obs, velocity):
    with pytest.raises(ValueError):
        build_critic_observation(policy_obs, velocity, step=0, max_steps=10)


def test_migration_preserves_active_stable_values_adam_and_all_other_session_state():
    original = session_state()
    before = copy.deepcopy(original)
    migrated = migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)
    observations = torch.linspace(-1, 1, 7 * POLICY_OBS_DIM).reshape(7, POLICY_OBS_DIM)
    context = torch.tensor([[3.0, -2.0, 0.7]]).expand(7, -1)

    for label in ("updater", "stable_state"):
        old = before[label]
        new = migrated[label]
        old_critic, new_critic = critic(POLICY_OBS_DIM), critic(POLICY_OBS_DIM + 3)
        old_critic.load_state_dict(old["critic"])
        new_critic.load_state_dict(new["critic"])
        with torch.no_grad():
            torch.testing.assert_close(
                new_critic(torch.cat((observations, context), dim=-1)), old_critic(observations),
                rtol=1e-6, atol=1e-7,
            )
            torch.testing.assert_close(
                new_critic(torch.cat((observations, -100 * context), dim=-1)), old_critic(observations),
                rtol=1e-6, atol=1e-7,
            )
        first_weight_id = old["critic_optimizer"]["param_groups"][0]["params"][0]
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            old_moment = old["critic_optimizer"]["state"][first_weight_id][name]
            new_moment = new["critic_optimizer"]["state"][first_weight_id][name]
            assert new_moment.shape == (128, POLICY_OBS_DIM + 3)
            torch.testing.assert_close(new_moment[:, :POLICY_OBS_DIM], old_moment, rtol=0, atol=0)
            assert torch.count_nonzero(new_moment[:, POLICY_OBS_DIM:]) == 0
        assert torch.count_nonzero(new["critic"]["0.weight"][:, POLICY_OBS_DIM:]) == 0

    # Strip only the intentionally widened columns, then compare every other
    # tensor/field, including distinct stable optimizer moments and Adam step.
    narrowed = copy.deepcopy(migrated)
    for label in ("updater", "stable_state"):
        state = narrowed[label]
        state["critic"]["0.weight"] = state["critic"]["0.weight"][:, :POLICY_OBS_DIM]
        weight_id = state["critic_optimizer"]["param_groups"][0]["params"][0]
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            state["critic_optimizer"]["state"][weight_id][name] = (
                state["critic_optimizer"]["state"][weight_id][name][:, :POLICY_OBS_DIM]
            )
    assert_tree_equal(narrowed, before)
    assert_tree_equal(original, before)

    # A real session must be able to restore both active and stable snapshots.
    restored = OnlinePPOSession(
        OnlinePPOUpdater(nn.Linear(2, 12), critic(POLICY_OBS_DIM + 3), PPOConfig(target_kl=None)),
        rollout_capacity=16,
    )
    restored.load_state_dict(migrated)
    assert restored.updater.policy_version == 6
    assert restored.counters == before["counters"]
    assert restored.state_dict()["has_unverified_candidate"]
    migrated["updater"]["critic"]["0.bias"].zero_()
    migrated["audit_metadata"]["indices"].append(99)
    assert_tree_equal(original, before)


def test_migrated_context_weights_receive_gradients_and_adam_continues_learning():
    original = session_state()
    migrated = migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)
    state = migrated["updater"]
    model = critic(POLICY_OBS_DIM + 3)
    model.load_state_dict(state["critic"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(state["critic_optimizer"])
    observations = torch.zeros(6, POLICY_OBS_DIM)
    velocity = torch.tensor([[0.1, 0.2], [0.3, -0.2], [-0.4, 0.7], [0.5, 0.8], [-0.6, 0.1], [0.7, 0.2]])
    contextual = build_critic_observation(observations, velocity, step=5, max_steps=10)
    before_step = optimizer.state[model[0].weight]["step"].item()
    baseline = model(contextual).detach()

    optimizer.zero_grad()
    (model(contextual) - baseline - 1.0).square().mean().backward()
    assert (model[0].weight.grad[:, POLICY_OBS_DIM:].abs().sum(dim=0) > 0).all()
    optimizer.step()

    assert (model[0].weight[:, POLICY_OBS_DIM:].abs().sum(dim=0) > 0).all()
    assert optimizer.state[model[0].weight]["step"].item() == before_step + 1
    assert not torch.equal(model(contextual).detach(), baseline)


@pytest.mark.parametrize("initialize_adam", [False, True])
def test_migration_is_idempotent_and_returns_an_independent_new_width_copy(initialize_adam):
    original = session_state(initialize_adam=initialize_adam)
    first = migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)
    second = migrate_critic_context(first, policy_obs_dim=POLICY_OBS_DIM)
    assert_tree_equal(second, first)
    assert second is not first
    assert second["updater"]["critic"]["0.weight"].data_ptr() != first["updater"]["critic"]["0.weight"].data_ptr()
    if not initialize_adam:
        assert second["updater"]["critic_optimizer"]["state"] == {}
        assert second["stable_state"]["critic_optimizer"]["state"] == {}


def test_migration_uses_optimizer_parameter_id_instead_of_assuming_zero():
    original = session_state()
    for label in ("updater", "stable_state"):
        optimizer = original[label]["critic_optimizer"]
        optimizer["state"] = {key + 70: value for key, value in optimizer["state"].items()}
        for group in optimizer["param_groups"]:
            group["params"] = [key + 70 for key in group["params"]]
    migrated = migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)
    for label in ("updater", "stable_state"):
        optimizer = migrated[label]["critic_optimizer"]
        assert optimizer["param_groups"][0]["params"][0] == 70
        assert optimizer["state"][70]["exp_avg"].shape == (128, POLICY_OBS_DIM + 3)
        assert_tree_equal(optimizer["state"][71], original[label]["critic_optimizer"]["state"][71])


@pytest.mark.parametrize("pending", [{}, {"observations": torch.empty(0, 45)}, {"observations": torch.zeros(1, 45)}])
def test_migration_requires_explicit_none_pending_boundary(pending):
    original = session_state(initialize_adam=False)
    original["pending_rollout"] = pending
    with pytest.raises(ValueError, match="pending_rollout=None"):
        migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)


@pytest.mark.parametrize("label", ["updater", "stable_state"])
@pytest.mark.parametrize("shape", [(128, 46), (127, 45), (128, 49)])
def test_migration_rejects_incompatible_critic_dimensions_without_changing_input(label, shape):
    original = session_state()
    original[label]["critic"]["0.weight"] = torch.zeros(shape)
    before = copy.deepcopy(original)
    with pytest.raises(ValueError, match="0.weight"):
        migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)
    assert_tree_equal(original, before)


@pytest.mark.parametrize("new_width", [False, True])
def test_migration_rejects_adam_dimensions_that_do_not_match_the_critic(new_width):
    original = session_state(POLICY_OBS_DIM + 3 if new_width else POLICY_OBS_DIM)
    original["updater"]["critic_optimizer"]["state"][0]["exp_avg"] = torch.zeros(128, 47)
    before = copy.deepcopy(original)
    with pytest.raises(ValueError, match="exp_avg.*shape"):
        migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)
    assert_tree_equal(original, before)


def test_migration_rejects_ambiguous_optimizer_parameter_mapping():
    original = session_state()
    original["updater"]["critic_optimizer"]["param_groups"][0]["params"].pop()
    with pytest.raises(ValueError, match="parameter mapping"):
        migrate_critic_context(original, policy_obs_dim=POLICY_OBS_DIM)
