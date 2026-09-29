import copy
import json
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
from rl_training.online_adaptation.ppo_rollout import PPORollout, compute_gae


class GaussianPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def get_distribution(self, obs):
        return Normal(self.linear(obs), self.log_std.exp())

    def evaluate_actions(self, obs, actions):
        dist = self.get_distribution(obs)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)

    def mean_action(self, obs):
        return self.linear(obs)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, obs):
        return self.linear(obs)


def test_online_ppo_defaults_use_conservative_learning_rates():
    config = PPOConfig()
    assert config.actor_lr == 5e-5
    assert config.critic_lr == 5e-5


def make_rollout(policy, critic, *, version=0, length=12):
    torch.manual_seed(20)
    obs = torch.randn(length, 2)
    actions = policy.get_distribution(obs).sample()
    old_log_probs = policy.get_distribution(obs).log_prob(actions).sum(-1, keepdim=True).detach()
    old_values = critic(obs).detach()
    rewards = torch.ones(length, 1)
    terminated = torch.zeros(length, 1)
    truncated = torch.zeros(length, 1)
    return PPORollout(obs, actions, old_log_probs, old_values, rewards, terminated, truncated, old_values, version)


def test_gae_bootstraps_timeout_but_cuts_recursion_at_episode_boundary():
    advantages, returns = compute_gae(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.zeros(3, 1),
        torch.tensor([[10.0], [20.0], [30.0]]),
        torch.tensor([[0.0], [0.0], [1.0]]),
        torch.tensor([[1.0], [0.0], [0.0]]),
        gamma=0.9,
        gae_lambda=1.0,
    )
    # t=0 uses timeout bootstrap (1 + .9*10), then stops before t=1.
    torch.testing.assert_close(advantages, torch.tensor([[10.0], [22.7], [3.0]]))
    torch.testing.assert_close(returns, advantages)


def test_updater_requires_matching_policy_version_and_advances_after_acceptance():
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, minibatch_size=4), policy_version=2)
    stale = make_rollout(policy, critic, version=1)
    with pytest.raises(ValueError, match="policy version"):
        updater.update(stale)
    rollout = make_rollout(policy, critic, version=2)
    stats = updater.update(rollout)
    assert stats.accepted
    assert stats.policy_version_after == 3


def test_updater_routes_actor_and_critic_observations_to_separate_networks():
    torch.manual_seed(24)
    policy, critic = GaussianPolicy(), nn.Linear(3, 1)
    config = PPOConfig(target_kl=None, epochs=2, minibatch_size=2, flat_anchor_coef=0.1)
    updater = OnlinePPOUpdater(policy, critic, config)
    observations = torch.stack((torch.arange(6.0), torch.linspace(-1.0, 1.0, 6)), dim=1)
    context = torch.cat((observations, torch.arange(10.0, 16.0).unsqueeze(-1)), dim=1)
    with torch.no_grad():
        distribution = policy.get_distribution(observations)
        actions = distribution.sample()
        old_values = critic(context)
        old_log_probs = distribution.log_prob(actions).sum(-1, keepdim=True)
        anchor_means = policy.mean_action(observations)
    rollout = PPORollout(
        observations, actions, old_log_probs, old_values, torch.ones(6, 1),
        torch.zeros(6, 1), torch.zeros(6, 1), old_values,
        critic_observations=context,
    )
    actor_inputs, critic_inputs = [], []
    actor_hook = policy.linear.register_forward_pre_hook(
        lambda module, args: actor_inputs.append(args[0].detach().clone())
    )
    critic_hook = critic.register_forward_pre_hook(
        lambda module, args: critic_inputs.append(args[0].detach().clone())
    )
    try:
        stats = updater.update(
            rollout, flat_anchor_observations=observations, flat_anchor_means=anchor_means,
        )
    finally:
        actor_hook.remove()
        critic_hook.remove()

    assert stats.accepted
    assert actor_inputs and all(value.shape[1] == 2 for value in actor_inputs)
    for value in actor_inputs:
        torch.testing.assert_close(value, observations[value[:, 0].long()])
    seen = torch.cat(critic_inputs)
    expected = context.repeat(config.epochs, 1)
    torch.testing.assert_close(seen[seen[:, 0].argsort()], expected[expected[:, 0].argsort()])


def test_legacy_shared_observations_match_explicit_critic_input_updates():
    policy, critic = GaussianPolicy(), Value()
    config = PPOConfig(target_kl=None, minibatch_size=4)
    shared = OnlinePPOUpdater(policy, critic, config)
    rollout = make_rollout(policy, critic)
    explicit = OnlinePPOUpdater(GaussianPolicy(), Value(), config)
    explicit.load_state_dict(shared.state_dict())

    torch.manual_seed(71)
    shared_stats = shared.update(rollout)
    torch.manual_seed(71)
    explicit_stats = explicit.update(replace(rollout, critic_observations=rollout.observations))

    assert rollout.critic_observations is None
    assert shared_stats == explicit_stats
    for model_name in ("policy", "critic"):
        for key, value in getattr(shared, model_name).state_dict().items():
            torch.testing.assert_close(getattr(explicit, model_name).state_dict()[key], value)


@pytest.mark.parametrize("context", [
    [[0.0, 0.0, 0.0]],
    torch.tensor(1.0),
    torch.zeros(11, 3),
    torch.full((12, 3), float("nan")),
    torch.full((12, 3), float("inf")),
    torch.zeros(12, 3, dtype=torch.complex64),
])
def test_rollout_rejects_invalid_critic_observations(context):
    policy, critic = GaussianPolicy(), Value()
    rollout = make_rollout(policy, critic)
    with pytest.raises(ValueError, match="critic_observations"):
        replace(rollout, critic_observations=context)


def test_rejected_large_kl_restores_policy_critic_and_optimizers():
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(
        policy, critic, PPOConfig(target_kl=1e-12, actor_lr=1.0, critic_lr=1.0, minibatch_size=12), policy_version=0
    )
    rollout = make_rollout(policy, critic, version=0)
    before = copy.deepcopy(updater.policy.state_dict())
    stats = updater.update(rollout)
    assert not stats.accepted
    assert stats.rejection_reason == "target_kl"
    for key, value in before.items():
        torch.testing.assert_close(updater.policy.state_dict()[key], value)
    assert updater.policy_version == 0


def test_reference_anchor_limit_rejects_policy_drift_and_restores_actor():
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(
        policy,
        critic,
        PPOConfig(
            target_kl=None,
            max_anchor_mse=1e-12,
            actor_lr=1.0,
            minibatch_size=12,
        ),
    )
    rollout = make_rollout(policy, critic)
    anchor_observations = rollout.observations.detach().clone()
    with torch.no_grad():
        anchor_means = policy.mean_action(anchor_observations).detach().clone()
    before = copy.deepcopy(policy.state_dict())
    stats = updater.update(
        rollout,
        flat_anchor_observations=anchor_observations,
        flat_anchor_means=anchor_means,
    )
    assert not stats.accepted
    assert stats.rejection_reason == "anchor_mse"
    assert stats.candidate_anchor_mse is not None
    assert stats.candidate_anchor_mse > updater.config.max_anchor_mse
    for key, value in before.items():
        torch.testing.assert_close(policy.state_dict()[key], value)


def make_diagnostic_rollout(policy, critic, *, rewards, valid_mask):
    with torch.no_grad():
        critic.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.linear.bias.zero_()
        observations = torch.tensor([[1.0, 0.0], [99.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        distribution = policy.get_distribution(observations)
        actions = distribution.mean + torch.tensor([[0.2], [0.4], [-0.3], [0.1]])
        return PPORollout(
            observations=observations,
            actions=actions,
            old_log_probs=distribution.log_prob(actions).sum(-1, keepdim=True),
            old_values=critic(observations),
            rewards=torch.tensor(rewards).reshape(-1, 1),
            terminated=torch.ones(4, 1),
            truncated=torch.zeros(4, 1),
            next_values=torch.zeros(4, 1),
            valid_mask=torch.tensor(valid_mask).reshape(-1, 1),
        )


@pytest.mark.parametrize("target_kl,accepted", [(None, True), (1e-12, False)])
def test_update_diagnostics_measure_raw_valid_rows_before_acceptance_or_rejection(target_kl, accepted):
    torch.manual_seed(21)
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(
        policy,
        critic,
        PPOConfig(target_kl=target_kl, actor_lr=0.1, critic_lr=0.1, epochs=3, minibatch_size=4),
    )
    # The masked row must not influence means, variance, or the unique count.
    # Every row terminates, so its return is exactly its reward.
    rollout = make_diagnostic_rollout(
        policy, critic, rewards=[2.0, 1000.0, 4.0, 8.0], valid_mask=[1, 0, 1, 1]
    )
    critic_before = copy.deepcopy(critic.state_dict())
    stats = updater.update(rollout)

    assert stats.accepted is accepted
    assert stats.value_mean == pytest.approx(7.0 / 3.0)
    assert stats.return_mean == pytest.approx(14.0 / 3.0)
    assert stats.advantage_mean == pytest.approx(7.0 / 3.0)
    assert stats.advantage_std == pytest.approx((14.0 / 9.0) ** 0.5)
    assert stats.explained_variance == pytest.approx(0.75)
    assert stats.num_valid_samples == 3
    assert stats.total_rollout_samples == 4
    assert stats.num_samples == 9  # Three epochs, preserving the existing metric.
    json.dumps(stats.__dict__, allow_nan=False)
    if accepted:
        with torch.no_grad():
            current_value_mean = float(critic(rollout.observations)[rollout.valid_mask.bool()].mean())
        assert current_value_mean != pytest.approx(stats.value_mean)
    else:
        assert stats.rejection_reason == "target_kl"
        for key, value in critic_before.items():
            torch.testing.assert_close(critic.state_dict()[key], value)


@pytest.mark.parametrize("valid_mask", [[1, 0, 1, 1], [1, 0, 0, 0]])
def test_constant_valid_returns_have_no_explained_variance_and_serialize_as_null(valid_mask):
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, minibatch_size=4))
    rollout = make_diagnostic_rollout(
        policy, critic, rewards=[7.0, 1000.0, 7.0, 7.0], valid_mask=valid_mask
    )
    stats = updater.update(rollout)
    assert stats.accepted
    assert stats.return_mean == 7.0
    assert stats.explained_variance is None
    assert stats.num_valid_samples == sum(valid_mask)
    if sum(valid_mask) == 1:
        assert stats.advantage_std == 0.0
    assert json.loads(json.dumps(stats.__dict__, allow_nan=False))["explained_variance"] is None


def test_rollout_rejects_mixed_episode_endings_and_non_requested_actions():
    policy, critic = GaussianPolicy(), Value()
    rollout = make_rollout(policy, critic)
    with pytest.raises(ValueError, match="both terminated and truncated"):
        PPORollout(
            rollout.observations,
            rollout.actions,
            rollout.old_log_probs,
            rollout.old_values,
            rollout.rewards,
            torch.ones(12, 1),
            torch.ones(12, 1),
            rollout.next_values,
        )
    with pytest.raises(ValueError, match="requested"):
        PPORollout(
            rollout.observations,
            rollout.actions,
            rollout.old_log_probs,
            rollout.old_values,
            rollout.rewards,
            rollout.terminated,
            rollout.truncated,
            rollout.next_values,
            action_semantics="executed",
        )


def test_updater_state_dict_round_trip_includes_version_and_optimizer_state():
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, minibatch_size=4))
    updater.update(make_rollout(policy, critic))
    state = copy.deepcopy(updater.state_dict())
    restored = OnlinePPOUpdater(GaussianPolicy(), Value(), PPOConfig(target_kl=None, minibatch_size=4))
    restored.load_state_dict(state)
    assert restored.policy_version == updater.policy_version
    for key, value in updater.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[key], value)
    broken = copy.deepcopy(state)
    broken["policy_version"] = -1
    with pytest.raises(ValueError, match="policy_version"):
        restored.load_state_dict(broken)


@pytest.mark.parametrize('field,value', [('terrain_anchor_coef', -1), ('terrain_anchor_coef', float('nan')),
                                       ('max_terrain_anchor_mse', 0), ('max_terrain_anchor_mse', float('inf'))])
def test_terrain_anchor_rejects_invalid_config(field, value):
    with pytest.raises(ValueError, match=field):
        PPOConfig(**{field: value})


def test_terrain_anchor_requires_explicit_teacher():
    with pytest.raises(ValueError, match='fixed terrain reference'):
        OnlinePPOUpdater(GaussianPolicy(), Value(), PPOConfig(terrain_anchor_coef=100))


def test_terrain_anchor_rejection_restores_actor_critic_and_both_optimizers():
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(policy, critic,
        PPOConfig(target_kl=None, actor_lr=0.01, critic_lr=0.01, max_terrain_anchor_mse=1e-14),
        terrain_reference_policy=policy)
    # Populate optimizer states so restoration covers more than empty Adam slots.
    for model, optimizer in ((policy, updater.actor_optimizer), (critic, updater.critic_optimizer)):
        optimizer.zero_grad()
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
    before = updater.state_dict()
    stats = updater.update(make_rollout(policy, critic))
    assert not stats.accepted and stats.rejection_reason == 'terrain_anchor_mse'
    assert stats.candidate_terrain_anchor_mse > 1e-14
    after = updater.state_dict()
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b): equal(x, y)
        else:
            assert a == b
    equal(before, after)


def test_fixed_teacher_survives_caller_changes_and_training_checkpoint_restore():
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(policy, critic, terrain_reference_policy=policy)
    observations = torch.randn(4, 2)
    original = policy.mean_action(observations).detach().clone()
    with torch.no_grad():
        policy.linear.bias.add_(3)
    updater.load_state_dict(updater.state_dict())
    torch.testing.assert_close(updater.terrain_reference_policy.mean_action(observations), original)
    assert all(not p.requires_grad for p in updater.terrain_reference_policy.parameters())
    assert not updater.terrain_reference_policy.training


def test_terrain_anchor_soft_penalty_reduces_drift_without_flat_data():
    torch.manual_seed(5)
    teacher, critic = GaussianPolicy(), Value()
    candidate = copy.deepcopy(teacher)
    with torch.no_grad():
        candidate.linear.bias.add_(0.1)
    rollout = make_rollout(candidate, critic, length=64)
    anchored = OnlinePPOUpdater(copy.deepcopy(candidate), copy.deepcopy(critic),
        PPOConfig(target_kl=None, actor_lr=0.005, epochs=5, terrain_anchor_coef=100),
        terrain_reference_policy=teacher)
    control = OnlinePPOUpdater(copy.deepcopy(candidate), copy.deepcopy(critic),
        PPOConfig(target_kl=None, actor_lr=0.005, epochs=5), terrain_reference_policy=teacher)
    torch.manual_seed(3)
    a = anchored.update(rollout)
    torch.manual_seed(3)
    c = control.update(rollout)
    assert a.accepted and c.accepted
    assert a.candidate_terrain_anchor_mse < c.candidate_terrain_anchor_mse
    assert all(p.grad is None for p in anchored.terrain_reference_policy.parameters())


def test_terrain_anchor_ignores_invalid_rollout_rows():
    policy, critic = GaussianPolicy(), Value()
    teacher = copy.deepcopy(policy)
    with torch.no_grad():
        teacher.linear.weight.add_(1)
    rollout = make_rollout(policy, critic)
    obs = torch.zeros_like(rollout.observations)
    obs[-1].fill_(1e6)
    mask = torch.ones_like(rollout.rewards)
    mask[-1] = 0
    rollout = replace(rollout, observations=obs, valid_mask=mask,
                      old_log_probs=policy.get_distribution(obs).log_prob(rollout.actions).detach())
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, max_terrain_anchor_mse=1e-5),
                               terrain_reference_policy=teacher)
    stats = updater.update(rollout)
    assert stats.accepted
    assert stats.candidate_terrain_anchor_mse < 1e-5


def test_critic_warmup_never_uses_actor_likelihood_or_optimizer(monkeypatch):
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(epochs=2, minibatch_size=4))
    rollout = make_rollout(policy, critic)
    before = copy.deepcopy(updater.state_dict())
    monkeypatch.setattr(policy, "evaluate_actions", lambda *args: pytest.fail("warmup used actor likelihood"))
    stats = updater.update_critic(rollout)
    assert stats.accepted and not stats.actor_updated and stats.critic_minibatches == 6
    assert updater.policy_version == 0
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before["policy"][key])
    assert updater.actor_optimizer.state_dict() == before["actor_optimizer"]
    assert any(not torch.equal(v, before["critic"][k]) for k, v in critic.state_dict().items())
    assert stats.value_rmse_after < stats.value_rmse_before
    json.dumps(stats.__dict__, allow_nan=False)


def test_critic_warmup_uses_terminal_return_and_rolls_back_on_bad_gradient(monkeypatch):
    policy, critic = GaussianPolicy(), Value()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(gamma=.5, epochs=1))
    rollout = make_rollout(policy, critic, length=1)
    rollout = replace(rollout, terminated=torch.ones(1, 1), rewards=torch.full((1, 1), 3.))
    stats = updater.update_critic(rollout)
    assert stats.return_mean == pytest.approx(3.) and stats.explained_variance is None
    before = copy.deepcopy(updater.state_dict())
    hook = next(critic.parameters()).register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(ValueError, match="gradient"):
        updater.update_critic(rollout)
    hook.remove()
    for key, value in critic.state_dict().items():
        assert torch.equal(value, before["critic"][key])
    for pid, state in updater.critic_optimizer.state_dict()["state"].items():
        for key, value in state.items():
            assert torch.equal(value, before["critic_optimizer"]["state"][pid][key])
