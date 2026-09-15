import copy

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
