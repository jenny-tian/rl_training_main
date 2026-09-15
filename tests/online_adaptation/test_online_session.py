import copy

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rl_training.online_adaptation.online_session import OnlinePPOSession
from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
from rl_training.online_adaptation.ppo_rollout import PPORollout


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def get_distribution(self, obs):
        return Normal(self.linear(obs), self.log_std.exp())

    def evaluate_actions(self, obs, actions):
        dist = self.get_distribution(obs)
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, obs):
        return self.linear(obs)


def block(policy, critic, version=0, n=2):
    obs = torch.randn(n, 2)
    actions = policy.get_distribution(obs).sample()
    old_lp = policy.get_distribution(obs).log_prob(actions).detach()
    old_v = critic(obs).detach()
    return PPORollout(
        obs, actions, old_lp, old_v, torch.ones(n, 1), torch.zeros(n, 1),
        torch.zeros(n, 1), old_v, policy_version=version,
    )


def test_session_accumulates_exact_capacity_and_publishes_after_update():
    policy, critic = Policy(), Critic()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, minibatch_size=4))
    session = OnlinePPOSession(updater, rollout_capacity=4)
    session.append(block(policy, critic, n=2))
    assert not session.ready
    assert session.update() is None
    session.append(block(policy, critic, n=2))
    assert session.ready
    stats = session.update()
    assert stats is not None and stats.accepted
    assert session.pending_transitions == 0
    assert session.counters.transitions_collected == 4
    assert session.counters.accepted_updates == 1


def test_session_refuses_cross_version_or_over_capacity_blocks():
    policy, critic = Policy(), Critic()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None))
    session = OnlinePPOSession(updater, rollout_capacity=3)
    session.append(block(policy, critic, n=2))
    with pytest.raises(ValueError, match="policy version"):
        session.append(block(policy, critic, version=1, n=1))
    with pytest.raises(ValueError, match="capacity"):
        session.append(block(policy, critic, n=2))


def test_session_checkpoint_round_trip_restores_pending_rollout_and_updater():
    policy, critic = Policy(), Critic()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None))
    session = OnlinePPOSession(updater, rollout_capacity=4)
    session.append(block(policy, critic, n=2))
    state = copy.deepcopy(session.state_dict())
    restored = OnlinePPOSession(
        OnlinePPOUpdater(Policy(), Critic(), PPOConfig(target_kl=None)), rollout_capacity=4
    )
    restored.load_state_dict(state)
    assert restored.pending_transitions == 2
    assert restored.counters.transitions_collected == 2
    assert restored.updater.policy_version == 0
    with pytest.raises(ValueError, match="capacity"):
        OnlinePPOSession(
            OnlinePPOUpdater(Policy(), Critic(), PPOConfig(target_kl=None)), rollout_capacity=8
        ).load_state_dict(state)
