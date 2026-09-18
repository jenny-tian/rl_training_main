import copy
import io
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rl_training.online_adaptation.online_session import OnlinePPOSession
from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
from rl_training.online_adaptation.ppo_rollout import PPORollout
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher


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


def block(policy, critic, version=0, n=2, *, critic_observations=None):
    obs = torch.randn(n, 2)
    actions = policy.get_distribution(obs).sample()
    old_lp = policy.get_distribution(obs).log_prob(actions).detach()
    old_v = critic(obs if critic_observations is None else critic_observations).detach()
    return PPORollout(
        obs, actions, old_lp, old_v, torch.ones(n, 1), torch.zeros(n, 1),
        torch.zeros(n, 1), old_v, policy_version=version,
        critic_observations=critic_observations,
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


@pytest.mark.parametrize("legacy", [False, True])
def test_session_checkpoint_round_trip_restores_pending_rollout_and_updater(legacy):
    policy, critic = Policy(), Critic()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None))
    session = OnlinePPOSession(updater, rollout_capacity=4)
    session.append(block(policy, critic, n=2))
    state = copy.deepcopy(session.state_dict())
    if legacy:
        del state["pending_rollout"]["critic_observations"]
    restored = OnlinePPOSession(
        OnlinePPOUpdater(Policy(), Critic(), PPOConfig(target_kl=None)), rollout_capacity=4
    )
    restored.load_state_dict(state)
    assert restored.pending_transitions == 2
    assert restored.counters.transitions_collected == 2
    assert restored.updater.policy_version == 0
    assert restored.state_dict()["pending_rollout"]["critic_observations"] is None
    with pytest.raises(ValueError, match="capacity"):
        OnlinePPOSession(
            OnlinePPOUpdater(Policy(), Critic(), PPOConfig(target_kl=None)), rollout_capacity=8
        ).load_state_dict(state)
    restored.append(block(restored.updater.policy, restored.updater.critic, n=2))
    assert restored.update().accepted


def test_pending_critic_observations_survive_combination_serialization_and_resume():
    policy, critic = Policy(), nn.Linear(3, 1)
    config = PPOConfig(target_kl=None, minibatch_size=2)
    session = OnlinePPOSession(OnlinePPOUpdater(policy, critic, config), rollout_capacity=6)
    first = torch.arange(6.0).reshape(2, 3)
    second = torch.arange(6.0, 12.0).reshape(2, 3)
    session.append(block(policy, critic, critic_observations=first))
    session.append(block(policy, critic, critic_observations=second))
    state = session.state_dict()
    expected = torch.cat((first, second))
    torch.testing.assert_close(state["pending_rollout"]["critic_observations"], expected)
    assert state["pending_rollout"]["observations"].shape == (4, 2)
    serialized = io.BytesIO()
    torch.save(state, serialized)
    serialized.seek(0)
    restored = OnlinePPOSession(
        OnlinePPOUpdater(Policy(), nn.Linear(3, 1), config), rollout_capacity=6,
    )

    restored.load_state_dict(torch.load(serialized, map_location="cpu", weights_only=False))

    pending = restored.state_dict()["pending_rollout"]
    torch.testing.assert_close(pending["critic_observations"], expected)
    torch.testing.assert_close(pending["observations"], state["pending_rollout"]["observations"])
    assert restored.pending_transitions == restored.counters.transitions_collected == 4
    third = torch.arange(12.0, 18.0).reshape(2, 3)
    restored.append(block(restored.updater.policy, restored.updater.critic, critic_observations=third))
    stats = restored.update()
    assert stats.accepted and stats.num_valid_samples == 6


@pytest.mark.parametrize("first_has_context", [False, True])
def test_session_rejects_mixed_critic_observation_blocks_without_mutation(first_has_context):
    policy, critic = Policy(), Critic()
    session = OnlinePPOSession(
        OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None)), rollout_capacity=4,
    )
    plain = block(policy, critic)
    contextual = replace(plain, critic_observations=plain.observations)
    first, second = (contextual, plain) if first_has_context else (plain, contextual)
    session.append(first)
    before = session.state_dict()

    with pytest.raises(ValueError, match="mix blocks with and without critic observations"):
        session.append(second)

    after = session.state_dict()
    assert session.pending_transitions == 2
    assert after["counters"] == before["counters"]
    torch.testing.assert_close(after["pending_rollout"]["observations"], before["pending_rollout"]["observations"])
    if first_has_context:
        torch.testing.assert_close(after["pending_rollout"]["critic_observations"], first.critic_observations)
    else:
        assert after["pending_rollout"]["critic_observations"] is None
    session.clear()
    session.append(second)
    assert session.pending_transitions == 2


def test_terminal_rollback_restores_last_stable_candidate_and_invalidates_versioned_rollout():
    torch.manual_seed(4)
    policy, critic = Policy(), Critic()
    initial_policy = copy.deepcopy(policy.state_dict())
    publisher = DoubleBufferedActorPublisher(policy)
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, minibatch_size=2))
    session = OnlinePPOSession(updater, rollout_capacity=2, publisher=publisher)

    session.append(block(policy, critic, n=2))
    accepted = session.update()
    assert accepted is not None and accepted.accepted
    candidate_version = updater.policy_version
    assert candidate_version == 1
    with torch.no_grad():
        policy.linear.weight.add_(10.0)
    session.append(block(policy, critic, version=candidate_version, n=1))

    rollback = session.rollback("terminated")

    assert rollback.policy_version_before == candidate_version
    assert rollback.restored_stable_version == 0
    assert rollback.policy_version_after == candidate_version + 1
    assert rollback.discarded_transitions == 1
    assert session.pending_transitions == 0
    assert session.counters.rollbacks == 1
    assert session.counters.terminal_rollbacks == 1
    assert session.counters.last_rollback_reason == "terminated"
    assert updater.policy_version == 2
    for key, value in initial_policy.items():
        torch.testing.assert_close(policy.state_dict()[key], value)
    published, published_version = publisher.current()
    assert published_version == 2
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(published.state_dict()[key], value)
    with pytest.raises(ValueError, match="policy version"):
        session.append(block(policy, critic, version=1, n=1))


def test_severe_nonfoot_contact_rollback_is_audited_separately():
    policy, critic = Policy(), Critic()
    session = OnlinePPOSession(
        OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None)), rollout_capacity=4
    )
    session.append(block(policy, critic, n=1))
    rollback = session.rollback("severe_nonfoot_contact")
    assert rollback.reason == "severe_nonfoot_contact"
    assert session.counters.rollbacks == 1
    assert session.counters.nonfoot_contact_rollbacks == 1


def test_finalization_restores_unverified_candidate_and_discards_partial_rollout():
    torch.manual_seed(9)
    policy, critic = Policy(), Critic()
    initial_policy = copy.deepcopy(policy.state_dict())
    publisher = DoubleBufferedActorPublisher(policy)
    session = OnlinePPOSession(
        OnlinePPOUpdater(
            policy,
            critic,
            PPOConfig(target_kl=None, minibatch_size=2),
        ),
        rollout_capacity=2,
        publisher=publisher,
    )
    session.append(block(policy, critic, n=2))
    update = session.update()
    assert update is not None and update.accepted
    candidate_version = session.updater.policy_version
    session.append(block(policy, critic, version=candidate_version, n=1))

    finalization = session.finalize_for_checkpoint()

    assert finalization.policy_version_before == candidate_version
    assert finalization.stable_version_before == 0
    assert finalization.policy_version_after == candidate_version + 1
    assert finalization.discarded_transitions == 1
    assert finalization.restored_unverified_candidate
    assert session.pending_transitions == 0
    assert session.counters.discarded_transitions == 1
    assert session.counters.rollbacks == 0
    for key, value in initial_policy.items():
        torch.testing.assert_close(policy.state_dict()[key], value)
    published, published_version = publisher.current()
    assert published_version == candidate_version + 1
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(published.state_dict()[key], value)
    state = session.state_dict()
    assert not state["has_unverified_candidate"]
    assert state["stable_state"]["policy_version"] == candidate_version + 1


def test_finalization_keeps_initial_policy_when_only_partial_rollout_exists():
    policy, critic = Policy(), Critic()
    session = OnlinePPOSession(
        OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None)),
        rollout_capacity=4,
    )
    session.append(block(policy, critic, n=1))

    finalization = session.finalize_for_checkpoint()

    assert not finalization.restored_unverified_candidate
    assert finalization.policy_version_before == 0
    assert finalization.policy_version_after == 0
    assert finalization.discarded_transitions == 1
    assert session.counters.update_attempts == 0
    assert session.counters.discarded_transitions == 1


def test_finalization_can_keep_a_terminal_learning_candidate():
    policy, critic = Policy(), Critic()
    session = OnlinePPOSession(
        OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, minibatch_size=2)),
        rollout_capacity=2,
    )
    session.append(block(policy, critic, n=2))
    update = session.update()
    assert update is not None and update.accepted
    finalization = session.finalize_for_checkpoint(restore_unverified_candidate=False)
    assert not finalization.restored_unverified_candidate
    assert finalization.policy_version_after == 1
    assert session.state_dict()["has_unverified_candidate"]
    assert session.state_dict()["stable_state"]["policy_version"] == 0


@pytest.mark.parametrize("terminal_reward", [-8.0, 5.0])
def test_boundary_reward_is_used_and_unverified_candidate_keeps_rollback_target(terminal_reward):
    torch.manual_seed(12)
    policy, critic = Policy(), Critic()
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(target_kl=None, minibatch_size=4))
    session = OnlinePPOSession(updater, rollout_capacity=4)
    session.append(block(policy, critic, n=4))
    session.update()
    stable = copy.deepcopy(session.state_dict()["stable_state"])
    ending = block(policy, critic, version=1, n=3)
    ending.rewards[-1] = terminal_reward
    ending.terminated[-1] = 1.0
    ending.next_values[-1] = 0.0
    session.append(ending)
    seen = []
    original_update = updater.update

    def recording_update(rollout, **kwargs):
        seen.append(rollout)
        return original_update(rollout, **kwargs)

    updater.update = recording_update
    stats = session.update(force=True)
    assert stats.accepted
    assert seen[0].rewards[-1].item() == terminal_reward
    assert seen[0].terminated[-1].item() == 1.0
    final = session.finalize_for_checkpoint(restore_unverified_candidate=False)
    assert final.discarded_transitions == 0
    restored = OnlinePPOSession(
        OnlinePPOUpdater(Policy(), Critic(), PPOConfig(target_kl=None)), rollout_capacity=4,
    )
    restored.load_state_dict(session.state_dict())
    assert restored.state_dict()["has_unverified_candidate"]
    restored.rollback("failed_probe")
    for key, value in stable["policy"].items():
        torch.testing.assert_close(restored.updater.policy.state_dict()[key], value)
