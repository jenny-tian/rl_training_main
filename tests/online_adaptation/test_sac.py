from dataclasses import replace

import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.sac import SACConfig, SACUpdater, StableStateManager
from rl_training.online_adaptation.types import TransitionBatch


def make_batch(reward_value=1.0):
    return TransitionBatch(
        history=torch.randn(32, 25, 49),
        current_frame=torch.randn(32, 49),
        base_action=torch.randn(32, 12),
        residual_action=torch.zeros(32, 12),
        reward=torch.full((32, 1), reward_value),
        next_history=torch.randn(32, 25, 49),
        next_frame=torch.randn(32, 49),
        next_base_action=torch.randn(32, 12),
        done=torch.zeros(32, 1),
    )


def test_sac_update_changes_version_and_target_parameters():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    critic = TwinCritic(49, 12, 64)
    updater = SACUpdater(actor, critic, SACConfig())
    before = [p.detach().clone() for p in updater.target_critic.parameters()]
    stats = updater.update(make_batch())
    assert stats.accepted
    assert stats.critic_updated
    assert stats.actor_updated
    assert stats.version == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, updater.target_critic.parameters()))


def test_nonfinite_batch_is_rejected_without_version_change():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    stats = updater.update(make_batch(float("nan")))
    assert not stats.accepted
    assert stats.version == 0


def test_critic_only_update_keeps_actor_and_version_unchanged():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    target_before = [parameter.detach().clone() for parameter in updater.target_critic.parameters()]

    stats = updater.update(critic_batch=make_batch(), actor_batch=None, update_actor=False)

    assert stats.accepted
    assert stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 0
    assert all(torch.equal(before, after) for before, after in zip(actor_before, actor.parameters()))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, updater.target_critic.parameters())
    )


def test_full_update_accepts_a_separate_actor_batch():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())

    stats = updater.update(critic_batch=make_batch(), actor_batch=make_batch(0.5), update_actor=True)

    assert stats.accepted
    assert stats.critic_updated
    assert stats.actor_updated
    assert stats.version == 1


def test_nonfinite_actor_batch_rolls_back_the_preceding_critic_step():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    critic_before = [parameter.detach().clone() for parameter in updater.critic.parameters()]
    actor_batch = make_batch()
    actor_batch = replace(actor_batch, current_frame=torch.full_like(actor_batch.current_frame, float("nan")))

    stats = updater.update(critic_batch=make_batch(), actor_batch=actor_batch, update_actor=True)

    assert not stats.accepted
    assert not stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 0
    assert all(
        torch.equal(before, after) for before, after in zip(critic_before, updater.critic.parameters())
    )


def test_stable_state_rolls_back_after_sustained_degradation():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager(patience=3, min_delta=0.01)
    manager.observe(score=1.0, updater=updater)
    updater.version = 7
    assert not manager.observe(score=0.9, updater=updater)
    assert not manager.observe(score=0.8, updater=updater)
    assert manager.observe(score=0.7, updater=updater)
    assert updater.version == 0


def test_stable_state_can_roll_back_to_explicit_checkpoint():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager()
    manager.checkpoint(updater)
    updater.version = 11

    assert manager.rollback(updater)
    assert updater.version == 0
