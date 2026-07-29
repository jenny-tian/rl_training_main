import copy

import pytest
import torch
from dataclasses import replace

from rl_training.online_adaptation.metrics import AdaptationPerformanceDecision
from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.sac import (
    CandidateActorRejected,
    SACConfig,
    SACUpdater,
    StableStateManager,
)
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


def assert_state_dict_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_dict_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_state_dict_equal(actual_item, expected_item)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        assert actual == expected


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
    assert stats.rejection_reason == "nonfinite_batch"


def test_candidate_validator_rejection_restores_the_full_pre_update_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    warmup_stats = updater.update(make_batch())
    assert warmup_stats.accepted
    assert warmup_stats.version == 1
    before = copy.deepcopy(updater.state_dict())
    seen = []

    def validator(candidate, version):
        seen.append((candidate, version))
        return "nonfinite_policy_drift"

    stats = updater.update(make_batch(), candidate_validator=validator)

    assert seen == [(actor, 2)]
    assert not stats.accepted
    assert not stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 1
    assert stats.rejection_reason == "nonfinite_policy_drift"
    assert_state_dict_equal(updater.state_dict(), before)


def test_critic_only_update_does_not_call_candidate_validator():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    called = False

    def validator(candidate, version):
        nonlocal called
        called = True
        raise AssertionError("critic-only updates must not validate actor candidates")

    stats = updater.update(make_batch(), update_actor=False, candidate_validator=validator)

    assert stats.accepted
    assert stats.rejection_reason is None
    assert stats.version == 0
    assert not called


def test_candidate_validator_exception_restores_the_full_pre_update_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())

    def validator(candidate, version):
        raise RuntimeError("validator failed")

    with pytest.raises(RuntimeError, match="validator failed"):
        updater.update(make_batch(), candidate_validator=validator)

    assert_state_dict_equal(updater.state_dict(), before)


def test_candidate_actor_rejected_exception_propagates_and_restores_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())

    def validator(candidate, version):
        raise CandidateActorRejected("validator rejected via exception")

    with pytest.raises(CandidateActorRejected, match="validator rejected via exception"):
        updater.update(make_batch(), candidate_validator=validator)

    assert_state_dict_equal(updater.state_dict(), before)


def test_candidate_validator_floating_point_error_propagates_and_restores_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())

    def validator(candidate, version):
        raise FloatingPointError("validator numerical failure")

    with pytest.raises(FloatingPointError, match="validator numerical failure"):
        updater.update(make_batch(), candidate_validator=validator)

    assert_state_dict_equal(updater.state_dict(), before)


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
    assert stats.rejection_reason is None
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
    before = copy.deepcopy(updater.state_dict())
    actor_batch = make_batch()
    actor_batch = replace(actor_batch, current_frame=torch.full_like(actor_batch.current_frame, float("nan")))

    stats = updater.update(critic_batch=make_batch(), actor_batch=actor_batch, update_actor=True)

    assert not stats.accepted
    assert not stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 0
    assert stats.rejection_reason == "nonfinite_update"
    assert_state_dict_equal(updater.state_dict(), before)


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


def test_stable_state_manager_observes_adaptation_decisions():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager()
    manager.checkpoint(updater)
    updater.version = 2
    improved = AdaptationPerformanceDecision(
        cost=0.8,
        improved=True,
        degraded=False,
        should_rollback=False,
        rollback_reason=None,
    )

    assert not manager.observe_adaptation(improved, updater)
    updater.version = 3
    rollback = AdaptationPerformanceDecision(
        cost=1.0,
        improved=False,
        degraded=True,
        should_rollback=True,
        rollback_reason="performance_degradation",
    )

    assert manager.observe_adaptation(rollback, updater)
    assert updater.version == 2
