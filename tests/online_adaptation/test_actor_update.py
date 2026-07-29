import copy

import pytest
import torch

from rl_training.online_adaptation.actor_update import update_and_publish_actor
from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.policy_drift import ActorDriftMonitor
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher
from rl_training.online_adaptation.sac import SACConfig, SACUpdater
from rl_training.online_adaptation.types import TransitionBatch


def make_batch(value: float = 0.0) -> TransitionBatch:
    return TransitionBatch(
        history=torch.full((4, 3, 2), value),
        current_frame=torch.full((4, 2), value),
        base_action=torch.full((4, 1), value),
        residual_action=torch.zeros(4, 1),
        reward=torch.full((4, 1), value),
        next_history=torch.full((4, 3, 2), value),
        next_frame=torch.full((4, 2), value),
        next_base_action=torch.full((4, 1), value),
        done=torch.zeros(4, 1),
    )


def make_components() -> tuple[SACUpdater, DoubleBufferedActorPublisher, ActorDriftMonitor]:
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    updater = SACUpdater(actor, TwinCritic(2, 1, 4), SACConfig())
    publisher = DoubleBufferedActorPublisher(actor)
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    return updater, publisher, monitor


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


def test_active_finite_candidate_is_diagnosed_and_published():
    updater, publisher, monitor = make_components()

    outcome = update_and_publish_actor(
        updater=updater,
        publisher=publisher,
        drift_monitor=monitor,
        critic_batch=make_batch(0.1),
        actor_batch=make_batch(0.2),
        adaptation_episode=1,
        update_step=7,
        episode_is_active=lambda: True,
    )

    assert outcome.stats.accepted
    assert outcome.stats.actor_updated
    assert outcome.policy_drift is not None
    assert outcome.policy_drift.actor_version == outcome.stats.version
    assert outcome.policy_drift.update_step == 7
    _, published_version = publisher.current()
    assert published_version == outcome.stats.version
    published_actor, _ = publisher.current()
    assert all(
        torch.equal(published_parameter, training_parameter)
        for published_parameter, training_parameter in zip(
            published_actor.parameters(), updater.actor.parameters(), strict=True
        )
    )


def test_inactive_episode_rejects_without_measuring_or_publishing():
    updater, publisher, monitor = make_components()

    outcome = update_and_publish_actor(
        updater=updater,
        publisher=publisher,
        drift_monitor=monitor,
        critic_batch=make_batch(0.1),
        actor_batch=make_batch(0.2),
        adaptation_episode=1,
        update_step=7,
        episode_is_active=lambda: False,
    )

    assert not outcome.stats.accepted
    assert outcome.stats.rejection_reason == "stale_adaptation_episode"
    assert outcome.policy_drift is None
    assert updater.version == 0
    _, published_version = publisher.current()
    assert published_version == 0
    assert monitor._diagnostic_inputs is None


def test_monitor_floating_point_error_rejects_without_publishing(monkeypatch):
    updater, publisher, monitor = make_components()

    def raise_nonfinite(*args, **kwargs):
        raise FloatingPointError("non-finite policy drift")

    monkeypatch.setattr(monitor, "measure", raise_nonfinite)
    outcome = update_and_publish_actor(
        updater=updater,
        publisher=publisher,
        drift_monitor=monitor,
        critic_batch=make_batch(0.1),
        actor_batch=make_batch(0.2),
        adaptation_episode=1,
        update_step=7,
        episode_is_active=lambda: True,
    )

    assert not outcome.stats.accepted
    assert outcome.stats.rejection_reason == "nonfinite_policy_drift"
    assert outcome.policy_drift is None
    assert updater.version == 0
    _, published_version = publisher.current()
    assert published_version == 0


def test_unexpected_monitor_error_propagates_and_restores_updater(monkeypatch):
    updater, publisher, monitor = make_components()
    before = copy.deepcopy(updater.state_dict())

    def raise_unexpected(*args, **kwargs):
        raise RuntimeError("drift monitor failed")

    monkeypatch.setattr(monitor, "measure", raise_unexpected)
    with pytest.raises(RuntimeError, match="drift monitor failed"):
        update_and_publish_actor(
            updater=updater,
            publisher=publisher,
            drift_monitor=monitor,
            critic_batch=make_batch(0.1),
            actor_batch=make_batch(0.2),
            adaptation_episode=1,
            update_step=7,
            episode_is_active=lambda: True,
        )

    assert_state_dict_equal(updater.state_dict(), before)
    _, published_version = publisher.current()
    assert published_version == 0
