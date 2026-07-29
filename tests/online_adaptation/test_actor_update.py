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


def test_active_finite_candidate_is_diagnosed_before_publishing(monkeypatch):
    updater, publisher, monitor = make_components()
    events = []
    original_measure = monitor.measure
    original_publish = publisher.publish

    def measure(*args, **kwargs):
        events.append("measure")
        return original_measure(*args, **kwargs)

    def publish(*args, **kwargs):
        events.append("publish")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(monitor, "measure", measure)
    monkeypatch.setattr(publisher, "publish", publish)

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
    assert events == ["measure", "publish"]
    _, published_version = publisher.current()
    assert published_version == outcome.stats.version
    published_actor, _ = publisher.current()
    assert all(
        torch.equal(published_parameter, training_parameter)
        for published_parameter, training_parameter in zip(
            published_actor.parameters(), updater.actor.parameters(), strict=True
        )
    )


def test_inactive_episode_rejects_without_measuring_or_publishing(monkeypatch):
    updater, publisher, monitor = make_components()
    publish_calls = []

    def fail_if_measured(*args, **kwargs):
        raise AssertionError("stale candidate must not be measured")

    monkeypatch.setattr(monitor, "measure", fail_if_measured)
    monkeypatch.setattr(publisher, "publish", lambda *args, **kwargs: publish_calls.append(args))

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
    assert publish_calls == []


def test_episode_becoming_stale_at_final_publish_restores_complete_update_state(monkeypatch):
    updater, publisher, monitor = make_components()
    before = copy.deepcopy(updater.state_dict())
    publish_calls = []
    final_attempts = []

    monkeypatch.setattr(publisher, "publish", lambda *args: publish_calls.append(args))

    def reject_final_publish(candidate_actor, candidate_version):
        final_attempts.append((candidate_actor, candidate_version))
        return False

    outcome = update_and_publish_actor(
        updater=updater,
        publisher=publisher,
        drift_monitor=monitor,
        critic_batch=make_batch(0.1),
        actor_batch=make_batch(0.2),
        adaptation_episode=1,
        update_step=7,
        episode_is_active=lambda: True,
        final_publish_attempt=reject_final_publish,
    )

    assert len(final_attempts) == 1
    assert publish_calls == []
    assert not outcome.stats.accepted
    assert not outcome.stats.critic_updated
    assert not outcome.stats.actor_updated
    assert outcome.stats.rejection_reason == "stale_adaptation_episode"
    assert outcome.stats.version == before["version"]
    assert outcome.policy_drift is None
    assert_state_dict_equal(updater.state_dict(), before)
    _, published_version = publisher.current()
    assert published_version == before["version"]


def test_monitor_floating_point_error_rejects_without_publishing(monkeypatch):
    updater, publisher, monitor = make_components()
    publish_calls = []

    def raise_nonfinite(*args, **kwargs):
        raise FloatingPointError("non-finite policy drift")

    monkeypatch.setattr(monitor, "measure", raise_nonfinite)
    monkeypatch.setattr(publisher, "publish", lambda *args, **kwargs: publish_calls.append(args))
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
    assert publish_calls == []


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


def test_publish_failure_restores_updater_and_resynchronizes_publisher(monkeypatch):
    updater, publisher, monitor = make_components()
    before = copy.deepcopy(updater.state_dict())
    published_before, published_version_before = publisher.current()
    published_parameters_before = [parameter.detach().clone() for parameter in published_before.parameters()]
    original_publish = publisher.publish
    publish_calls = 0

    def fail_once(training_actor, version):
        nonlocal publish_calls
        publish_calls += 1
        original_publish(training_actor, version)
        if publish_calls == 1:
            raise RuntimeError("publish failed after swap")

    monkeypatch.setattr(publisher, "publish", fail_once)
    with pytest.raises(RuntimeError, match="publish failed after swap"):
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

    assert publish_calls == 2
    assert_state_dict_equal(updater.state_dict(), before)
    restored_published, restored_version = publisher.current()
    assert restored_version == published_version_before
    assert all(
        torch.equal(restored_parameter, before_parameter)
        for restored_parameter, before_parameter in zip(
            restored_published.parameters(), published_parameters_before, strict=True
        )
    )
