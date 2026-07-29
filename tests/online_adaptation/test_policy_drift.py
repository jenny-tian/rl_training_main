import pytest
import torch

from rl_training.online_adaptation.networks import ResidualActor
from rl_training.online_adaptation.policy_drift import ActorDriftMonitor
from rl_training.online_adaptation.types import TransitionBatch


def make_actor():
    return ResidualActor(
        frame_dim=2,
        action_dim=1,
        history_dim=4,
        action_limits=torch.tensor([0.2]),
    )


def make_batch(value=0.0, dtype=torch.float32):
    return TransitionBatch(
        history=torch.full((4, 3, 2), value, dtype=dtype),
        current_frame=torch.full((4, 2), value, dtype=dtype),
        base_action=torch.full((4, 1), value, dtype=dtype),
        residual_action=torch.zeros(4, 1, dtype=dtype),
        reward=torch.zeros(4, 1, dtype=dtype),
        next_history=torch.zeros(4, 3, 2, dtype=dtype),
        next_frame=torch.zeros(4, 2, dtype=dtype),
        next_base_action=torch.zeros(4, 1, dtype=dtype),
        done=torch.zeros(4, 1, dtype=dtype),
    )


def test_identical_actor_has_zero_policy_drift_and_call_metadata():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=3)

    drift = monitor.measure(
        actor,
        make_batch(),
        actor_version=7,
        update_step=100,
        adaptation_episode=3,
    )

    assert drift.residual_mse == 0.0
    assert drift.joint_abs_change_mean == 0.0
    assert drift.joint_abs_change_max == 0.0
    assert drift.parameter_l2 == 0.0
    assert drift.adaptation_episode == 3
    assert drift.actor_version == 7
    assert drift.update_step == 100


def test_known_mean_layer_bias_change_has_positive_finite_drift():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.add_(0.1)

    drift = monitor.measure(
        actor,
        make_batch(),
        actor_version=1,
        update_step=20,
        adaptation_episode=1,
    )

    values = (
        drift.residual_mse,
        drift.joint_abs_change_mean,
        drift.joint_abs_change_max,
        drift.parameter_l2,
    )
    assert all(value > 0.0 for value in values)
    assert all(torch.isfinite(torch.tensor(value)) for value in values)


def test_float64_parameter_l2_stays_finite_for_large_finite_delta():
    actor = make_actor().double()
    batch = make_batch(dtype=torch.float64)
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.fill_(1e20)

    with torch.inference_mode():
        candidate_action, _, _ = actor.sample(
            batch.history, batch.current_frame, batch.base_action, deterministic=True
        )
    assert torch.isfinite(candidate_action).all()

    drift = monitor.measure(
        actor,
        batch,
        actor_version=1,
        update_step=20,
        adaptation_episode=1,
    )

    assert torch.isfinite(torch.tensor(drift.parameter_l2))
    assert drift.parameter_l2 > 1e19


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_nonfinite_candidate_parameter_is_rejected(invalid_value):
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.fill_(invalid_value)

    with pytest.raises(FloatingPointError, match="non-finite policy drift"):
        monitor.measure(
            actor,
            make_batch(),
            actor_version=1,
            update_step=20,
            adaptation_episode=1,
        )


def test_monitor_reuses_first_diagnostic_inputs_within_episode():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.add_(0.1)

    first = monitor.measure(
        actor,
        make_batch(0.0),
        actor_version=1,
        update_step=20,
        adaptation_episode=1,
    )
    second = monitor.measure(
        actor,
        make_batch(100.0),
        actor_version=2,
        update_step=30,
        adaptation_episode=1,
    )

    assert second.residual_mse == first.residual_mse
    assert second.joint_abs_change_mean == first.joint_abs_change_mean
    assert second.joint_abs_change_max == first.joint_abs_change_max


def test_new_episode_refreshes_reference_and_diagnostic_inputs():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=1)
    with torch.no_grad():
        actor.mean_layer.bias.add_(0.1)
    monitor.measure(
        actor,
        make_batch(),
        actor_version=1,
        update_step=20,
        adaptation_episode=1,
    )

    monitor.start_episode(actor, adaptation_episode=2)
    drift = monitor.measure(
        actor,
        make_batch(2.0),
        actor_version=1,
        update_step=40,
        adaptation_episode=2,
    )

    assert monitor.adaptation_episode == 2
    assert drift.residual_mse == 0.0
    assert drift.joint_abs_change_mean == 0.0
    assert drift.joint_abs_change_max == 0.0
    assert drift.parameter_l2 == 0.0


def test_old_episode_clear_does_not_remove_new_reference():
    actor = make_actor()
    monitor = ActorDriftMonitor()
    monitor.start_episode(actor, adaptation_episode=2)

    monitor.clear_episode(adaptation_episode=1)

    drift = monitor.measure(
        actor,
        make_batch(),
        actor_version=1,
        update_step=40,
        adaptation_episode=2,
    )
    assert monitor.adaptation_episode == 2
    assert drift.residual_mse == 0.0


def test_start_episode_rejects_nonpositive_episode():
    actor = make_actor()
    monitor = ActorDriftMonitor()

    for adaptation_episode in (0, -1):
        with pytest.raises(ValueError, match="must be positive"):
            monitor.start_episode(actor, adaptation_episode=adaptation_episode)

    assert monitor.adaptation_episode == 0


def test_measure_rejects_missing_or_mismatched_reference():
    actor = make_actor()
    monitor = ActorDriftMonitor()

    with pytest.raises(RuntimeError, match="no reference actor"):
        monitor.measure(
            actor,
            make_batch(),
            actor_version=1,
            update_step=10,
            adaptation_episode=1,
        )

    monitor.start_episode(actor, adaptation_episode=2)
    with pytest.raises(RuntimeError, match="no reference actor"):
        monitor.measure(
            actor,
            make_batch(),
            actor_version=1,
            update_step=20,
            adaptation_episode=1,
        )
