import math

import pytest

from rl_training.online_adaptation.novelty import NoveltyConfig, NoveltyGate, NoveltyState


def config(**overrides):
    values = {
        "calibration_steps": 3,
        "enter_threshold": 2.5,
        "exit_threshold": 1.0,
        "enter_count": 2,
        "enter_window": 3,
        "recovery_steps": 2,
        "cooldown_steps": 2,
        "contact_count": 2,
        "contact_window": 3,
        "velocity_scale_floor": 0.05,
        "tilt_scale_floor": 0.02,
        "reward_scale_floor": 0.10,
    }
    values.update(overrides)
    return NoveltyConfig(**values)


def observe(gate, *, velocity_error=0.0, tilt_rms=0.0, nonfoot_contact=False, reward=1.0, paused=False):
    return gate.observe(velocity_error, tilt_rms, nonfoot_contact, reward, paused=paused)


def calibrate(gate):
    return [observe(gate) for _ in range(gate.config.calibration_steps)]


def test_calibration_lasts_exactly_configured_number_of_valid_observations():
    gate = NoveltyGate(config())

    decisions = calibrate(gate)

    assert all(decision.state is NoveltyState.CALIBRATING for decision in decisions)
    assert observe(gate).state is NoveltyState.MONITORING


def test_persistent_continuous_novelty_enters_adapting_and_opens_actor_gate():
    gate = NoveltyGate(config())
    calibrate(gate)
    observe(gate)

    first = observe(gate, velocity_error=1.0)
    second = observe(gate, velocity_error=1.0)

    assert first.state is NoveltyState.MONITORING
    assert not first.entered_adapting
    assert second.state is NoveltyState.ADAPTING
    assert second.entered_adapting
    assert second.actor_update_enabled
    assert second.velocity_excess > gate.config.enter_threshold


def test_one_frame_continuous_spike_cannot_enter_adapting():
    gate = NoveltyGate(config())
    calibrate(gate)
    observe(gate)

    spike = observe(gate, velocity_error=1.0)
    settled = observe(gate)

    assert spike.state is NoveltyState.MONITORING
    assert settled.state is NoveltyState.MONITORING
    assert not settled.actor_update_enabled


def test_persistent_nonfoot_contacts_enter_adapting():
    gate = NoveltyGate(config())
    calibrate(gate)
    observe(gate)

    first = observe(gate, nonfoot_contact=True)
    second = observe(gate, nonfoot_contact=True)

    assert first.state is NoveltyState.MONITORING
    assert second.state is NoveltyState.ADAPTING
    assert second.entered_adapting
    assert second.actor_update_enabled


def test_recovery_leaves_adapting_then_cooldown_returns_to_monitoring():
    gate = NoveltyGate(config())
    calibrate(gate)
    observe(gate)
    observe(gate, velocity_error=1.0)
    entered = observe(gate, velocity_error=1.0)
    assert entered.state is NoveltyState.ADAPTING

    first_recovery = observe(gate)
    left = observe(gate)

    assert first_recovery.state is NoveltyState.ADAPTING
    assert left.state is NoveltyState.COOLDOWN
    assert left.left_adapting
    assert not left.actor_update_enabled
    assert observe(gate).state is NoveltyState.COOLDOWN
    assert observe(gate).state is NoveltyState.MONITORING


def test_force_cooldown_closes_actor_gate_and_resets_episode_counters():
    gate = NoveltyGate(config())
    calibrate(gate)
    observe(gate)
    observe(gate, velocity_error=1.0)
    assert observe(gate, velocity_error=1.0).actor_update_enabled

    forced = gate.force_cooldown()

    assert forced.state is NoveltyState.COOLDOWN
    assert forced.left_adapting
    assert not forced.actor_update_enabled
    assert observe(gate, velocity_error=1.0).state is NoveltyState.COOLDOWN


def test_force_cooldown_during_calibration_does_not_bypass_baseline_collection():
    gate = NoveltyGate(config(calibration_steps=3, cooldown_steps=1))
    observe(gate, velocity_error=0.10, tilt_rms=0.02, reward=1.0)

    forced = gate.force_cooldown()

    assert forced.state is NoveltyState.CALIBRATING
    assert not forced.actor_update_enabled
    assert observe(gate, velocity_error=0.10, tilt_rms=0.02, reward=1.0).state is NoveltyState.CALIBRATING
    assert observe(gate, velocity_error=0.10, tilt_rms=0.02, reward=1.0).state is NoveltyState.CALIBRATING
    assert observe(gate, velocity_error=0.10, tilt_rms=0.02, reward=1.0).state is NoveltyState.MONITORING


def test_calibration_baseline_is_unbiased_for_a_constant_signal():
    gate = NoveltyGate(config(calibration_steps=256))
    for _ in range(256):
        observe(gate, velocity_error=0.10, tilt_rms=0.02, reward=1.0)

    decision = observe(gate, velocity_error=0.10, tilt_rms=0.02, reward=1.0)

    assert decision.state is NoveltyState.MONITORING
    assert decision.novelty_score == pytest.approx(0.0, abs=1e-6)


def test_paused_observation_freezes_baseline_and_transition_counters():
    gate = NoveltyGate(config())
    calibrate(gate)
    observe(gate)
    first = observe(gate, velocity_error=1.0)

    paused = observe(gate, velocity_error=100.0, nonfoot_contact=True, reward=-100.0, paused=True)
    second = observe(gate, velocity_error=1.0)

    assert paused.state is NoveltyState.MONITORING
    assert paused.novelty_score == first.novelty_score
    assert second.state is NoveltyState.ADAPTING


def test_paused_observation_closes_actor_gate_without_leaving_adapting():
    gate = NoveltyGate(config())
    calibrate(gate)
    observe(gate)
    observe(gate, velocity_error=1.0)
    assert observe(gate, velocity_error=1.0).actor_update_enabled

    paused = observe(gate, paused=True)

    assert paused.state is NoveltyState.ADAPTING
    assert not paused.actor_update_enabled


@pytest.mark.parametrize(
    "overrides",
    [
        {"calibration_steps": 0},
        {"enter_count": 0},
        {"enter_window": 0},
        {"recovery_steps": 0},
        {"cooldown_steps": 0},
        {"contact_count": 0},
        {"contact_window": 0},
        {"velocity_scale_floor": 0.0},
        {"tilt_scale_floor": 0.0},
        {"reward_scale_floor": 0.0},
        {"enter_count": 4, "enter_window": 3},
        {"contact_count": 4, "contact_window": 3},
        {"exit_threshold": 2.5},
        {"enter_threshold": -0.1},
        {"exit_threshold": -0.1},
        {"enter_threshold": math.nan},
        {"exit_threshold": math.inf},
        {"velocity_scale_floor": math.nan},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        NoveltyConfig(**{**config().__dict__, **overrides})


@pytest.mark.parametrize(
    "field,value",
    [
        ("velocity_error", math.nan),
        ("velocity_error", math.inf),
        ("tilt_rms", math.nan),
        ("reward", -math.inf),
    ],
)
def test_nonfinite_observations_are_rejected(field, value):
    gate = NoveltyGate(config())
    values = {"velocity_error": 0.1, "tilt_rms": 0.02, "nonfoot_contact": False, "reward": 1.0}
    values[field] = value

    with pytest.raises(ValueError):
        gate.observe(**values)
