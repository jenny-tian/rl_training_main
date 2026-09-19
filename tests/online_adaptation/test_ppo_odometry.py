import math

import pytest
import torch

from rl_training.online_adaptation.ppo_odometry import PlanarProprioOdometry


def assert_snapshot_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def seeded_odometry():
    odometry = PlanarProprioOdometry(2, "cpu")
    odometry.observe(torch.tensor([[0.3, 0.1], [-0.2, 0.4]]), torch.tensor([[0.1], [-0.2]]), step_dt=0.2)
    return odometry


def test_initial_context_and_straight_motion_use_the_spawn_frame():
    odometry = PlanarProprioOdometry(2, "cpu", dtype=torch.float64)
    assert torch.equal(odometry.context(), torch.zeros(2, 3, dtype=torch.float64))
    velocity = torch.tensor([[0.3, 0.0], [-0.2, 0.1]], dtype=torch.float64)
    angular_velocity = torch.zeros(2, 1, dtype=torch.float64)
    for _ in range(20):
        context = odometry.observe(velocity, angular_velocity, step_dt=0.1)

    torch.testing.assert_close(odometry.snapshot()["relative_xy"], 2.0 * velocity, rtol=0, atol=1e-14)
    torch.testing.assert_close(odometry.snapshot()["heading"], angular_velocity, rtol=0, atol=0)
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2 / 1.65]], dtype=torch.float64)
    torch.testing.assert_close(context, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("angular_velocity", [0.4, -0.9])
def test_constant_turn_matches_analytic_planar_trajectory(angular_velocity):
    odometry = PlanarProprioOdometry(1, "cpu", dtype=torch.float64)
    vx, vy, dt, steps = 0.7, 0.15, 0.01, 200
    velocity = torch.tensor([[vx, vy]], dtype=torch.float64)
    omega = torch.tensor([[angular_velocity]], dtype=torch.float64)
    for _ in range(steps):
        odometry.observe(velocity, omega, step_dt=dt)
    duration, theta = steps * dt, steps * dt * angular_velocity
    expected = torch.tensor([[
        (vx * math.sin(theta) + vy * (math.cos(theta) - 1.0)) / angular_velocity,
        (vx * (1.0 - math.cos(theta)) + vy * math.sin(theta)) / angular_velocity,
    ]], dtype=torch.float64)
    bound = duration * math.hypot(vx, vy) * angular_velocity**2 * dt**2 / 24.0

    assert torch.linalg.vector_norm(odometry.snapshot()["relative_xy"] - expected).item() <= bound + 1e-12
    assert odometry.snapshot()["heading"].item() == pytest.approx(theta, abs=1e-12)


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.02])
def test_midpoint_global_error_respects_quadrature_bound(dt):
    speed, omega, duration = 0.6, 1.2, 2.0
    odometry = PlanarProprioOdometry(1, "cpu", dtype=torch.float64)
    for _ in range(round(duration / dt)):
        odometry.observe(torch.tensor([[speed, 0.0]], dtype=torch.float64),
                         torch.tensor([[omega]], dtype=torch.float64), step_dt=dt)
    theta = omega * duration
    exact = torch.tensor([[speed * math.sin(theta) / omega, speed * (1.0 - math.cos(theta)) / omega]],
                         dtype=torch.float64)
    # Composite midpoint quadrature: ||error|| <= T * h^2 * sup||f''|| / 24.
    bound = duration * dt**2 * speed * omega**2 / 24.0
    error = torch.linalg.vector_norm(odometry.snapshot()["relative_xy"] - exact).item()
    assert 0.0 < error <= bound + 1e-12


def test_heading_wraps_and_turns_body_forward_velocity_into_relative_lateral_motion():
    odometry = PlanarProprioOdometry(1, "cpu", dtype=torch.float64)
    odometry.observe(torch.zeros(1, 2, dtype=torch.float64),
                     torch.tensor([[4.5 * math.pi]], dtype=torch.float64), step_dt=1.0)
    context = odometry.observe(torch.tensor([[1.0, 0.0]], dtype=torch.float64),
                               torch.zeros(1, 1, dtype=torch.float64), step_dt=0.5)

    assert -math.pi <= odometry.snapshot()["heading"].item() <= math.pi
    assert odometry.snapshot()["heading"].item() == pytest.approx(math.pi / 2.0, abs=1e-12)
    torch.testing.assert_close(odometry.snapshot()["relative_xy"], torch.tensor([[0.0, 0.5]], dtype=torch.float64),
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(context, torch.tensor([[1.0, -1.0, 0.5 / 1.65]], dtype=torch.float64),
                               rtol=0, atol=1e-12)


def test_lateral_context_is_bounded_but_snapshot_retains_full_displacement():
    odometry = PlanarProprioOdometry(2, "cpu", lateral_scale=1.0)
    context = odometry.observe(torch.tensor([[0.0, 4.0], [0.0, -5.0]]), torch.zeros(2, 1), step_dt=1.0)

    torch.testing.assert_close(context[:, 2], torch.tensor([2.0, -2.0]), rtol=0, atol=0)
    torch.testing.assert_close(odometry.snapshot()["relative_xy"][:, 1], torch.tensor([4.0, -5.0]), rtol=0, atol=0)


@pytest.mark.parametrize("mask", [torch.tensor([True, False]), torch.tensor([[1.0], [0.0]])])
def test_reset_only_clears_selected_environments(mask):
    odometry = seeded_odometry()
    before, mask_before = odometry.snapshot(), mask.clone()

    odometry.reset(mask)

    snapshot = odometry.snapshot()
    for key in snapshot:
        torch.testing.assert_close(snapshot[key][0], torch.zeros_like(snapshot[key][0]), rtol=0, atol=0)
        torch.testing.assert_close(snapshot[key][1], before[key][1], rtol=0, atol=0)
    torch.testing.assert_close(mask, mask_before, rtol=0, atol=0)
    odometry.reset()
    assert torch.equal(odometry.context(), torch.zeros(2, 3))
    assert torch.equal(odometry.snapshot()["relative_xy"], torch.zeros(2, 2))


def test_observe_reset_mask_returns_new_episode_context_after_integrating_other_environments():
    odometry = PlanarProprioOdometry(2, "cpu")
    velocity, omega = torch.tensor([[1.0, 0.0], [2.0, 0.0]]), torch.zeros(2, 1)
    odometry.observe(velocity, omega, step_dt=1.0)
    context = odometry.observe(velocity, omega, step_dt=1.0, reset_mask=torch.tensor([True, False]))

    torch.testing.assert_close(odometry.snapshot()["relative_xy"], torch.tensor([[0.0, 0.0], [4.0, 0.0]]),
                               rtol=0, atol=0)
    assert torch.equal(context[0], torch.zeros(3))
    odometry.observe(velocity, omega, step_dt=1.0)
    torch.testing.assert_close(odometry.snapshot()["relative_xy"][0], torch.tensor([1.0, 0.0]), rtol=0, atol=0)


def test_inputs_snapshots_and_contexts_never_alias_mutable_state_or_build_autograd_history():
    odometry = PlanarProprioOdometry(2, "cpu")
    velocity = torch.tensor([[0.2, 0.1], [-0.1, 0.3]], requires_grad=True)
    omega = torch.tensor([[0.3], [-0.4]], requires_grad=True)
    velocity_before, omega_before = velocity.detach().clone(), omega.detach().clone()
    context = odometry.observe(velocity, omega, step_dt=0.1)
    before = odometry.snapshot()
    owned_snapshot = odometry.snapshot()
    owned_snapshot["heading"].fill_(99.0)
    owned_snapshot["relative_xy"].fill_(99.0)
    context.fill_(99.0)
    assert_snapshot_equal(odometry.snapshot(), before)
    previous_context = odometry.context()
    expected_context = previous_context.clone()
    saved_before = {key: value.clone() for key, value in before.items()}

    odometry.observe(velocity, omega, step_dt=0.1)

    assert_snapshot_equal(before, saved_before)
    torch.testing.assert_close(previous_context, expected_context, rtol=0, atol=0)
    torch.testing.assert_close(velocity, velocity_before, rtol=0, atol=0)
    torch.testing.assert_close(omega, omega_before, rtol=0, atol=0)
    assert not odometry.context().requires_grad
    assert all(not value.requires_grad for value in odometry.snapshot().values())


@pytest.mark.parametrize("name,value", [
    ("body_lin_vel_xy", torch.zeros(2, 3)),
    ("body_lin_vel_xy", torch.zeros(1, 2)),
    ("body_lin_vel_xy", torch.tensor([[0.0, 0.0], [float("nan"), 0.0]])),
    ("body_lin_vel_xy", torch.full((2, 2), float("inf"))),
    ("body_lin_vel_xy", torch.zeros(2, 2, dtype=torch.int64)),
    ("body_lin_vel_xy", torch.zeros(2, 2, dtype=torch.complex64)),
    ("body_lin_vel_xy", torch.zeros(2, 2, device="meta")),
    ("body_lin_vel_xy", torch.full((2, 2), 1e100, dtype=torch.float64)),
    ("body_ang_vel_z", torch.zeros(2)),
    ("body_ang_vel_z", torch.zeros(1, 1)),
    ("body_ang_vel_z", torch.tensor([[0.0], [float("nan")]])),
    ("body_ang_vel_z", torch.full((2, 1), float("inf"))),
    ("body_ang_vel_z", torch.zeros(2, 1, dtype=torch.int64)),
    ("body_ang_vel_z", torch.zeros(2, 1, device="meta")),
])
def test_bad_measurement_cannot_partially_update_any_environment(name, value):
    odometry = seeded_odometry()
    before = odometry.snapshot()
    arguments = {"body_lin_vel_xy": torch.ones(2, 2), "body_ang_vel_z": torch.ones(2, 1), name: value}

    with pytest.raises(ValueError, match=name):
        odometry.observe(**arguments, step_dt=0.1)

    assert_snapshot_equal(odometry.snapshot(), before)


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf"), True, 1e100, 1e-100])
def test_invalid_step_duration_preserves_state(dt):
    odometry = seeded_odometry()
    before = odometry.snapshot()
    with pytest.raises(ValueError, match="step_dt"):
        odometry.observe(torch.ones(2, 2), torch.ones(2, 1), step_dt=dt)
    assert_snapshot_equal(odometry.snapshot(), before)


@pytest.mark.parametrize("mask", [
    torch.zeros(2, 2), torch.zeros(1), torch.tensor([0, 2]), torch.tensor([0.0, float("nan")]),
    torch.tensor([0.0, 0.5]), torch.zeros(2, dtype=torch.complex64), torch.zeros(2, device="meta"), [True, False],
])
def test_invalid_reset_mask_preserves_state_for_both_apis(mask):
    odometry = seeded_odometry()
    before = odometry.snapshot()
    with pytest.raises(ValueError, match="reset mask"):
        odometry.observe(torch.ones(2, 2), torch.ones(2, 1), step_dt=0.1, reset_mask=mask)
    assert_snapshot_equal(odometry.snapshot(), before)
    with pytest.raises(ValueError, match="reset mask"):
        odometry.reset(mask)
    assert_snapshot_equal(odometry.snapshot(), before)


@pytest.mark.parametrize("component", ["velocity", "angular_velocity"])
def test_finite_measurements_that_overflow_integration_do_not_change_state(component):
    odometry = seeded_odometry()
    before = odometry.snapshot()
    velocity, omega = torch.ones(2, 2), torch.zeros(2, 1)
    if component == "velocity":
        velocity[1, 0] = torch.finfo(torch.float32).max
    else:
        omega[1, 0] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="non-finite state"):
        odometry.observe(velocity, omega, step_dt=2.0, reset_mask=torch.tensor([False, True]))
    assert_snapshot_equal(odometry.snapshot(), before)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_size_must_be_positive_integer(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        PlanarProprioOdometry(batch_size, "cpu")


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), True, 1e100, 1e-100])
def test_lateral_scale_must_be_positive_finite_and_representable(scale):
    with pytest.raises(ValueError, match="lateral_scale"):
        PlanarProprioOdometry(1, "cpu", lateral_scale=scale)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64, "float32"])
def test_non_floating_state_dtype_is_rejected(dtype):
    with pytest.raises(ValueError, match="dtype"):
        PlanarProprioOdometry(1, "cpu", dtype=dtype)
