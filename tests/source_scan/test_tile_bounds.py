from __future__ import annotations

import torch

import pytest

from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated

SIZE = (12.0, 4.0)
FLAT_LENGTH_M = 3.0
COLLISION_RADIUS_M = 0.27
NUMERICAL_MARGIN_M = 0.001
CONTROL_PERIOD_S = 0.02


def _guard(positions, velocities, origins, levels, types):
    return source_tile_bounds_terminated(
        positions,
        velocities,
        origins,
        levels,
        types,
        size=SIZE,
        flat_length_m=FLAT_LENGTH_M,
        collision_radius_m=COLLISION_RADIUS_M,
        numerical_margin_m=NUMERICAL_MARGIN_M,
        control_period_s=CONTROL_PERIOD_S,
    )


def _origin():
    return torch.tensor([[[100.0, 200.0, 0.0]]])


@pytest.mark.parametrize(
    "point",
    [
        (100.0 - 1.5 + COLLISION_RADIUS_M + NUMERICAL_MARGIN_M - 0.001, 200.0),
        (100.0 + 10.5 - COLLISION_RADIUS_M - NUMERICAL_MARGIN_M + 0.001, 200.0),
        (100.0, 200.0 - 2.0 + COLLISION_RADIUS_M + NUMERICAL_MARGIN_M - 0.001),
        (100.0, 200.0 + 2.0 - COLLISION_RADIUS_M - NUMERICAL_MARGIN_M + 0.001),
    ],
    ids=("x_min", "x_max", "y_min", "y_max"),
)
def test_each_tile_edge_terminates_before_the_collision_envelope_crosses(point):
    positions = torch.tensor([[[point[0], point[1], 0.4]]])
    velocities = torch.zeros_like(positions)

    assert _guard(positions, velocities, _origin(), torch.tensor([0]), torch.tensor([0])).tolist() == [True]


def test_selects_the_owning_row_and_column_for_each_environment():
    origins = torch.zeros((2, 3, 3))
    origins[1, 2] = torch.tensor([112.0, 208.0, 0.0])
    origins[0, 1] = torch.tensor([100.0, 204.0, 0.0])
    positions = torch.tensor(
        [
            [[112.0 + 9.3, 208.0, 0.4]],
            [[100.0 + 9.3, 204.0, 0.4]],
        ]
    )
    velocities = torch.zeros_like(positions)

    result = _guard(
        positions,
        velocities,
        origins,
        torch.tensor([1, 0]),
        torch.tensor([2, 1]),
    )

    assert result.tolist() == [False, False]


def test_completion_plus_nominal_hold_remains_inside_the_candidate_guard():
    # The registered completion is 9.0 m from reset; the 50-step hold at
    # 0.3 m/s adds 0.3 m, leaving 1.2 m to the local track end.
    positions = torch.tensor([[[100.0 + 9.3, 200.0, 0.4]]])
    velocities = torch.tensor([[[0.3, 0.0, 0.0]]])

    assert _guard(positions, velocities, _origin(), torch.tensor([0]), torch.tensor([0])).tolist() == [False]


def test_completion_plus_hold_allows_the_conservative_rigid_body_reach():
    # Lite3 joint-origin lengths bound any body-frame origin to < 0.693 m from
    # the root. The guard's 0.271 m inset includes the collision radius and
    # numerical margin, leaving 0.236 m at the worst x extent after the hold.
    body_reach_m = 0.693
    positions = torch.tensor(
        [[
            [100.0 + 9.3, 200.0, 0.4],
            [100.0 + 9.3 + body_reach_m, 200.0, 0.4],
            [100.0 + 9.3 - body_reach_m, 200.0, 0.4],
            [100.0 + 9.3, 200.0 + body_reach_m, 0.4],
            [100.0 + 9.3, 200.0 - body_reach_m, 0.4],
        ]]
    )
    velocities = torch.zeros_like(positions)

    assert _guard(
        positions, velocities, _origin(), torch.tensor([0]), torch.tensor([0])
    ).tolist() == [False]


def test_lookahead_terminates_an_outward_motion_before_the_next_check_crosses():
    safe_upper_x = 100.0 + 10.5 - COLLISION_RADIUS_M - NUMERICAL_MARGIN_M
    positions = torch.tensor([[[safe_upper_x - 0.05, 200.0, 0.4]]])
    velocities = torch.tensor([[[3.0, 0.0, 0.0]]])

    result = _guard(positions, velocities, _origin(), torch.tensor([0]), torch.tensor([0]))

    assert result.tolist() == [True]


def test_termination_is_per_environment_and_does_not_mutate_unreset_state():
    origins = torch.tensor(
        [
            [[100.0, 200.0, 0.0]],
            [[112.0, 200.0, 0.0]],
        ]
    )
    positions = torch.tensor(
        [
            [[100.0 + 9.3, 200.0, 0.4]],
            [[112.0 + 10.4, 200.0, 0.4]],
        ]
    )
    velocities = torch.zeros_like(positions)
    snapshot = (origins.clone(), positions.clone(), velocities.clone())

    result = _guard(
        positions,
        velocities,
        origins,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )

    assert result.tolist() == [False, True]
    assert torch.equal(origins, snapshot[0])
    assert torch.equal(positions, snapshot[1])
    assert torch.equal(velocities, snapshot[2])


def test_nonfinite_body_state_fails_closed_for_affected_environment_only():
    positions = torch.tensor(
        [
            [[100.0 + 9.3, 200.0, 0.4]],
            [[float("nan"), 200.0, 0.4]],
        ]
    )
    velocities = torch.zeros_like(positions)
    origins = torch.tensor([[[100.0, 200.0, 0.0]], [[112.0, 200.0, 0.0]]])

    result = _guard(
        positions,
        velocities,
        origins,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )

    assert result.tolist() == [False, True]


def test_invalid_terrain_owner_fails_closed_without_indexing_another_tile():
    positions = torch.tensor([[[100.0 + 9.3, 200.0, 0.4]]])
    velocities = torch.zeros_like(positions)

    result = _guard(positions, velocities, _origin(), torch.tensor([4]), torch.tensor([0]))

    assert result.tolist() == [True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": (12.0, 0.0)},
        {"flat_length_m": 13.0},
        {"collision_radius_m": 0.0},
        {"numerical_margin_m": -0.1},
        {"control_period_s": 0.0},
    ],
)
def test_invalid_guard_configuration_is_rejected(kwargs):
    base = {
        "size": SIZE,
        "flat_length_m": FLAT_LENGTH_M,
        "collision_radius_m": COLLISION_RADIUS_M,
        "numerical_margin_m": NUMERICAL_MARGIN_M,
        "control_period_s": CONTROL_PERIOD_S,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        source_tile_bounds_terminated(
            torch.tensor([[[100.0, 200.0, 0.4]]]),
            torch.zeros((1, 1, 3)),
            _origin(),
            torch.tensor([0]),
            torch.tensor([0]),
            **base,
        )
