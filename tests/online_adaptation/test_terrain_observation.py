import pytest
import torch

from rl_training.online_adaptation.terrain_observation import (
    HeightScanConfig,
    summarize_height_scan,
)


def test_summary_has_stable_order_and_directional_statistics():
    heights = torch.tensor(
        [
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ]
        ]
    )
    config = HeightScanConfig(
        rows=4,
        cols=4,
        height_scale_m=0.1,
        clip_m=0.2,
        min_valid_fraction=1.0,
    )

    features, valid = summarize_height_scan(heights, config)

    expected_grid = torch.tensor(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    expected = torch.tensor(
        [
            1.0,
            0.0,
            2.0,
            expected_grid.mean(),
            expected_grid.std(unbiased=False),
            expected_grid[:, 2:].min(),
            expected_grid[:, 2:].mean(),
            expected_grid[:, 2:].max(),
            expected_grid[2:, :].mean() - expected_grid[:2, :].mean(),
            expected_grid[:, 2:].mean() - expected_grid[:, :2].mean(),
        ]
    )
    torch.testing.assert_close(features, expected.unsqueeze(0))
    torch.testing.assert_close(valid, torch.ones(1, 1, dtype=torch.bool))


def test_non_square_scan_uses_lateral_rows_and_forward_columns():
    heights = torch.zeros(1, 4, 6)
    heights[:, :, 3:] = 0.1
    heights[:, 2:, :] += 0.05

    features, valid = summarize_height_scan(
        heights,
        HeightScanConfig(rows=4, cols=6, height_scale_m=0.1, clip_m=0.2),
    )

    assert valid.item()
    assert features[0, 6].item() == pytest.approx(1.25)
    assert features[0, 8].item() == pytest.approx(0.5)
    assert features[0, 9].item() == pytest.approx(1.0)


def test_nonfinite_hits_are_zeroed_but_validity_uses_finite_fraction():
    heights = torch.tensor(
        [[[float("nan"), 0.1], [0.2, 0.3]]], dtype=torch.float32
    )
    features, valid = summarize_height_scan(
        heights,
        HeightScanConfig(rows=2, cols=2, min_valid_fraction=0.75),
    )

    assert torch.isfinite(features).all()
    assert valid.item() is True
    assert features[0, 0].item() == pytest.approx(0.75)


def test_all_invalid_hits_fail_closed_with_finite_zero_features():
    heights = torch.full((2, 2, 2), float("nan"))

    features, valid = summarize_height_scan(
        heights,
        HeightScanConfig(rows=2, cols=2, min_valid_fraction=0.5),
    )

    torch.testing.assert_close(features, torch.zeros(2, 10))
    torch.testing.assert_close(valid, torch.zeros(2, 1, dtype=torch.bool))


@pytest.mark.parametrize(
    "overrides",
    [
        {"rows": 0},
        {"cols": 0},
        {"height_scale_m": 0.0},
        {"clip_m": -0.1},
        {"min_valid_fraction": 1.1},
    ],
)
def test_height_scan_config_rejects_invalid_values(overrides):
    values = {
        "rows": 2,
        "cols": 2,
        "height_scale_m": 0.1,
        "clip_m": 0.2,
        "min_valid_fraction": 0.9,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        HeightScanConfig(**values)


def test_height_scan_rejects_wrong_grid_shape():
    with pytest.raises(ValueError, match="grid shape"):
        summarize_height_scan(
            torch.zeros(1, 2, 3),
            HeightScanConfig(rows=2, cols=2),
        )
