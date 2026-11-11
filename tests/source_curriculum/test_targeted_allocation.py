"""Weighted sampling retains geometry, origins and distinct pre-move outcomes."""
from types import SimpleNamespace

import pytest
import torch

from rl_training.source_curriculum.allocation import apply_targeted_origins, targeted_column_counts
from rl_training.source_curriculum.logic import ColumnStats


@pytest.mark.parametrize("size", [120, 2048, 4096])
def test_deterministic_counts_preserve_every_column(size):
    counts = targeted_column_counts(size)
    assert counts == targeted_column_counts(size)
    assert sum(counts) == size and len(counts) == 20 and min(counts) > 0
    assert abs(sum(counts[9:13]) - .3 * size) <= 1
    assert abs(counts[5] - .2 * size) <= 1
    others = [v for i, v in enumerate(counts) if i not in (5, 9, 10, 11, 12)]
    assert max(others) - min(others) <= 1
    if size == 4096:
        assert sum(counts[9:13]) == 1229 and counts[5] == 819 and sum(others) == 2048


@pytest.mark.parametrize("size", [True, 0, 20, 4096.0])
def test_invalid_or_column_starving_count_rejected(size):
    with pytest.raises(ValueError):
        targeted_column_counts(size)


def test_assignment_changes_origins_without_mutating_grid_or_levels():
    origins = torch.arange(600, dtype=torch.float).reshape(10, 20, 3)
    original = origins.clone()
    levels = torch.arange(120) % 5
    terrain = SimpleNamespace(device="cpu", terrain_levels=levels.clone())
    assigned = apply_targeted_origins(terrain, 120, origins)
    assert torch.equal(origins, original) and torch.equal(terrain.terrain_levels, levels)
    assert torch.bincount(terrain.terrain_types, minlength=20).tolist() == targeted_column_counts(120)
    assert torch.equal(assigned, origins[levels, terrain.terrain_types])
    with pytest.raises(ValueError, match="10 x 20"):
        apply_targeted_origins(terrain, 120, origins[:8])


def test_column_counts_keep_stairs_and_rows_separate_with_failure_priority():
    stats = ColumnStats("cpu")
    columns, rows = torch.tensor([4, 5, 5, 5, 5]), torch.tensor([9, 9, 4, 4, 0])
    stats.record_starts(columns, rows)
    # The last row is an initial reset with no outcome. Simultaneous ending at
    # L2 on the descending column must count only as failure.
    stats.update(columns, rows, torch.tensor([1, 1, 1, 0, 0]).bool(),
                 torch.tensor([0, 1, 0, 0, 0]).bool(), torch.tensor([1, 1, 0, 1, 0]).bool())
    rows[:] = 0  # later curriculum moves must not relabel already counted outcomes
    report = stats.report()
    assert report["columns/stairs_ascending/L2/completion_count"] == 1
    assert report["columns/stairs_descending/L2/failure_count"] == 1
    assert report["columns/stairs_descending/L2/completion_count"] == 0
    assert report["columns/stairs_descending/L1/completion_rate"] == .5
    assert report["columns/stairs_descending/L1/timeout_count"] == 1
    assert report["columns/stairs_descending/all/starts"] == 4
    assert report["columns/stairs_descending/all/endings"] == 3
    assert stats.snapshot()["endings_by_column_row_outcome"][5][9] == [0, 1, 0]


def test_empty_resets_do_not_add_samples():
    stats = ColumnStats("cpu")
    ids, flags = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)
    stats.record_starts(ids, ids)
    stats.update(ids, ids, flags, flags, flags)
    assert stats.starts.sum() == stats.outcomes.sum() == 0
