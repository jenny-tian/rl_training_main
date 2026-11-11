import torch
import pytest

from rl_training.source_curriculum.allocation import targeted_column_counts
from rl_training.source_curriculum.block import BlockStats, axis_vectors, block_axis, block_completed, block_progress, replica_rows


@pytest.mark.parametrize("n", [120, 4096])
def test_replicas_cover_all_rows_for_large_columns_and_balance_small_columns(n):
    counts = torch.tensor(targeted_column_counts(n))
    columns = torch.repeat_interleave(torch.arange(20), counts)
    rows = replica_rows(columns)
    assert rows.min() == 0 and rows.max() == 9
    for c in range(20):
        hist = torch.bincount(rows[columns == c], minlength=10)
        assert hist.max() - hist.min() <= 1


def test_stairs_cycle_four_exits_while_slopes_follow_geometry_rotation():
    ids = torch.arange(80)
    columns, rows = ids % 20, ids % 10
    stair = (columns == 4) | (columns == 5)
    axes = torch.stack([block_axis(rows, columns, torch.full_like(ids, k), ids) for k in range(4)])
    assert torch.equal(axes[:, ~stair], (rows[~stair] % 4).expand(4, -1))
    assert torch.equal(axes[:, stair].sort(dim=0).values, torch.arange(4)[:, None].expand(-1, int(stair.sum())))


def test_completion_requires_all_feet_past_edge_in_every_world_direction():
    axes = torch.arange(4)
    d = axis_vectors(axes)
    origins = torch.tensor([[100., -80.]]).expand(4, -1)
    roots = origins + d * 3.45
    feet = origins[:, None] + d[:, None] * torch.tensor([3.1, 3.2, 3.7, 3.6])[None, :, None]
    root, progress = block_progress(roots, feet, origins, axes)
    omni = torch.zeros(4, dtype=torch.bool)
    assert block_completed(root, progress, omni).all()
    progress[0, 0] = 2.99
    root[1] = 3.39
    progress[2, 0] = float("nan")
    omni[3] = True
    assert not block_completed(root, progress, omni).any()


def test_l2_statistics_count_all_replicas_and_failure_wins():
    stats = BlockStats("cpu")
    col, rows, axes = torch.tensor([4, 4, 5]), torch.tensor([0, 9, 3]), torch.tensor([0, 2, 1])
    stats.record_starts(col, rows, axes)
    stats.record_endings(col, rows, axes, torch.tensor([True, True, False]), torch.tensor([False, True, False]),
                         torch.tensor([False, True, True]), torch.tensor([5, 2, 1]), {"safety": torch.tensor([False, True, False])})
    assert stats.outcomes.sum().item() == 3
    assert stats.report()["blocks/stairs_ascending/completion"] == 1
    assert stats.report()["blocks/stairs_ascending/failure"] == 1
    assert stats.snapshot()["rows_are"] == "spatial_replicas"
