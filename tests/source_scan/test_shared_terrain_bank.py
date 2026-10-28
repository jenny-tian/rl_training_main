"""Offline scheduling evidence only; shared terrain is not wired into runtime."""

from collections import Counter
from dataclasses import replace
import random

import pytest

from rl_training.source_scan.parallel_terrain_allocation import (
    SOURCE_RESET_BLOCK_COUNTS,
    ReservedRowTerrainAllocator,
)
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator


@pytest.mark.parametrize("num_envs,bank_rows", [(1, 1), (60, 4), (120, 8), (4096, 8)])
def test_bank_preserves_episode_stream_and_unreset_environments(num_envs, bank_rows):
    old = ReservedRowTerrainAllocator(num_envs, SOURCE_RESET_BLOCK_COUNTS, 130001)
    new = SharedBankTerrainAllocator(num_envs, SOURCE_RESET_BLOCK_COUNTS, 130001, bank_rows=bank_rows)
    rng = random.Random(73001)
    history = []
    batches = [list(range(num_envs))] + [rng.sample(range(num_envs), min(num_envs, 7)) for _ in range(180)]
    for batch in batches:
        previous = new.assignments
        expected = old.reset(batch)
        actual = new.reset(batch)
        assert tuple(replace(a, row=a.env_id) for a in actual) == expected
        active = new.assignments
        assert all(active[i] == previous[i] for i in range(num_envs) if i not in batch)
        assert all(0 <= a.row < bank_rows and 0 <= a.column < 15 for a in actual)
        history.extend(a.subterrain for a in actual)
    for start in range(0, len(history) - 119, 120):
        assert Counter(history[start:start + 120]) == SOURCE_RESET_BLOCK_COUNTS
    assert new.grid_shape == (bank_rows, 15)
    if num_envs > bank_rows * 15:
        assert len({(a.row, a.column) for a in new.assignments}) < num_envs


@pytest.mark.parametrize("ids", [[0, 0], [-1], [2], [True], [0.0]])
def test_invalid_shared_reset_leaves_schedule_unchanged(ids):
    allocator = SharedBankTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, 130001, bank_rows=1)
    reference = SharedBankTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, 130001, bank_rows=1)
    assert allocator.reset([0, 1]) == reference.reset([0, 1])
    with pytest.raises(ValueError):
        allocator.reset(ids)
    assert allocator.assignments == reference.assignments
    assert allocator.reset([1]) == reference.reset([1])


@pytest.mark.parametrize("bank_rows", [0, -1, True, 1.5])
def test_bank_rejects_invalid_geometry_capacity(bank_rows):
    with pytest.raises(ValueError):
        SharedBankTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, 130001, bank_rows=bank_rows)


@pytest.mark.parametrize("target", tuple(SOURCE_RESET_BLOCK_COUNTS))
def test_asynchronous_lifetimes_can_fill_every_environment_with_one_class(target):
    # Keep target-class episodes alive; reset only other classes. A fixed pool
    # sized by average proportions cannot serve this valid asynchronous trace.
    count = 60
    allocator = ReservedRowTerrainAllocator(count, SOURCE_RESET_BLOCK_COUNTS, 130001)
    history = list(allocator.reset(range(count)))
    for _ in range(count * 120):
        candidate = next((a.env_id for a in allocator.assignments if a.subterrain != target), None)
        if candidate is None:
            break
        history.extend(allocator.reset([candidate]))
    assert all(a.subterrain == target for a in allocator.assignments)
    for start in range(0, len(history) - 119, 120):
        assert Counter(a.subterrain for a in history[start:start + 120]) == SOURCE_RESET_BLOCK_COUNTS
