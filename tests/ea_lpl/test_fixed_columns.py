from types import SimpleNamespace

import pytest
import torch

from rl_training.source_scan.parallel_terrain_allocation import (
    SOURCE_RESET_BLOCK_COUNTS,
    FixedColumnTerrainAllocator,
    assign_source_terrain_for_parallel_reset,
)


def fake_env(num_envs):
    labels = tuple(SOURCE_RESET_BLOCK_COUNTS)
    sub = {name: SimpleNamespace(proportion=1.0 / len(labels)) for name in labels}
    generator = SimpleNamespace(num_rows=num_envs, num_cols=len(labels), sub_terrains=sub)
    cfg = SimpleNamespace(terrain_type="generator", use_terrain_origins=True,
                          terrain_generator=generator, shared_bank_rows=None)
    origins = torch.arange(num_envs * len(labels) * 3, dtype=torch.float32).reshape(num_envs, len(labels), 3)
    terrain = SimpleNamespace(cfg=cfg, terrain_origins=origins, env_origins=torch.zeros(num_envs, 3),
                              terrain_types=torch.zeros(num_envs, dtype=torch.long),
                              terrain_levels=torch.zeros(num_envs, dtype=torch.long))
    return SimpleNamespace(num_envs=num_envs, scene=SimpleNamespace(terrain=terrain))


def test_fixed_allocator_keeps_columns_and_counts_episodes():
    columns = tuple(i % 15 for i in range(30))
    allocator = FixedColumnTerrainAllocator(30, SOURCE_RESET_BLOCK_COUNTS, 7, columns, bank_rows=1)
    first = allocator.reset(list(range(30)))
    assert [a.column for a in first] == list(columns)
    assert all(a.row == 0 for a in first)
    again = allocator.reset([3, 1])
    assert [(a.env_id, a.column, a.episode_index) for a in again] == [(1, 1, 1), (3, 3, 1)]


def test_fixed_allocator_rejects_bad_columns():
    with pytest.raises(ValueError):
        FixedColumnTerrainAllocator(3, SOURCE_RESET_BLOCK_COUNTS, 0, (0, 1), bank_rows=1)
    with pytest.raises(ValueError):
        FixedColumnTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, 0, (0, 15), bank_rows=1)
    allocator = FixedColumnTerrainAllocator(2, SOURCE_RESET_BLOCK_COUNTS, 0, (0, 1))
    with pytest.raises(ValueError):
        allocator.reset([0])


def test_parallel_reset_with_fixed_columns_never_changes_columns():
    env = fake_env(30)
    columns = tuple((3 * i) % 15 for i in range(30))
    assign_source_terrain_for_parallel_reset(env, list(range(30)), 5, fixed_columns=columns)
    assert env.scene.terrain.terrain_types.tolist() == list(columns)
    assign_source_terrain_for_parallel_reset(env, [4, 9], 5, fixed_columns=columns)
    assert env.scene.terrain.terrain_types.tolist() == list(columns)
    assert env._source_parallel_terrain_allocator.assignments[4].episode_index == 1


def test_parallel_reset_rejects_changed_fixed_columns():
    env = fake_env(30)
    columns = tuple(i % 15 for i in range(30))
    assign_source_terrain_for_parallel_reset(env, list(range(30)), 5, fixed_columns=columns)
    with pytest.raises(RuntimeError):
        assign_source_terrain_for_parallel_reset(env, [0], 5, fixed_columns=tuple(reversed(columns)))
