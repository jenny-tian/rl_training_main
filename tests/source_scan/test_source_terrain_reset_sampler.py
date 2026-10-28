"""Reset-level Source Terrain scheduling without loading Isaac Sim."""

import importlib.util
import numpy as np
import sys
import torch
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_terrain_reset_sampler.py"
)
SPEC = importlib.util.spec_from_file_location(
    "source_reset_sampler_under_test", MODULE_PATH
)
sampler_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sampler_module
SPEC.loader.exec_module(sampler_module)

ResetBlockTerrainSampler = sampler_module.ResetBlockTerrainSampler
assign_source_terrain_for_reset = sampler_module.assign_source_terrain_for_reset
terrain_column_subterrain_indices = sampler_module.terrain_column_subterrain_indices


SOURCE_SUBTERRAIN_PROPORTIONS = {
    "flat": 0.10,
    "multi_scale_rough": 0.20,
    "slope_uphill": 0.05,
    "slope_downhill": 0.05,
    "slope_cross_left": 0.05,
    "slope_cross_right": 0.05,
    "stairs_ascending": 0.125,
    "stairs_descending": 0.125,
    "slope_rough_uphill": 0.05,
    "slope_rough_downhill": 0.05,
    "slope_rough_cross_left": 0.025,
    "slope_rough_cross_right": 0.025,
    "friction_left_right": 1 / 30,
    "friction_front_back": 1 / 30,
    "friction_alternating": 1 / 30,
}


def _isaaclab_column_assignment(proportions, num_cols):
    """Independent transcription of TerrainGenerator's proportion loop."""
    weights = np.asarray(list(proportions.values()))
    weights /= np.sum(weights)
    cumulative = np.cumsum(weights)
    return tuple(
        int(np.min(np.where(column / num_cols + 0.001 < cumulative)[0]))
        for column in range(num_cols)
    )


def _fake_env(num_envs=1):
    terrain_origins = torch.arange(120 * 3, dtype=torch.float32).reshape(1, 120, 3)
    sub_terrains = {
        name: SimpleNamespace(proportion=proportion)
        for name, proportion in SOURCE_SUBTERRAIN_PROPORTIONS.items()
    }
    generator_cfg = SimpleNamespace(num_rows=1, num_cols=120, sub_terrains=sub_terrains)
    terrain_cfg = SimpleNamespace(
        terrain_type="generator",
        use_terrain_origins=True,
        terrain_generator=generator_cfg,
    )
    terrain = SimpleNamespace(
        cfg=terrain_cfg,
        terrain_origins=terrain_origins,
        env_origins=terrain_origins[:, 0, :].clone(),
        terrain_types=torch.zeros((1,), dtype=torch.long),
        terrain_levels=torch.zeros((1,), dtype=torch.long),
    )
    return SimpleNamespace(num_envs=num_envs, scene=SimpleNamespace(terrain=terrain))


def test_column_map_matches_isaaclab_and_registered_120_reset_counts():
    actual = terrain_column_subterrain_indices(SOURCE_SUBTERRAIN_PROPORTIONS, 120)
    assert actual == _isaaclab_column_assignment(SOURCE_SUBTERRAIN_PROPORTIONS, 120)
    assert Counter(actual) == Counter(
        {
            0: 12,
            1: 24,
            2: 6,
            3: 6,
            4: 6,
            5: 6,
            6: 15,
            7: 15,
            8: 6,
            9: 6,
            10: 3,
            11: 3,
            12: 4,
            13: 4,
            14: 4,
        }
    )


def test_every_reset_block_visits_each_generated_column_once():
    terrain_map = terrain_column_subterrain_indices(SOURCE_SUBTERRAIN_PROPORTIONS, 120)
    sequence = ResetBlockTerrainSampler(num_cols=120, sampler_seed=91_207).take(360)
    for block_start in (0, 120, 240):
        block = sequence[block_start : block_start + 120]
        assert sorted(block) == list(range(120))
        assert Counter(terrain_map[column] for column in block) == Counter(terrain_map)


def test_fixed_seed_is_reproducible_and_chunking_does_not_change_schedule():
    whole = ResetBlockTerrainSampler(num_cols=120, sampler_seed=44_021).take(287)
    chunked_sampler = ResetBlockTerrainSampler(num_cols=120, sampler_seed=44_021)
    chunked = (
        chunked_sampler.take(13)
        + chunked_sampler.take(120)
        + chunked_sampler.take(1)
        + chunked_sampler.take(153)
    )
    repeated = ResetBlockTerrainSampler(num_cols=120, sampler_seed=44_021).take(287)
    assert whole == chunked == repeated


@pytest.mark.parametrize(
    "proportions,num_cols",
    [
        ({}, 120),
        ({"flat": 0.0}, 120),
        ({"flat": -0.1}, 120),
        ({"flat": float("nan")}, 120),
        ({"flat": float("inf")}, 120),
        ({"flat": 1.0}, 0),
    ],
)
def test_column_map_rejects_invalid_weights_and_sizes(proportions, num_cols):
    with pytest.raises(ValueError):
        terrain_column_subterrain_indices(proportions, num_cols)


@pytest.mark.parametrize("seed", [None, -1, True, 1.5])
def test_sampler_requires_an_explicit_nonnegative_integer_seed(seed):
    with pytest.raises(ValueError, match="sampler_seed"):
        ResetBlockTerrainSampler(num_cols=120, sampler_seed=seed)


def test_reset_hook_updates_type_level_and_spawn_origin_before_reset():
    env = _fake_env()
    selected = assign_source_terrain_for_reset(
        env, torch.tensor([0]), sampler_seed=13_007
    )
    assert env.scene.terrain.terrain_types.tolist() == [selected]
    assert env.scene.terrain.terrain_levels.tolist() == [0]
    assert torch.equal(
        env.scene.terrain.env_origins[0],
        env.scene.terrain.terrain_origins[0, selected],
    )
    assert 0 <= selected < 120


def test_reset_hook_keeps_one_sequence_across_separate_resets():
    env = _fake_env()
    actual = [
        assign_source_terrain_for_reset(env, [0], sampler_seed=13_007)
        for _ in range(120)
    ]
    expected = ResetBlockTerrainSampler(num_cols=120, sampler_seed=13_007).take(120)
    assert actual == list(expected)


@pytest.mark.parametrize("num_envs,env_ids", [(2, [0]), (1, [1]), (1, [0, 0])])
def test_reset_hook_fails_closed_outside_one_unique_environment(num_envs, env_ids):
    env = _fake_env(num_envs=num_envs)
    with pytest.raises(RuntimeError):
        assign_source_terrain_for_reset(env, env_ids, sampler_seed=23)
    assert env.scene.terrain.terrain_types.tolist() == [0]


def test_reset_hook_rejects_multirow_or_changed_layout_without_advancing():
    env = _fake_env()
    first = assign_source_terrain_for_reset(env, [0], sampler_seed=13_007)
    env.scene.terrain.cfg.terrain_generator.num_rows = 2
    with pytest.raises(RuntimeError, match="one terrain row"):
        assign_source_terrain_for_reset(env, [0], sampler_seed=13_007)
    assert env.scene.terrain.terrain_types.tolist() == [first]


def test_empty_reset_does_not_consume_a_terrain_assignment():
    env = _fake_env()
    assert assign_source_terrain_for_reset(env, [], sampler_seed=13_007) is None
    actual = assign_source_terrain_for_reset(env, [0], sampler_seed=13_007)
    expected = ResetBlockTerrainSampler(num_cols=120, sampler_seed=13_007).next_column()
    assert actual == expected
