"""Offline environment configuration checks for pinned bank realizations."""

import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from rl_training.source_scan import terrain_config
from scripts.reinforcement_learning.online_adaptation import run_source_qualification_scan as runner


@pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.987654321, 1.0])
def test_environment_uses_one_fixed_realization_without_another_random_draw(monkeypatch, difficulty):
    cfg = NS(scene=NS(height_scanner_base=NS(update_period=0.02), terrain=NS(terrain_generator=NS())),
             sim=NS(dt=0.005), terminations=NS(), commands=NS(base_velocity=NS(ranges=NS())))
    utils = ModuleType("isaaclab_tasks.utils")
    utils.parse_env_cfg = lambda *a, **kw: cfg
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", ModuleType("isaaclab_tasks"))
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", utils)
    monkeypatch.setitem(sys.modules, "rl_training.tasks", ModuleType("rl_training.tasks"))
    monkeypatch.setattr(terrain_config, "source_subterrain_from_cell", lambda cell, seed: NS(seed=seed))
    cell = {"cell_id": "test_cell", "resolved_parameters": {"geometry": {"size": [12.0, 4.0]}},
            "qualification_geometry_difficulty": difficulty}
    actual = runner._build_env_cfg(cell, 10001, 20030, device="cpu")
    generator = actual.scene.terrain.terrain_generator
    assert generator.seed == 10001 and generator.difficulty_range == (difficulty, difficulty)
    assert generator.num_rows == generator.num_cols == 1
    assert generator.sub_terrains["test_cell"].seed == 10001
    assert actual.scene.height_scanner_base.update_period == 0.005


def test_historical_scan_cells_keep_their_original_realization():
    assert terrain_config.source_geometry_difficulty({}) == 0.0


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf"), "0.5", None])
def test_invalid_realization_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        terrain_config.source_geometry_difficulty({"qualification_geometry_difficulty": value})
