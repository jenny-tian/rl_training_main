"""Real native rough generation must be deterministic and isolate global RNG state."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
from source_block_geometry_audit import load_block_function


def assert_rng_equal(a, b):
    assert a[0] == b[0] and a[2:] == b[2:]
    np.testing.assert_array_equal(a[1], b[1])


def test_native_rough_determinism_seed_diversity_and_rng_isolation():
    function, cfg, _ = load_block_function()
    cfg.__dict__.update(family="multi_scale_rough", size=(8., 8.), seed=20261111,
                        rough_profile="reference_uniform")
    state = np.random.get_state()
    meshes, origin = function(.041, cfg)
    assert_rng_equal(state, np.random.get_state())
    repeat, other_origin = function(.041, cfg)
    np.testing.assert_array_equal(meshes[0].vertices, repeat[0].vertices)
    np.testing.assert_array_equal(origin, other_origin)
    assert np.allclose(origin, [4., 4., .06])
    for difficulty, seed in ((.141, cfg.seed), (.041, cfg.seed + 1)):
        other = copy.deepcopy(cfg)
        other.seed = seed
        changed, _ = function(difficulty, other)
        assert not np.array_equal(meshes[0].vertices, changed[0].vertices)
    assert_rng_equal(state, np.random.get_state())


def test_native_failure_restores_global_rng():
    function, cfg, _ = load_block_function()
    cfg.__dict__.update(family="multi_scale_rough", size=(8., 8.), seed=42,
                        rough_profile="reference_uniform")
    def fail(*_):
        np.random.uniform(size=20)
        raise RuntimeError("injected native generation failure")
    function.__globals__["HfRandomUniformTerrainCfg"] = lambda **_: SimpleNamespace(function=fail)
    before = np.random.get_state()
    with pytest.raises(RuntimeError, match="injected"):
        function(.041, cfg)
    assert_rng_equal(before, np.random.get_state())


def test_statistics_keep_reference_rough_separate_from_original_l2():
    from rl_training.source_curriculum.block import BlockStats
    baseline = BlockStats("cpu").snapshot()
    changed = BlockStats("cpu", rough_profile="reference_uniform").snapshot()
    assert baseline["physical_difficulty"] == "L2"
    assert "blocks/multi_scale_rough_a/L2_transitions" in baseline["metrics"]
    assert changed["physical_difficulty"] == "L2_except_reference_uniform_rough"
    assert "blocks/multi_scale_rough_a/L2_transitions" not in changed["metrics"]
    assert "blocks/reference_uniform_rough_a/reference_rough_transitions" in changed["metrics"]
    assert "blocks/reference_uniform_rough_b/reference_rough_transitions" in changed["metrics"]
    assert changed["physical_difficulty_by_column"].count("reference_rough") == 2
    assert changed["physical_difficulty_by_column"].count("L2") == 18
