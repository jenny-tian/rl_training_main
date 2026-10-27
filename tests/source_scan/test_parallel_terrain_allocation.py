"""Synthetic allocation tests; these are not Isaac Sim collision evidence."""

import ast
import random
from collections import Counter
from pathlib import Path

import pytest

from rl_training.source_scan.parallel_terrain_allocation import (
    ReservedRowTerrainAllocator,
)

COUNTS = {
    "flat": 12,
    "multi_scale_rough": 24,
    "slope_uphill": 6,
    "slope_downhill": 6,
    "slope_cross_slope_left": 6,
    "slope_cross_slope_right": 6,
    "stairs_ascending": 15,
    "stairs_descending": 15,
    "slope_rough_uphill": 6,
    "slope_rough_downhill": 6,
    "slope_rough_cross_slope_left": 3,
    "slope_rough_cross_slope_right": 3,
    "friction_left_right_bands": 4,
    "friction_front_back_bands": 4,
    "friction_alternating_regions": 4,
}
ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/"
    "config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py"
)


def _allocator(num_envs=120, seed=130001):
    return ReservedRowTerrainAllocator(num_envs, COUNTS, seed)


def _check_active_slots(allocator):
    active = allocator.assignments
    assert all(item is not None for item in active)
    assert len({(item.row, item.column) for item in active}) == len(active)
    assert [item.row for item in active] == list(range(len(active)))


def _check_blocks(history):
    for offset in range(0, len(history) - 119, 120):
        block = history[offset : offset + 120]
        assert Counter(item.subterrain for item in block) == COUNTS
        assert [item.index_in_block for item in block] == list(range(120))
        assert {item.block_index for item in block} == {offset // 120}
    assert [item.assignment_index for item in history] == list(range(len(history)))


def test_block_labels_and_weights_match_actual_source_config_without_loading_isaac():
    # Execute only the pure config factory, with a record-only stand-in for
    # SourceTerrainTrackCfg; no Isaac modules or application are imported.
    tree = ast.parse(SOURCE_CONFIG.read_text())
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_source_sub_terrains")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id in {"SOURCE_TERRAIN_PROPORTIONS", "_LEVEL_CANDIDATES"}
                for t in node.targets
            )
        )
    ]
    namespace = {"SourceTerrainTrackCfg": lambda **kwargs: kwargs}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_CONFIG), "exec"),
        namespace,
    )
    actual = namespace["_source_sub_terrains"]("L1")
    assert tuple(actual) == tuple(COUNTS)
    for name, cfg in actual.items():
        assert cfg["proportion"] == pytest.approx(COUNTS[name] / 120)


def test_initial_batch_reserves_unique_rows_and_balances_all_directions():
    allocator = _allocator()
    assert allocator.grid_shape == (120, 15)
    history = allocator.reset(range(120))
    _check_active_slots(allocator)
    _check_blocks(history)


def test_fast_environment_never_moves_119_live_environments_or_skips_labels():
    allocator = _allocator()
    history = list(allocator.reset(range(120)))
    sleeping = allocator.assignments[1:]
    for _ in range(1200):
        history.extend(allocator.reset([0]))
        assert allocator.assignments[1:] == sleeping
        _check_active_slots(allocator)
    assert {item.subterrain for item in history[120:]} == set(COUNTS)
    _check_blocks(history)


@pytest.mark.parametrize("seed", [0, 130001, 130002, 130003])
def test_many_asynchronous_batches_preserve_occupancy_and_global_reset_counts(seed):
    allocator = _allocator(seed=seed)
    batches = random.Random(20427)
    history = list(allocator.reset(range(120)))
    for _ in range(300):
        ids = batches.sample(range(120), batches.randrange(121))
        before = allocator.assignments
        history.extend(allocator.reset(ids))
        for env_id in set(range(120)) - set(ids):
            assert allocator.assignments[env_id] == before[env_id]
        _check_active_slots(allocator)
    _check_blocks(history)


def test_seed_and_reset_order_are_reproducible_and_input_order_is_canonical():
    left, right = _allocator(), _allocator()
    for ids in [list(range(120)), [19, 0, 7], [], [7], [118, 77]] * 3:
        assert left.reset(ids) == right.reset(list(reversed(ids)))
    other = _allocator(seed=130002)
    assert other.reset(range(120)) != _allocator().reset(range(120))


def test_chunking_same_assignment_order_preserves_ticket_sequence():
    left, right = _allocator(), _allocator()
    left.reset(range(120))
    right.reset(range(120))
    expected = left.reset(range(120))
    actual = right.reset(range(17)) + right.reset(range(17, 120))
    assert actual == expected


@pytest.mark.parametrize(
    "ids", [[120], [-1], [0, 0], [False], [1.0], "0", None, [0, 1, 999]]
)
def test_invalid_batch_does_not_consume_rng_or_change_live_assignments(ids):
    candidate, reference = _allocator(), _allocator()
    candidate.reset(range(120))
    reference.reset(range(120))
    before = candidate.assignments
    with pytest.raises(ValueError):
        candidate.reset(ids)
    assert candidate.assignments == before
    assert candidate.reset(range(120)) == reference.reset(range(120))


def test_first_reset_requires_all_environments_and_empty_reset_is_noop():
    allocator = _allocator()
    assert allocator.reset([]) == ()
    with pytest.raises(ValueError, match="initialize every"):
        allocator.reset([0])
    assert allocator.assignments == (None,) * 120
    assert allocator.reset(range(120)) == _allocator().reset(range(120))


@pytest.mark.parametrize("num_envs", [0, -1, True, 1.5])
def test_rejects_invalid_environment_count(num_envs):
    with pytest.raises(ValueError):
        _allocator(num_envs=num_envs)


@pytest.mark.parametrize("seed", [None, -1, True, 1.5])
def test_requires_explicit_seed(seed):
    with pytest.raises(ValueError, match="sampler_seed"):
        _allocator(seed=seed)


@pytest.mark.parametrize(
    "counts", [{}, {"flat": 0}, {"flat": -1}, {"flat": True}, {"flat": 1.5}, {"": 1}]
)
def test_rejects_invalid_episode_counts(counts):
    with pytest.raises(ValueError):
        ReservedRowTerrainAllocator(120, counts, 0)


def test_input_mapping_mutation_cannot_change_active_layout_or_schedule():
    counts = dict(COUNTS)
    allocator = ReservedRowTerrainAllocator(120, counts, 130001)
    counts.clear()
    assert allocator.reset(range(120)) == _allocator().reset(range(120))


@pytest.mark.parametrize("num_envs", [1, 7, 240])
def test_environment_count_and_120_episode_block_are_independent(num_envs):
    allocator = _allocator(num_envs)
    history = list(allocator.reset(range(num_envs)))
    while len(history) < 360:
        history.extend(allocator.reset([0]))
    _check_active_slots(allocator)
    _check_blocks(history)
