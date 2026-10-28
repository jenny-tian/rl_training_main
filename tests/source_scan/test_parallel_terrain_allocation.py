"""Synthetic allocation tests; these are not Isaac Sim collision evidence."""

import ast
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_training.source_scan.parallel_terrain_allocation import (
    ReservedRowTerrainAllocator,
    SOURCE_RESET_BLOCK_COUNTS,
    assign_source_terrain_for_parallel_reset,
    initialize_reserved_row_environment_origins,
    set_uniform_subterrain_proportions,
)

COUNTS = SOURCE_RESET_BLOCK_COUNTS
ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / (
    "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/"
    "config/quadruped/deeprobotics_lite3/source_terrain_env_cfg.py"
)


def _allocator(num_envs=120, seed=130001):
    return ReservedRowTerrainAllocator(num_envs, COUNTS, seed)


def _fake_parallel_env(num_envs=120, *, proportions=None, num_rows=None, num_cols=None):
    num_cols = len(SOURCE_RESET_BLOCK_COUNTS) if num_cols is None else num_cols
    num_rows = num_envs if num_rows is None else num_rows
    terrain_origins = torch.arange(
        num_rows * num_cols * 3, dtype=torch.float32
    ).reshape(num_rows, num_cols, 3)
    terrain_origins[..., 2] += 0.25
    names = tuple(SOURCE_RESET_BLOCK_COUNTS)
    if proportions is None:
        proportions = [1.0 / num_cols] * num_cols
    sub_terrains = {
        name: SimpleNamespace(proportion=weight)
        for name, weight in zip(names, proportions)
    }
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            terrain_type="generator",
            use_terrain_origins=True,
            terrain_generator=SimpleNamespace(
                num_rows=num_rows, num_cols=num_cols, sub_terrains=sub_terrains
            ),
        ),
        terrain_origins=terrain_origins,
        env_origins=terrain_origins[:, 0, :].clone(),
        terrain_types=torch.zeros(num_envs, dtype=torch.long),
        terrain_levels=torch.zeros(num_envs, dtype=torch.long),
    )
    robot = SimpleNamespace(
        default_root_position=torch.tensor([0.0, 0.0, 0.5]).repeat(num_envs, 1),
        root_position=torch.zeros((num_envs, 3)),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        scene=SimpleNamespace(terrain=terrain),
        robot=robot,
    )

    def reset_root_positions(env_ids):
        robot.root_position[env_ids] = (
            robot.default_root_position[env_ids] + terrain.env_origins[env_ids]
        )

    env.reset_root_positions = reset_root_positions
    return env


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
    namespace = {"SourceTerrainTrackCfg": lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_CONFIG), "exec"),
        namespace,
    )
    actual = namespace["_source_sub_terrains"]("L1")
    assert tuple(actual) == tuple(COUNTS)
    for name, cfg in actual.items():
        assert cfg.proportion == pytest.approx(COUNTS[name] / 120)
    set_uniform_subterrain_proportions(actual)
    assert all(cfg.proportion == pytest.approx(1 / 15) for cfg in actual.values())


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


def test_parallel_reset_adapter_initializes_unique_rows_before_root_reset():
    env = _fake_parallel_env()
    assignments = assign_source_terrain_for_parallel_reset(
        env, torch.arange(env.num_envs), sampler_seed=130001
    )
    assert len(assignments) == 120
    terrain = env.scene.terrain
    assert terrain.terrain_levels.tolist() == list(range(120))
    assert Counter(terrain.terrain_types.tolist()) == Counter(
        {index: COUNTS[name] for index, name in enumerate(COUNTS)}
    )
    assert all(
        torch.equal(
            terrain.env_origins[item.env_id],
            terrain.terrain_origins[item.row, item.column],
        )
        for item in assignments
    )

    # The configured reset event adds scene.env_origins to the robot's default
    # root state, so verify that the adapter has changed those origins first.
    env.reset_root_positions(torch.arange(env.num_envs))
    expected = terrain.env_origins + env.robot.default_root_position
    assert torch.equal(env.robot.root_position, expected)
    assert len({tuple(row.tolist()) for row in terrain.env_origins}) == 120


def test_importer_startup_origin_initialization_reserves_rows_before_reset_hook():
    env = _fake_parallel_env(num_envs=7)
    terrain = env.scene.terrain
    initialize_reserved_row_environment_origins(terrain, env.num_envs)
    assert terrain.terrain_levels.tolist() == list(range(env.num_envs))
    assert terrain.terrain_types.tolist() == [0] * env.num_envs
    assert torch.equal(terrain.env_origins, terrain.terrain_origins[:, 0, :])
    assert len({tuple(row.tolist()) for row in terrain.env_origins}) == env.num_envs

    assignments = assign_source_terrain_for_parallel_reset(
        env, torch.arange(env.num_envs), sampler_seed=44_001
    )
    assert [item.row for item in assignments] == list(range(env.num_envs))
    assert [item.column for item in assignments] == terrain.terrain_types.tolist()


def test_parallel_async_reset_changes_only_selected_rows_and_uses_new_origin():
    env = _fake_parallel_env()
    initial = assign_source_terrain_for_parallel_reset(
        env, torch.arange(env.num_envs), sampler_seed=130001
    )
    terrain = env.scene.terrain
    old_levels = terrain.terrain_levels.clone()
    old_types = terrain.terrain_types.clone()
    old_origins = terrain.env_origins.clone()
    old_active = env._source_parallel_terrain_allocator.assignments
    reset_ids = [1, 7, 54, 119]

    updated = assign_source_terrain_for_parallel_reset(
        env, torch.tensor(reset_ids), sampler_seed=130001
    )
    assert [item.env_id for item in updated] == reset_ids
    assert [item.row for item in updated] == reset_ids
    assert all(item.row == item.env_id for item in updated)
    assert torch.equal(terrain.terrain_levels[reset_ids], old_levels[reset_ids])
    assert torch.equal(
        terrain.terrain_types[reset_ids],
        torch.tensor([item.column for item in updated]),
    )
    for env_id in set(range(120)) - set(reset_ids):
        assert terrain.terrain_types[env_id] == old_types[env_id]
        assert torch.equal(terrain.env_origins[env_id], old_origins[env_id])
        assert (
            env._source_parallel_terrain_allocator.assignments[env_id]
            == old_active[env_id]
        )

    env.reset_root_positions(torch.tensor(reset_ids))
    assert torch.equal(
        env.robot.root_position[reset_ids],
        terrain.env_origins[reset_ids] + env.robot.default_root_position[reset_ids],
    )


def test_parallel_reset_records_follow_actual_asynchronous_reset_ticket_order():
    env = _fake_parallel_env(num_envs=7)
    history = list(
        assign_source_terrain_for_parallel_reset(
            env, torch.arange(7), sampler_seed=44_001
        )
    )
    for env_id in [0, 1, 2, 3, 4, 5, 6] * 17:
        history.extend(
            assign_source_terrain_for_parallel_reset(env, [env_id], sampler_seed=44_001)
        )
    _check_blocks(history)
    assert all(
        item.row == item.env_id
        for item in env._source_parallel_terrain_allocator.assignments
    )


@pytest.mark.parametrize(
    "kwargs,mutate",
    [
        ({"num_rows": 119}, None),
        ({"num_cols": 14}, None),
        ({}, "nonuniform"),
        ({}, "wrong_terrain_type"),
        ({}, "no_origins"),
        ({}, "nan_origin"),
    ],
)
def test_parallel_reset_adapter_fails_closed_before_assignment_on_invalid_map(
    kwargs, mutate
):
    env = _fake_parallel_env(**kwargs)
    if mutate == "nonuniform":
        env.scene.terrain.cfg.terrain_generator.sub_terrains["flat"].proportion = 0.1
    elif mutate == "wrong_terrain_type":
        env.scene.terrain.cfg.terrain_type = "plane"
    elif mutate == "no_origins":
        env.scene.terrain.cfg.use_terrain_origins = False
    elif mutate == "nan_origin":
        env.scene.terrain.terrain_origins[0, 0, 0] = float("nan")
    before_levels = env.scene.terrain.terrain_levels.clone()
    before_types = env.scene.terrain.terrain_types.clone()
    before_origins = env.scene.terrain.env_origins.clone()

    with pytest.raises(RuntimeError):
        assign_source_terrain_for_parallel_reset(
            env, torch.arange(env.num_envs), sampler_seed=130001
        )
    assert torch.equal(env.scene.terrain.terrain_levels, before_levels)
    assert torch.equal(env.scene.terrain.terrain_types, before_types)
    assert torch.equal(env.scene.terrain.env_origins, before_origins)
    assert not hasattr(env, "_source_parallel_terrain_allocator")


def test_parallel_reset_adapter_requires_full_first_reset_and_preserves_queue_after_failure():
    env = _fake_parallel_env(num_envs=7)
    with pytest.raises(ValueError, match="initialize every"):
        assign_source_terrain_for_parallel_reset(env, [0], sampler_seed=130001)
    assert not hasattr(env, "_source_parallel_terrain_allocator")
    initialized = assign_source_terrain_for_parallel_reset(
        env, torch.arange(7), sampler_seed=130001
    )
    reference = _allocator(num_envs=7, seed=130001).reset(list(range(7)))
    assert initialized == reference


def test_parallel_reset_adapter_rejects_seed_change_without_moving_live_environments():
    env = _fake_parallel_env(num_envs=7)
    assign_source_terrain_for_parallel_reset(env, torch.arange(7), sampler_seed=130001)
    active = env._source_parallel_terrain_allocator.assignments
    before = env.scene.terrain.env_origins.clone()
    with pytest.raises(RuntimeError, match="seed cannot change"):
        assign_source_terrain_for_parallel_reset(env, [0], sampler_seed=130002)
    assert env._source_parallel_terrain_allocator.assignments == active
    assert torch.equal(env.scene.terrain.env_origins, before)


def test_parallel_sampler_is_disabled_in_source_environment_config_by_default():
    tree = ast.parse(SOURCE_CONFIG.read_text())
    config_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeeproboticsLite3SourceTerrainEnvCfg"
    )
    defaults = {
        node.target.id: node.value
        for node in config_cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert isinstance(defaults["source_parallel_sampler_enabled"], ast.Constant)
    assert defaults["source_parallel_sampler_enabled"].value is False
