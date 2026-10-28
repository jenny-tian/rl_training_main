"""Asynchronous Source terrain scheduling and transactional reset adapters.

Each environment owns one row containing one tile per sub-terrain. The
weighted episode schedule is independent of the physical tile proportions.
The adapters only change selected origin buffers; they never create a simulator or start training.
Unique tile indices alone are not proof of physical collision isolation.
"""

from __future__ import annotations

import random
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers

SOURCE_RESET_BLOCK_COUNTS = {
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


@dataclass(frozen=True)
class TerrainAssignment:
    env_id: int
    row: int
    column: int
    subterrain: str
    # Zero-based global assignment index, including the initial episodes.
    assignment_index: int
    episode_index: int
    block_index: int
    index_in_block: int


class ReservedRowTerrainAllocator:
    """Reserve a complete terrain bank for every environment.

    ``block_counts`` gives integer episode counts in insertion order. A
    physical grid must have ``num_envs`` rows and ``len(block_counts)``
    columns, with column labels matching that order. All tiles remain L1;
    the row is an ownership index, not a difficulty level.

    Call ``reset(range(num_envs))`` first to assign all initial episodes.
    Subsequent calls may reset any subset. Within a call, IDs are sorted,
    so the result does not depend on the input tensor's ordering. Replaying
    requires the same seed, label order, and sequence of reset batches.
    """

    def __init__(
        self, num_envs: int, block_counts: Mapping[str, int], sampler_seed: int
    ) -> None:
        self._positive_int(num_envs, "num_envs")
        if type(sampler_seed) is not int or sampler_seed < 0:
            raise ValueError("sampler_seed must be an explicit nonnegative integer")
        if not isinstance(block_counts, Mapping) or not block_counts:
            raise ValueError("block_counts must be a nonempty ordered mapping")
        counts = tuple(block_counts.items())
        for name, count in counts:
            if not isinstance(name, str) or not name:
                raise ValueError("sub-terrain names must be nonempty strings")
            self._positive_int(count, "block count")

        self._num_envs = num_envs
        self._labels = tuple(name for name, _ in counts)
        self.sampler_seed = sampler_seed
        self._tickets = tuple(
            column for column, (_, count) in enumerate(counts) for _ in range(count)
        )
        self._rng = random.Random(sampler_seed)
        self._order: list[int] = []
        self._cursor = 0
        self._assignment_count = 0
        self._active: list[TerrainAssignment | None] = [None] * num_envs
        self._episodes = [0] * num_envs

    @staticmethod
    def _positive_int(value: int, name: str) -> None:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._num_envs, len(self._labels)

    @property
    def column_labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def block_size(self) -> int:
        return len(self._tickets)

    @property
    def assignments(self) -> tuple[TerrainAssignment | None, ...]:
        """Read-only snapshot of the active assignments, ordered by env ID."""
        return tuple(self._active)

    def reset(self, env_ids: Sequence[int]) -> tuple[TerrainAssignment, ...]:
        """Assign new episodes without moving any non-resetting environment.

        Invalid batches are rejected before RNG or assignment state changes.
        Empty batches are no-ops. This is scheduling only: a future runtime
        adapter must validate its grid before calling and apply all returned
        assignments before resuming physics.
        """
        if not isinstance(env_ids, Sequence) or isinstance(env_ids, (str, bytes)):
            raise ValueError("env_ids must be a sequence of integer IDs")
        ids = tuple(env_ids)
        if any(type(i) is not int or not 0 <= i < self._num_envs for i in ids):
            raise ValueError("env_ids must contain in-range integer IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("env_ids must be unique")
        ids = tuple(sorted(ids))
        if not ids:
            return ()
        if self._assignment_count == 0 and ids != tuple(range(self._num_envs)):
            raise ValueError("the first reset must initialize every environment")

        result = []
        for env_id in ids:
            if self._cursor == len(self._order):
                self._order = list(self._tickets)
                self._rng.shuffle(self._order)
                self._cursor = 0
            column = self._order[self._cursor]
            assignment = TerrainAssignment(
                env_id=env_id,
                row=env_id,
                column=column,
                subterrain=self._labels[column],
                assignment_index=self._assignment_count,
                episode_index=self._episodes[env_id],
                block_index=self._assignment_count // self.block_size,
                index_in_block=self._assignment_count % self.block_size,
            )
            self._active[env_id] = assignment
            self._episodes[env_id] += 1
            self._assignment_count += 1
            self._cursor += 1
            result.append(assignment)
        return tuple(result)


def assign_source_terrain_for_parallel_reset(
    env, env_ids, sampler_seed: int
) -> tuple[TerrainAssignment, ...]:
    """Apply scheduled row/column/origin assignments before Isaac Lab resets assets.

    The Source configuration provides a reserved row per environment, or an
    explicitly enabled shared bank with audited collision groups, and one
    equally weighted column per registered sub-terrain. Isaac Lab calls
    curriculum terms before ``scene.reset`` and reset events, so changing
    ``terrain.env_origins`` here updates the origin consumed by the subsequent
    root-state reset. This adapter does not qualify physical separation or
    collisions between moving robots.
    """

    if type(sampler_seed) is not int or sampler_seed < 0:
        raise ValueError("sampler_seed must be an explicit nonnegative integer")
    num_envs = getattr(env, "num_envs", None)
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, numbers.Integral)
        or num_envs < 1
    ):
        raise RuntimeError(
            "parallel Source terrain sampling requires a positive environment count"
        )
    num_envs = int(num_envs)
    reset_ids = _normalize_reset_ids(env_ids, num_envs)
    if not reset_ids:
        return ()

    scene = getattr(env, "scene", None)
    terrain = getattr(scene, "terrain", None)
    terrain_cfg = getattr(terrain, "cfg", None)
    generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if terrain is None or terrain_cfg is None or generator_cfg is None:
        raise RuntimeError(
            "parallel Source terrain sampling requires a generated terrain importer"
        )
    if getattr(terrain_cfg, "terrain_type", None) != "generator":
        raise RuntimeError(
            "parallel Source terrain sampling requires terrain_type='generator'"
        )
    if not getattr(terrain_cfg, "use_terrain_origins", False):
        raise RuntimeError("parallel Source terrain sampling requires terrain origins")

    num_rows = getattr(generator_cfg, "num_rows", None)
    num_cols = getattr(generator_cfg, "num_cols", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", None)
    bank_rows = getattr(terrain_cfg, "shared_bank_rows", None)
    if bank_rows is not None:
        if type(bank_rows) is not int or not 1 <= bank_rows <= num_envs:
            raise RuntimeError("shared bank rows must be between one and the environment count")
        from .shared_terrain_bank import audit_shared_collision_groups

        audit_shared_collision_groups(scene)
    expected_rows = num_envs if bank_rows is None else bank_rows
    if num_rows != expected_rows:
        raise RuntimeError(
            "Source terrain map rows do not match the explicit reserved-row/shared-bank layout"
        )
    if not isinstance(sub_terrains, Mapping) or not sub_terrains:
        raise RuntimeError(
            "parallel Source terrain sub-terrain configuration is unavailable"
        )
    labels = tuple(sub_terrains)
    if num_cols != len(labels) or labels != tuple(SOURCE_RESET_BLOCK_COUNTS):
        raise RuntimeError(
            "parallel Source terrain columns must match the registered 15-class layout"
        )
    expected_proportion = 1.0 / len(labels)
    try:
        proportions = tuple(float(sub_terrains[name].proportion) for name in labels)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "parallel Source terrain map proportions are unavailable"
        ) from error
    if any(
        not math.isclose(value, expected_proportion, rel_tol=0.0, abs_tol=1e-12)
        for value in proportions
    ):
        raise RuntimeError(
            "parallel Source terrain map columns must be equally weighted"
        )

    terrain_origins = getattr(terrain, "terrain_origins", None)
    env_origins = getattr(terrain, "env_origins", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if any(
        value is None
        for value in (terrain_origins, env_origins, terrain_types, terrain_levels)
    ):
        raise RuntimeError(
            "terrain importer must expose initialized origins, types, and levels"
        )
    if tuple(terrain_origins.shape) != (expected_rows, len(labels), 3):
        raise RuntimeError(
            "terrain origin grid does not match Source allocation configuration"
        )
    if tuple(env_origins.shape) != (num_envs, 3):
        raise RuntimeError(
            "terrain environment-origin buffer does not match environment count"
        )
    if tuple(terrain_types.shape) != (num_envs,) or tuple(terrain_levels.shape) != (
        num_envs,
    ):
        raise RuntimeError("terrain type/level buffers do not match environment count")
    if not bool(terrain_origins.isfinite().all().item()):
        raise RuntimeError("terrain origin grid contains non-finite values")

    signature = (num_envs, labels, proportions, tuple(terrain_origins.shape), bank_rows)
    current_allocator = getattr(env, "_source_parallel_terrain_allocator", None)
    is_new_allocator = current_allocator is None
    if is_new_allocator:
        if bank_rows is None:
            candidate_allocator = ReservedRowTerrainAllocator(
                num_envs, SOURCE_RESET_BLOCK_COUNTS, sampler_seed
            )
        else:
            from .shared_terrain_bank import SharedBankTerrainAllocator

            candidate_allocator = SharedBankTerrainAllocator(
                num_envs, SOURCE_RESET_BLOCK_COUNTS, sampler_seed, bank_rows=bank_rows
            )
    else:
        if getattr(env, "_source_parallel_terrain_signature", None) != signature:
            raise RuntimeError(
                "parallel Source terrain layout changed after reset scheduling started"
            )
        if current_allocator.sampler_seed != sampler_seed:
            raise RuntimeError(
                "parallel Source sampler seed cannot change while the environment is running"
            )
        candidate_allocator = copy.deepcopy(current_allocator)

    assignments = candidate_allocator.reset(reset_ids)
    env_index = terrain_levels.new_tensor([item.env_id for item in assignments])
    rows = terrain_levels.new_tensor([item.row for item in assignments])
    columns = terrain_types.new_tensor([item.column for item in assignments])
    old_levels = terrain_levels[env_index].clone()
    old_types = terrain_types[env_index].clone()
    old_origins = env_origins[env_index].clone()
    try:
        terrain_levels[env_index] = rows
        terrain_types[env_index] = columns
        env_origins[env_index] = terrain_origins[rows, columns]
    except Exception:
        terrain_levels[env_index] = old_levels
        terrain_types[env_index] = old_types
        env_origins[env_index] = old_origins
        raise
    env._source_parallel_terrain_allocator = candidate_allocator
    env._source_parallel_terrain_signature = signature
    return assignments


def set_uniform_subterrain_proportions(sub_terrains: Mapping[str, object]) -> None:
    """Make one physical map column per ordered Source sub-terrain class."""

    if not isinstance(sub_terrains, Mapping) or not sub_terrains:
        raise ValueError("sub_terrains must be a nonempty ordered mapping")
    proportion = 1.0 / len(sub_terrains)
    for cfg in sub_terrains.values():
        cfg.proportion = proportion


def initialize_reserved_row_environment_origins(terrain, num_envs: int) -> None:
    """Give every environment a unique owner row before the first env reset."""

    initialize_source_environment_origins(terrain, num_envs)


def initialize_source_environment_origins(terrain, num_envs: int, *, bank_rows: int | None = None) -> None:
    """Initialize reserved or explicitly shared origins without consuming tickets."""

    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if bank_rows is not None and (type(bank_rows) is not int or not 1 <= bank_rows <= num_envs):
        raise ValueError("bank_rows must be between one and the environment count")
    expected_rows = num_envs if bank_rows is None else bank_rows
    terrain_origins = getattr(terrain, "terrain_origins", None)
    env_origins = getattr(terrain, "env_origins", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if any(
        value is None
        for value in (terrain_origins, env_origins, terrain_types, terrain_levels)
    ):
        raise RuntimeError(
            "terrain importer must expose initialized origins, types, and levels"
        )
    if len(terrain_origins.shape) != 3 or tuple(terrain_origins.shape[0:1]) != (
        expected_rows,
    ):
        raise RuntimeError("Source terrain origins must match the declared row count")
    if (
        terrain_origins.shape[1] != len(SOURCE_RESET_BLOCK_COUNTS)
        or terrain_origins.shape[2] != 3
    ):
        raise RuntimeError(
            "Source terrain must provide 15 columns of xyz origins"
        )
    if tuple(env_origins.shape) != (num_envs, 3):
        raise RuntimeError(
            "terrain environment-origin buffer does not match environment count"
        )
    if tuple(terrain_types.shape) != (num_envs,) or tuple(terrain_levels.shape) != (
        num_envs,
    ):
        raise RuntimeError("terrain type/level buffers do not match environment count")
    if not bool(terrain_origins.isfinite().all().item()):
        raise RuntimeError("terrain origin grid contains non-finite values")

    rows = terrain_levels.new_tensor([i % expected_rows for i in range(num_envs)])
    initial_columns = terrain_types.new_zeros((num_envs,))
    old_levels = terrain_levels.clone()
    old_types = terrain_types.clone()
    old_origins = env_origins.clone()
    try:
        terrain_levels.copy_(rows)
        terrain_types.copy_(initial_columns)
        terrain.env_origins.copy_(terrain_origins[rows, initial_columns])
    except Exception:
        terrain_levels.copy_(old_levels)
        terrain_types.copy_(old_types)
        terrain.env_origins.copy_(old_origins)
        raise


def _normalize_reset_ids(env_ids, num_envs: int) -> tuple[int, ...]:
    if isinstance(env_ids, slice):
        values = tuple(range(num_envs)[env_ids])
    elif hasattr(env_ids, "reshape") and hasattr(env_ids, "tolist"):
        values = tuple(env_ids.reshape(-1).tolist())
    elif hasattr(env_ids, "tolist"):
        values = tuple(env_ids.tolist())
    elif isinstance(env_ids, numbers.Integral) and not isinstance(env_ids, bool):
        values = (int(env_ids),)
    elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
        values = tuple(env_ids)
    else:
        raise ValueError("env_ids must identify environments being reset")
    if any(type(env_id) is not int or not 0 <= env_id < num_envs for env_id in values):
        raise ValueError(
            "env_ids must contain only in-range integer environment indices"
        )
    if len(set(values)) != len(values):
        raise ValueError("env_ids must contain unique environment indices")
    return values


__all__ = [
    "ReservedRowTerrainAllocator",
    "SOURCE_RESET_BLOCK_COUNTS",
    "TerrainAssignment",
    "assign_source_terrain_for_parallel_reset",
    "initialize_reserved_row_environment_origins",
    "initialize_source_environment_origins",
    "set_uniform_subterrain_proportions",
]
