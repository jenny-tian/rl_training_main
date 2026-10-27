"""Offline prototype for asynchronous Source terrain assignment.

Each environment owns one row containing one tile per sub-terrain. The
weighted episode schedule is independent of the physical tile proportions.
This module never creates a simulator, changes origins, or starts training.
Unique tile indices alone are not proof of physical collision isolation.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


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


__all__ = ["ReservedRowTerrainAllocator", "TerrainAssignment"]
