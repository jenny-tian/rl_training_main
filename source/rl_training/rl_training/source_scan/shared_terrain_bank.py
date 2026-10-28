"""Offline candidate for reusing Source geometry across robot environments.

No simulator-facing configuration imports this module. Shared positions require
validated inter-environment collision filtering, ray-caster targets and reset
isolation before this allocator can be connected to a runtime environment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .parallel_terrain_allocation import ReservedRowTerrainAllocator, TerrainAssignment


class SharedBankTerrainAllocator:
    """Keep the existing episode schedule but map robots to a bounded tile bank.

    Environment i uses geometry variant i % bank_rows and the scheduled column.
    Multiple environments may occupy exactly the same tile; this is deliberate
    and must NEVER be applied to a scene that permits collisions between them.
    Geometry diversity is bank_rows variants per class, not one per robot.
    """

    def __init__(
        self, num_envs: int, block_counts: Mapping[str, int], sampler_seed: int, *, bank_rows: int
    ) -> None:
        if type(bank_rows) is not int or bank_rows < 1:
            raise ValueError("bank_rows must be a positive integer")
        self._schedule = ReservedRowTerrainAllocator(num_envs, block_counts, sampler_seed)
        self._bank_rows = bank_rows

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._bank_rows, len(self._schedule.column_labels)

    @property
    def assignments(self) -> tuple[TerrainAssignment | None, ...]:
        return tuple(None if assignment is None else self._map(assignment)
                     for assignment in self._schedule.assignments)

    def _map(self, assignment: TerrainAssignment) -> TerrainAssignment:
        return replace(assignment, row=assignment.env_id % self._bank_rows)

    def reset(self, env_ids: Sequence[int]) -> tuple[TerrainAssignment, ...]:
        return tuple(self._map(assignment) for assignment in self._schedule.reset(env_ids))


__all__ = ["SharedBankTerrainAllocator"]
