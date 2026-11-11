"""Deterministic environment allocation; physical terrain proportions stay equal."""
from __future__ import annotations

from fractions import Fraction

import torch

from .rows import COLUMN_SPECS, NUM_COLUMNS, NUM_ROWS, OMNI_COLUMNS

TARGETED_PROFILE = "stairs20_omni30_v1"
DESCENDING_COLUMN = next(i for i, (name, _) in enumerate(COLUMN_SPECS) if name == "stairs_descending")


def _apportion(total, weights):
    exact = [total * weight for weight in weights]
    counts = [int(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def targeted_column_counts(num_envs):
    """Largest remainder by group, then equally by column (ties: column order)."""
    if type(num_envs) is not int or num_envs < NUM_COLUMNS:
        raise ValueError("environment count must be an integer covering all 20 columns")
    omni, descending, other = _apportion(num_envs, [Fraction(3, 10), Fraction(2, 10), Fraction(5, 10)])
    others = [i for i in range(NUM_COLUMNS) if i not in (*OMNI_COLUMNS, DESCENDING_COLUMN)]
    counts = [0] * NUM_COLUMNS
    counts[DESCENDING_COLUMN] = descending
    for columns, total in ((OMNI_COLUMNS, omni), (others, other)):
        for column, count in zip(columns, _apportion(total, [Fraction(1, len(columns))] * len(columns))):
            counts[column] = count
    if min(counts) < 1:
        raise ValueError("environment count is too small to retain every column under 30/20/50 allocation")
    return counts


def apply_targeted_origins(terrain, num_envs, origins):
    """Called inside the importer, before scene/command/curriculum construction."""
    if tuple(origins.shape) != (NUM_ROWS, NUM_COLUMNS, 3):
        raise ValueError("targeted curriculum requires the unchanged 10 x 20 terrain grid")
    counts = torch.tensor(targeted_column_counts(num_envs), device=terrain.device, dtype=torch.long)
    terrain.terrain_types = torch.repeat_interleave(torch.arange(NUM_COLUMNS, device=terrain.device), counts)
    return origins[terrain.terrain_levels, terrain.terrain_types]
