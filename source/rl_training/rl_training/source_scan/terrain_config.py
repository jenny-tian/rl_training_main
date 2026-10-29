"""Translate one fully resolved scan cell into an Isaac Lab sub-terrain cfg."""

from __future__ import annotations

from typing import Any
import math


def source_geometry_difficulty(cell: dict[str, Any]) -> float:
    """Resolve a pinned bank realization; ordinary scan cells retain zero."""
    value = cell.get("qualification_geometry_difficulty", 0.0)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("qualification_geometry_difficulty must be finite in [0, 1]")
    return float(value)


def source_subterrain_from_cell(cell: dict[str, Any], *, seed: int):
    """Build a one-cell Source Terrain cfg from manifest-resolved parameters.

    This function does not start Isaac Sim and is kept separate from the runner
    so that family mapping and level resolution can be tested offline.
    """

    if not isinstance(cell, dict):
        raise TypeError("cell must be a mapping")
    family = cell.get("terrain_family")
    resolved = cell.get("resolved_parameters")
    if not isinstance(resolved, dict) or not isinstance(resolved.get("geometry"), dict):
        raise ValueError("cell must contain resolved_parameters.geometry")
    geometry = resolved["geometry"]
    direction = cell.get("direction_or_layout")

    from rl_training.tasks.manager_based.locomotion.velocity.terrains.source_terrain_isaac import (
        SourceTerrainTrackCfg,
    )

    common = {
        "proportion": 1.0,
        "family": family,
        "seed": seed,
        "size": tuple(geometry["size"]),
        "flat_length_m": geometry["flat_length_m"],
        "transition_length_m": geometry["transition_length_m"],
        "grid_spacing_m": geometry["grid_spacing_m"],
    }
    if family in ("multi_scale_rough", "slope", "slope_plus_rough"):
        common["exit_flat_length_m"] = geometry["exit_flat_length_m"]
    if family in ("multi_scale_rough", "slope_plus_rough"):
        common.update(
            rough_peak_valley_m=geometry["rough_peak_valley_m"],
            rough_audit_window_m=geometry["rough_audit_window_m"],
            spatial_scales_m=tuple(geometry["spatial_scales_m"]),
        )
    if family in ("slope", "slope_plus_rough"):
        common.update(
            slope_angle_deg=geometry["slope_angle_deg"],
            direction=direction,
        )
    if family == "multi_step_stairs":
        common.update(
            step_count=geometry["step_count"],
            step_height_m=geometry["step_height_m"],
            tread_depth_m=geometry["tread_depth_m"],
            direction=direction,
        )
    if family == "friction_variation":
        common.update(
            friction_range=tuple(geometry["friction_range"]),
            friction_region_count=geometry["friction_region_count"],
            friction_layout=direction,
        )
    if family not in {
        "flat", "multi_scale_rough", "slope", "multi_step_stairs",
        "slope_plus_rough", "friction_variation",
    }:
        raise ValueError(f"unsupported Source terrain family: {family!r}")
    return SourceTerrainTrackCfg(**common)


__all__ = ["source_subterrain_from_cell", "source_geometry_difficulty"]
