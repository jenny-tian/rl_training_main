"""Isaac Lab adapters for the registered Lite3 Source Terrain families.

The geometry implementation in :mod:`source_terrain_tracks` deliberately has no
Isaac Sim dependency.  This module is the narrow bridge used by
``TerrainGenerator``.  It keeps the physical parameters in one place and adds
an internal face-colour encoding for spatial friction regions.  The latter is
consumed by :class:`SourceTerrainImporter`; it must not be treated as a visual
colour scheme.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .source_terrain_tracks import (
    SOURCE_TERRAIN_FAMILIES,
    SourceTerrainCfg,
    generate_source_terrain,
)


# Alpha 254 is reserved for the private friction-region transport encoding.
# The RGB channels contain a fixed-point coefficient so the importer does not
# need to guess which sub-terrain produced a face after TerrainGenerator has
# concatenated and translated all meshes.
FRICTION_REGION_MARKER_ALPHA = 254
FRICTION_COEFFICIENT_SCALE = 1_000_000
FRICTION_COEFFICIENT_MAX_Q = (1 << 24) - 1


def encode_friction_region_colour(coefficient: float) -> np.ndarray:
    """Encode a positive candidate coefficient into one RGBA face colour."""

    if isinstance(coefficient, bool) or not isinstance(coefficient, numbers.Real):
        raise ValueError("friction coefficient must be a finite real number")
    value = float(coefficient)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("friction coefficient must be a finite positive number")
    quantized = int(round(value * FRICTION_COEFFICIENT_SCALE))
    if not 0 < quantized <= FRICTION_COEFFICIENT_MAX_Q:
        raise ValueError(
            "friction coefficient is outside the face-colour transport range "
            f"(0, {FRICTION_COEFFICIENT_MAX_Q / FRICTION_COEFFICIENT_SCALE}]"
        )
    return np.asarray(
        [quantized >> 16, (quantized >> 8) & 0xFF, quantized & 0xFF, FRICTION_REGION_MARKER_ALPHA],
        dtype=np.uint8,
    )


def decode_friction_region_colour(colour: Any) -> float | None:
    """Decode a friction coefficient or return ``None`` for an ordinary face."""

    rgba = np.asarray(colour, dtype=np.uint8).reshape(-1)
    if rgba.size != 4 or int(rgba[3]) != FRICTION_REGION_MARKER_ALPHA:
        return None
    quantized = (int(rgba[0]) << 16) | (int(rgba[1]) << 8) | int(rgba[2])
    if quantized == 0:
        raise ValueError("friction-region marker contains a zero coefficient")
    return quantized / FRICTION_COEFFICIENT_SCALE


def _mark_friction_regions(
    meshes: list[trimesh.Trimesh], geometry: Any
) -> None:
    """Attach the coefficient transport colour to every friction-region mesh."""

    for region in geometry.friction_regions:
        mesh = meshes[region.mesh_index]
        colour = encode_friction_region_colour(region.friction_coefficient_candidate)
        mesh.visual.face_colors = np.tile(colour, (len(mesh.faces), 1))


def generate_source_terrain_geometry(
    difficulty: float, cfg: "SourceTerrainTrackCfg"
):
    """Generate the auditable geometry for one Isaac Lab Source track.

    ``TerrainGenerator`` supplies the common seed to every sub-terrain before
    calling this function.  A deterministic child seed derived from that seed,
    the family, and the per-cell difficulty gives repeated family instances
    distinct geometry.  ``difficulty`` selects only the child seed; candidate
    physical values are kept explicit in the configuration until the scan
    manifest selects them.
    """

    if not isinstance(difficulty, numbers.Real) or not np.isfinite(float(difficulty)):
        raise ValueError("difficulty must be a finite real number")
    if not 0.0 <= float(difficulty) <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")

    family_index = SOURCE_TERRAIN_FAMILIES.index(cfg.family)
    difficulty_word = min((1 << 32) - 1, int(float(difficulty) * (1 << 32)))
    seed_sequence = np.random.SeedSequence(
        [int(cfg.seed) & 0xFFFFFFFF, int(cfg.seed) >> 32, family_index, difficulty_word]
    )
    instance_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    geometry_cfg = SourceTerrainCfg(
        family=cfg.family,
        size=(float(cfg.size[0]), float(cfg.size[1])),
        seed=instance_seed,
        flat_length_m=float(cfg.flat_length_m),
        transition_length_m=float(cfg.transition_length_m),
        exit_flat_length_m=float(cfg.exit_flat_length_m),
        rough_peak_valley_m=float(cfg.rough_peak_valley_m),
        rough_audit_window_m=float(cfg.rough_audit_window_m),
        spatial_scales_m=tuple(float(value) for value in cfg.spatial_scales_m),
        slope_angle_deg=float(cfg.slope_angle_deg),
        direction=cfg.direction,
        step_count=int(cfg.step_count),
        step_height_m=float(cfg.step_height_m),
        tread_depth_m=float(cfg.tread_depth_m),
        friction_range=(float(cfg.friction_range[0]), float(cfg.friction_range[1])),
        friction_layout=cfg.friction_layout,
        friction_region_count=int(cfg.friction_region_count),
        grid_spacing_m=float(cfg.grid_spacing_m),
    )
    return generate_source_terrain(geometry_cfg)


def source_terrain_track(
    difficulty: float, cfg: "SourceTerrainTrackCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one Source track using the standard Isaac Lab callback shape.

    The geometry helper is kept separate so the scan runner can use the same
    deterministic instance metadata for its offline geometry metrics.
    """

    geometry = generate_source_terrain_geometry(difficulty, cfg)
    meshes = [mesh.copy() for mesh in geometry.meshes]
    if geometry.requires_region_material_binding:
        _mark_friction_regions(meshes, geometry)
    return meshes, np.asarray(geometry.origin, dtype=np.float64)


@configclass
class SourceTerrainTrackCfg(SubTerrainBaseCfg):
    """Configuration for one of the six registered Source Terrain families."""

    function = source_terrain_track
    family: str = "flat"
    seed: int = 10_000
    flat_length_m: float = 3.0
    transition_length_m: float = 0.5
    exit_flat_length_m: float = 0.0
    rough_peak_valley_m: float = 0.02
    rough_audit_window_m: float = 0.4
    spatial_scales_m: tuple[float, ...] = (0.2, 0.4)
    slope_angle_deg: float = 5.0
    direction: str = "uphill"
    step_count: int = 3
    step_height_m: float = 0.02
    tread_depth_m: float = 0.5
    friction_range: tuple[float, float] = (0.7, 1.0)
    friction_layout: str = "left_right_bands"
    friction_region_count: int = 4
    grid_spacing_m: float = 0.1

    def __post_init__(self) -> None:
        if self.family not in SOURCE_TERRAIN_FAMILIES:
            raise ValueError(f"unsupported Source Terrain family: {self.family!r}")


__all__ = [
    "FRICTION_COEFFICIENT_MAX_Q",
    "FRICTION_COEFFICIENT_SCALE",
    "FRICTION_REGION_MARKER_ALPHA",
    "SourceTerrainTrackCfg",
    "decode_friction_region_colour",
    "generate_source_terrain_geometry",
    "encode_friction_region_colour",
    "source_terrain_track",
]
