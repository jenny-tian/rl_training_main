"""Pure, seeded geometry generators for the Lite3 Source Terrain scan.

This module intentionally does not import Isaac Sim or bind PhysX materials.
It produces auditable trimesh geometry and metadata that a later environment
adapter can consume. In particular, friction-region meshes are not ready for
simulation until each region is bound to its registered material.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh


SOURCE_TERRAIN_FAMILIES = (
    "flat",
    "multi_scale_rough",
    "slope",
    "multi_step_stairs",
    "slope_plus_rough",
    "friction_variation",
)
SLOPE_DIRECTIONS = ("uphill", "downhill", "cross_slope_left", "cross_slope_right")
STAIR_DIRECTIONS = ("ascending", "descending")
FRICTION_LAYOUTS = ("left_right_bands", "front_back_bands", "alternating_regions")

_SUPPORT_DEPTH_M = 1.0
_MAX_HEIGHTFIELD_VERTICES = 100_000
_MAX_FRICTION_REGIONS = 256


@dataclass(frozen=True)
class SourceTerrainCfg:
    """Complete geometric parameters for one Source scan cell.

    Numeric difficulty values remain candidates until a separate scan
    manifest is locked. ``friction_range`` records material candidates only;
    it does not assert effective PhysX friction.
    """

    family: Literal[
        "flat",
        "multi_scale_rough",
        "slope",
        "multi_step_stairs",
        "slope_plus_rough",
        "friction_variation",
    ]
    size: tuple[float, float] = (10.0, 4.0)
    seed: int = 10_000
    flat_length_m: float = 3.0
    transition_length_m: float = 0.5
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
        _validate_size(self.size)
        _validate_seed(self.seed)
        _finite_positive(self.flat_length_m, "flat_length_m")
        _finite_positive(self.transition_length_m, "transition_length_m")
        _finite_positive(self.grid_spacing_m, "grid_spacing_m")
        if self.flat_length_m + self.transition_length_m >= self.size[0]:
            raise ValueError("flat and transition lengths must leave a challenge segment")
        if self.family in {"multi_scale_rough", "slope_plus_rough"}:
            _validate_roughness(self.rough_peak_valley_m, self.spatial_scales_m)
            _finite_positive(self.rough_audit_window_m, "rough_audit_window_m")
            if self.rough_audit_window_m > min(
                self.size[1], self.size[0] - self.flat_length_m - self.transition_length_m
            ):
                raise ValueError("rough audit window must fit in the challenge segment")
        if self.family in {"slope", "slope_plus_rough"}:
            _finite_positive(self.slope_angle_deg, "slope_angle_deg")
            if self.slope_angle_deg >= 45.0:
                raise ValueError("slope_angle_deg must be below 45 degrees for this scan generator")
            if self.direction not in SLOPE_DIRECTIONS:
                raise ValueError(f"direction must be one of {SLOPE_DIRECTIONS}")
        if self.family == "multi_step_stairs":
            if self.direction not in STAIR_DIRECTIONS:
                raise ValueError(f"direction must be one of {STAIR_DIRECTIONS}")
            if type(self.step_count) is not int or not 1 <= self.step_count <= 64:
                raise ValueError("step_count must be an integer in [1, 64]")
            _finite_positive(self.step_height_m, "step_height_m")
            _finite_positive(self.tread_depth_m, "tread_depth_m")
            stair_end = self.flat_length_m + self.transition_length_m + self.step_count * self.tread_depth_m
            if stair_end >= self.size[0]:
                raise ValueError("stair geometry must leave an exit segment")
            if self.step_height_m * self.step_count >= _SUPPORT_DEPTH_M:
                raise ValueError("total stair height must be less than the support depth")
        if self.family == "friction_variation":
            low, high = _validate_interval(self.friction_range, "friction_range")
            if low <= 0.0:
                raise ValueError("friction candidates must be positive")
            if self.friction_layout not in FRICTION_LAYOUTS:
                raise ValueError(f"friction_layout must be one of {FRICTION_LAYOUTS}")
            if type(self.friction_region_count) is not int or not 2 <= self.friction_region_count <= _MAX_FRICTION_REGIONS:
                raise ValueError(
                    f"friction_region_count must be an integer in [2, {_MAX_FRICTION_REGIONS}]"
                )
            if high < low:
                raise ValueError("friction_range upper bound must be >= lower bound")


@dataclass(frozen=True)
class FrictionRegion:
    """Geometry-to-material assignment proposal; not a bound PhysX material."""

    region_id: int
    mesh_index: int
    friction_coefficient_candidate: float
    bounds_xy_m: tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceTerrainGeometry:
    """Generated track geometry, start origin, and geometry audit metadata."""

    family: str
    meshes: tuple[trimesh.Trimesh, ...]
    origin: tuple[float, float, float]
    metadata: dict[str, object] = field(default_factory=dict)
    friction_regions: tuple[FrictionRegion, ...] = ()

    @property
    def requires_region_material_binding(self) -> bool:
        return bool(self.friction_regions)

    def as_isaaclab_mesh_result(self) -> tuple[list[trimesh.Trimesh], np.ndarray]:
        """Return the standard mesh result where one terrain material suffices.

        Friction-variation geometry deliberately fails closed: the current
        importer binds one material to a combined terrain mesh, so treating
        these region meshes as ready-to-run would silently erase the intended
        material differences.
        """

        if self.requires_region_material_binding:
            raise RuntimeError(
                "friction-region geometry requires a material-aware collider importer"
            )
        return list(self.meshes), np.asarray(self.origin, dtype=np.float64)


def generate_source_terrain(cfg: SourceTerrainCfg) -> SourceTerrainGeometry:
    """Generate exactly one seeded terrain instance for a registered family."""

    if not isinstance(cfg, SourceTerrainCfg):
        raise TypeError("cfg must be a SourceTerrainCfg")
    if cfg.family == "flat":
        return _generate_flat(cfg)
    if cfg.family == "multi_scale_rough":
        return _generate_rough(cfg)
    if cfg.family == "slope":
        return _generate_slope(cfg)
    if cfg.family == "multi_step_stairs":
        return _generate_stairs(cfg)
    if cfg.family == "slope_plus_rough":
        return _generate_slope_plus_rough(cfg)
    if cfg.family == "friction_variation":
        return _generate_friction_variation(cfg)
    raise AssertionError("family validation and generator dispatch differ")


def _generate_flat(cfg: SourceTerrainCfg) -> SourceTerrainGeometry:
    length, width = cfg.size
    mesh = _box_mesh(0.0, length, 0.0, width, -_SUPPORT_DEPTH_M, 0.0)
    return SourceTerrainGeometry(
        family=cfg.family,
        meshes=(mesh,),
        origin=(cfg.flat_length_m / 2.0, width / 2.0, 0.0),
        metadata={"geometry": "plane", "track_length_m": length, "track_width_m": width},
    )


def _generate_rough(cfg: SourceTerrainCfg) -> SourceTerrainGeometry:
    x, y = _heightfield_axes(cfg, cfg.spatial_scales_m)
    heights, rough_meta = _rough_heightfield(cfg, x, y)
    mesh = _heightfield_prism(x, y, heights)
    width = cfg.size[1]
    return SourceTerrainGeometry(
        family=cfg.family,
        meshes=(mesh,),
        origin=(cfg.flat_length_m / 2.0, width / 2.0, 0.0),
        metadata={
            "track_length_m": cfg.size[0],
            "track_width_m": width,
            "rough_start_x_m": cfg.flat_length_m,
            "fully_rough_start_x_m": cfg.flat_length_m + cfg.transition_length_m,
            "rough_lane_width_m": width,
            **rough_meta,
        },
    )


def _generate_slope(cfg: SourceTerrainCfg) -> SourceTerrainGeometry:
    x, y = _heightfield_axes(cfg, (4.0 * cfg.grid_spacing_m,))
    heights, slope_meta = _slope_heightfield(cfg, x, y)
    return SourceTerrainGeometry(
        family=cfg.family,
        meshes=(_heightfield_prism(x, y, heights),),
        origin=(cfg.flat_length_m / 2.0, cfg.size[1] / 2.0, 0.0),
        metadata={"track_length_m": cfg.size[0], "track_width_m": cfg.size[1], **slope_meta},
    )


def _generate_stairs(cfg: SourceTerrainCfg) -> SourceTerrainGeometry:
    length, width = cfg.size
    entry_x = cfg.flat_length_m + cfg.transition_length_m
    exit_x = entry_x + cfg.step_count * cfg.tread_depth_m
    if cfg.direction == "ascending":
        initial_height = 0.0
        levels = [cfg.step_height_m * (index + 1) for index in range(cfg.step_count)]
    else:
        initial_height = cfg.step_height_m * cfg.step_count
        levels = [cfg.step_height_m * (cfg.step_count - index - 1) for index in range(cfg.step_count)]

    bottom_z = min(-_SUPPORT_DEPTH_M, min(levels + [initial_height]) - 0.1)
    profile: list[tuple[float, float]] = [(0.0, initial_height), (entry_x, initial_height)]
    previous_height = initial_height
    edges = []
    for index, level in enumerate(levels):
        riser_x = entry_x + index * cfg.tread_depth_m
        tread_end_x = riser_x + cfg.tread_depth_m
        profile.extend(((riser_x, level), (tread_end_x, level)))
        edges.append({"riser_x_m": riser_x, "tread_end_x_m": tread_end_x,
                      "height_before_m": previous_height, "height_after_m": level})
        previous_height = level
    terminal_height = levels[-1]
    profile.extend(((length, terminal_height), (length, bottom_z), (0.0, bottom_z)))
    profile = _remove_collinear_profile_points(profile)
    mesh = _extrude_xz_profile(profile, width)
    return SourceTerrainGeometry(
        family=cfg.family,
        meshes=(mesh,),
        origin=(cfg.flat_length_m / 2.0, width / 2.0, initial_height),
        metadata={
            "track_length_m": length,
            "track_width_m": width,
            "direction": cfg.direction,
            "step_count": cfg.step_count,
            "step_height_m": cfg.step_height_m,
            "tread_depth_m": cfg.tread_depth_m,
            "entry_x_m": entry_x,
            "exit_x_m": exit_x,
            "initial_surface_height_m": initial_height,
            "terminal_surface_height_m": terminal_height,
            "steps": edges,
        },
    )


def _generate_slope_plus_rough(cfg: SourceTerrainCfg) -> SourceTerrainGeometry:
    x, y = _heightfield_axes(cfg, cfg.spatial_scales_m)
    slope, slope_meta = _slope_heightfield(cfg, x, y)
    rough, rough_meta = _rough_heightfield(cfg, x, y)
    heights = slope + rough
    slope_meta["max_combined_abs_height_m"] = float(np.max(np.abs(heights)))
    return SourceTerrainGeometry(
        family=cfg.family,
        meshes=(_heightfield_prism(x, y, heights),),
        origin=(cfg.flat_length_m / 2.0, cfg.size[1] / 2.0, 0.0),
        metadata={
            "track_length_m": cfg.size[0],
            "track_width_m": cfg.size[1],
            **slope_meta,
            **rough_meta,
            "composition": "slope_heightfield_plus_tapered_multiscale_heightfield",
        },
    )


def _generate_friction_variation(cfg: SourceTerrainCfg) -> SourceTerrainGeometry:
    length, width = cfg.size
    low, high = cfg.friction_range
    rng = np.random.default_rng(cfg.seed)
    count = cfg.friction_region_count
    # Include both candidate endpoints when the interval has width, so each
    # instance exercises the registered range rather than an accidental narrow sample.
    values = np.linspace(low, high, count, dtype=np.float64)
    rng.shuffle(values)
    boxes = _friction_region_bounds(cfg, count)
    meshes: list[trimesh.Trimesh] = []
    regions: list[FrictionRegion] = []
    for region_id, ((x0, x1, y0, y1), coefficient) in enumerate(zip(boxes, values, strict=True)):
        mesh_index = len(meshes)
        meshes.append(_box_mesh(x0, x1, y0, y1, -_SUPPORT_DEPTH_M, 0.0))
        regions.append(
            FrictionRegion(
                region_id=region_id,
                mesh_index=mesh_index,
                friction_coefficient_candidate=float(coefficient),
                bounds_xy_m=(x0, x1, y0, y1),
            )
        )
    return SourceTerrainGeometry(
        family=cfg.family,
        meshes=tuple(meshes),
        origin=(cfg.flat_length_m / 2.0, width / 2.0, 0.0),
        metadata={
            "track_length_m": length,
            "track_width_m": width,
            "layout": cfg.friction_layout,
            "candidate_friction_range": [low, high],
            "effective_friction_confirmed": False,
            "material_binding_status": "not_implemented_geometry_only",
            "region_count": count,
        },
        friction_regions=tuple(regions),
    )


def _friction_region_bounds(
    cfg: SourceTerrainCfg, count: int
) -> list[tuple[float, float, float, float]]:
    length, width = cfg.size
    bounds: list[tuple[float, float, float, float]] = []
    if cfg.friction_layout == "left_right_bands":
        edges = np.linspace(0.0, width, count + 1)
        bounds.extend((0.0, length, float(edges[i]), float(edges[i + 1])) for i in range(count))
    elif cfg.friction_layout == "front_back_bands":
        edges = np.linspace(0.0, length, count + 1)
        bounds.extend((float(edges[i]), float(edges[i + 1]), 0.0, width) for i in range(count))
    else:
        # Each full-width row receives its own number of columns. This tiles
        # the entire track even for prime counts and extreme aspect ratios.
        rows = min(count, max(2, int(round(math.sqrt(count * width / length)))))
        counts = [count // rows + (row < count % rows) for row in range(rows)]
        y_edges = np.linspace(0.0, width, rows + 1)
        for row in range(rows):
            x_edges = np.linspace(0.0, length, counts[row] + 1)
            for column in range(counts[row]):
                bounds.append((float(x_edges[column]), float(x_edges[column + 1]),
                               float(y_edges[row]), float(y_edges[row + 1])))
    if len(bounds) != count:
        raise AssertionError("friction region layout produced an incorrect region count")
    return bounds


def _heightfield_axes(cfg: SourceTerrainCfg, scales: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    length, width = cfg.size
    spacing = min(cfg.grid_spacing_m, min(scales) / 4.0)
    if (math.ceil(length / spacing) + 4) * (math.ceil(width / spacing) + 1) > _MAX_HEIGHTFIELD_VERTICES:
        raise ValueError("source terrain heightfield exceeds the vertex budget")
    x = _axis(0.0, length, spacing, (cfg.flat_length_m,
                                     cfg.flat_length_m + cfg.transition_length_m,
                                     length - cfg.transition_length_m))
    y = _axis(0.0, width, spacing)
    if len(x) * len(y) > _MAX_HEIGHTFIELD_VERTICES:
        raise ValueError("source terrain heightfield exceeds the vertex budget")
    return x, y


def _slope_heightfield(
    cfg: SourceTerrainCfg, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    angle_rad = math.radians(cfg.slope_angle_deg)
    grade = math.tan(angle_rad)
    entry = cfg.flat_length_m
    exit_start = cfg.size[0] - cfg.transition_length_m
    ramp_in = _smoothstep((x - entry) / cfg.transition_length_m)
    ramp_out = _smoothstep((cfg.size[0] - x) / cfg.transition_length_m)
    envelope = ramp_in * ramp_out
    if cfg.direction == "uphill":
        sign = 1.0
        gradient = sign * grade * envelope
        profile = _integrate_gradient(x, gradient)
        heights = np.broadcast_to(profile[None, :], (len(y), len(x))).copy()
        metadata = {
            "direction": cfg.direction,
            "slope_angle_deg": cfg.slope_angle_deg,
            "main_axis_grade_sign": sign,
            "max_x_grade": float(np.tan(np.radians(cfg.slope_angle_deg)) * np.max(envelope)),
            "entry_transition_x_m": [entry, entry + cfg.transition_length_m],
            "exit_transition_x_m": [exit_start, cfg.size[0]],
        }
    elif cfg.direction == "downhill":
        sign = -1.0
        gradient = sign * grade * envelope
        profile = _integrate_gradient(x, gradient)
        heights = np.broadcast_to(profile[None, :], (len(y), len(x))).copy()
        metadata = {
            "direction": cfg.direction,
            "slope_angle_deg": cfg.slope_angle_deg,
            "main_axis_grade_sign": sign,
            "max_x_grade": float(np.tan(np.radians(cfg.slope_angle_deg)) * np.max(envelope)),
            "entry_transition_x_m": [entry, entry + cfg.transition_length_m],
            "exit_transition_x_m": [exit_start, cfg.size[0]],
        }
    else:
        sign = 1.0 if cfg.direction == "cross_slope_left" else -1.0
        transverse = sign * grade * envelope
        heights = transverse[None, :] * (y[:, None] - cfg.size[1] / 2.0)
        metadata = {
            "direction": cfg.direction,
            "slope_angle_deg": cfg.slope_angle_deg,
            "cross_axis_grade_sign": sign,
            "positive_y_semantics": "left",
            "max_cross_grade": float(np.max(np.abs(transverse))),
            "entry_transition_x_m": [entry, entry + cfg.transition_length_m],
            "exit_transition_x_m": [exit_start, cfg.size[0]],
        }
    metadata["slope_height_range_m"] = [float(np.min(heights)), float(np.max(heights))]
    return heights, metadata


def _rough_heightfield(
    cfg: SourceTerrainCfg, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(cfg.seed)
    signal = np.zeros((len(y), len(x)), dtype=np.float64)
    band_rms: list[float] = []
    for scale in cfg.spatial_scales_m:
        band = _sample_coarse_noise(rng, x, y, scale)
        band -= float(np.mean(band))
        rms = float(np.sqrt(np.mean(np.square(band))))
        if not math.isfinite(rms) or rms <= 1.0e-12:
            raise ValueError("rough noise band has no finite spatial variation")
        signal += band / rms
        band_rms.append(rms)
    signal /= math.sqrt(float(len(cfg.spatial_scales_m)))
    plateau_start = cfg.flat_length_m + cfg.transition_length_m
    plateau_mask = x >= plateau_start
    local_before = _local_rough_statistics(cfg, x, y, signal)
    peak_valley = local_before["sampled_local_peak_valley_max_m"]
    if not math.isfinite(peak_valley) or peak_valley <= 1.0e-12:
        raise ValueError("rough terrain produced a degenerate height field")
    signal *= cfg.rough_peak_valley_m / peak_valley
    taper = _smoothstep((x - cfg.flat_length_m) / cfg.transition_length_m)
    heights = signal * taper[None, :]
    measured = float(np.max(heights[:, plateau_mask]) - np.min(heights[:, plateau_mask]))
    metadata = {
        "rough_spatial_scales_m": [float(value) for value in cfg.spatial_scales_m],
        "rough_band_rms_before_scaling": band_rms,
        "requested_peak_valley_m": cfg.rough_peak_valley_m,
        "requested_peak_valley_semantics": "maximum over registered sampled contact-scale windows",
        "full_challenge_peak_valley_m": measured,
        **_local_rough_statistics(cfg, x, y, heights),
        "rough_height_range_m": [float(np.min(heights[:, plateau_mask])),
                                 float(np.max(heights[:, plateau_mask]))],
        "rough_transition_x_m": [cfg.flat_length_m, plateau_start],
    }
    return heights, metadata


def _local_rough_statistics(
    cfg: SourceTerrainCfg, x: np.ndarray, y: np.ndarray, heights: np.ndarray
) -> dict[str, object]:
    """Audit overlapping square windows using physical coordinates.

    The 0.4 m default is a development candidate, not a calibrated Lite3
    footprint. Windows advance by half their width and include the final
    boundary. Statistics describe mesh vertices in those windows; they are
    not actual footfall or contact measurements.
    """

    window = cfg.rough_audit_window_m
    start = cfg.flat_length_m + cfg.transition_length_m
    xs = _axis(start, cfg.size[0] - window, window / 2.0)
    ys = _axis(0.0, cfg.size[1] - window, window / 2.0)
    ranges = []
    for y0 in ys:
        iy0, iy1 = np.searchsorted(y, (y0 - 1e-10, y0 + window + 1e-10))
        for x0 in xs:
            ix0, ix1 = np.searchsorted(x, (x0 - 1e-10, x0 + window + 1e-10))
            sample = heights[iy0:iy1, ix0:ix1]
            if sample.size < 4 or min(sample.shape) < 2:
                raise ValueError("rough audit window is too small for the heightfield resolution")
            ranges.append(float(np.ptp(sample)))
    return {
        "rough_audit_window_m": window,
        "rough_audit_window_stride_max_m": window / 2.0,
        "rough_audit_window_count": len(ranges),
        "sampled_local_peak_valley_max_m": float(np.max(ranges)),
        "sampled_local_peak_valley_p50_m": float(np.percentile(ranges, 50)),
        "sampled_local_peak_valley_p95_m": float(np.percentile(ranges, 95)),
        "local_audit_semantics": "mesh vertex statistics in sampled square windows; not measured foot contacts",
    }


def _sample_coarse_noise(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, wavelength_m: float
) -> np.ndarray:
    nx = max(2, int(math.ceil((x[-1] - x[0]) / wavelength_m)) + 1)
    ny = max(2, int(math.ceil((y[-1] - y[0]) / wavelength_m)) + 1)
    cx = np.linspace(x[0], x[-1], nx, dtype=np.float64)
    cy = np.linspace(y[0], y[-1], ny, dtype=np.float64)
    coarse = rng.standard_normal((ny, nx))
    x_pos = (x - cx[0]) / (cx[-1] - cx[0]) * (nx - 1)
    y_pos = (y - cy[0]) / (cy[-1] - cy[0]) * (ny - 1)
    ix0 = np.floor(x_pos).astype(np.int64).clip(0, nx - 2)
    iy0 = np.floor(y_pos).astype(np.int64).clip(0, ny - 2)
    tx = (x_pos - ix0)[None, :]
    ty = (y_pos - iy0)[:, None]
    a = coarse[iy0[:, None], ix0[None, :]]
    b = coarse[iy0[:, None], ix0[None, :] + 1]
    c = coarse[iy0[:, None] + 1, ix0[None, :]]
    d = coarse[iy0[:, None] + 1, ix0[None, :] + 1]
    return ((1.0 - ty) * ((1.0 - tx) * a + tx * b)
            + ty * ((1.0 - tx) * c + tx * d))


def _axis(start: float, stop: float, spacing: float, boundaries: tuple[float, ...] = ()) -> np.ndarray:
    count = max(1, int(math.ceil((stop - start) / spacing)))
    points = list(np.linspace(start, stop, count + 1, dtype=np.float64))
    points.extend(float(value) for value in boundaries if start < value < stop)
    ordered = np.asarray(sorted(set(points)), dtype=np.float64)
    # Avoid near-duplicate grid/boundary coordinates and zero-area faces.
    return ordered[np.concatenate(([True], np.diff(ordered) > 1e-10))]


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _integrate_gradient(x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    profile = np.zeros_like(x, dtype=np.float64)
    profile[1:] = np.cumsum((gradient[:-1] + gradient[1:]) * 0.5 * np.diff(x))
    return profile


def _heightfield_prism(x: np.ndarray, y: np.ndarray, heights: np.ndarray) -> trimesh.Trimesh:
    if heights.shape != (len(y), len(x)):
        raise ValueError("heightfield shape must be [len(y), len(x)]")
    if not np.isfinite(heights).all():
        raise ValueError("heightfield contains non-finite values")
    bottom_z = min(-_SUPPORT_DEPTH_M, float(np.min(heights)) - 0.1)
    xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    top = np.column_stack((xy, heights.reshape(-1)))
    bottom = np.column_stack((xy, np.full(len(xy), bottom_z, dtype=np.float64)))
    vertices = np.vstack((top, bottom))
    faces: list[tuple[int, int, int]] = []
    rows, cols = len(y), len(x)
    offset = rows * cols
    for iy in range(rows - 1):
        for ix in range(cols - 1):
            a = iy * cols + ix
            b, c, d = a + 1, a + cols, a + cols + 1
            faces.extend(((a, b, c), (b, d, c),
                          (offset + a, offset + c, offset + b),
                          (offset + b, offset + c, offset + d)))
    for ix in range(cols - 1):
        a, b = ix, ix + 1
        faces.extend(((a, offset + a, b), (b, offset + a, offset + b)))
        a, b = (rows - 1) * cols + ix, (rows - 1) * cols + ix + 1
        faces.extend(((a, b, offset + a), (b, offset + b, offset + a)))
    for iy in range(rows - 1):
        a, b = iy * cols, (iy + 1) * cols
        faces.extend(((a, b, offset + a), (b, offset + b, offset + a)))
        a, b = iy * cols + cols - 1, (iy + 1) * cols + cols - 1
        faces.extend(((a, offset + a, b), (b, offset + a, offset + b)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def _box_mesh(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float
) -> trimesh.Trimesh:
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError("box mesh dimensions must be positive")
    mesh = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    mesh.apply_translation(((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0))
    return mesh


def _extrude_xz_profile(profile: list[tuple[float, float]], width: float) -> trimesh.Trimesh:
    if len(profile) < 3:
        raise ValueError("extruded profile must have at least three points")
    points = np.asarray(profile, dtype=np.float64)
    area2 = float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                         - np.roll(points[:, 0], -1) * points[:, 1]))
    if abs(area2) <= 1.0e-12:
        raise ValueError("extruded profile has zero area")
    if area2 < 0.0:
        points = points[::-1]
    triangles = _triangulate_simple_polygon(points)
    count = len(points)
    front = np.column_stack((points[:, 0], np.zeros(count), points[:, 1]))
    back = np.column_stack((points[:, 0], np.full(count, width), points[:, 1]))
    vertices = np.vstack((front, back))
    faces: list[tuple[int, int, int]] = []
    for a, b, c in triangles:
        faces.append((a, b, c))  # front cap faces -Y for CCW x/z profile
        faces.append((c + count, b + count, a + count))
    for index in range(count):
        nxt = (index + 1) % count
        faces.extend(((index, index + count, nxt + count),
                      (index, nxt + count, nxt)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def _triangulate_simple_polygon(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clip a simple 2D polygon without optional triangulation packages."""

    indices = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    epsilon = 1.0e-12

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return (cross(a, b, point) >= -epsilon and cross(b, c, point) >= -epsilon
                and cross(c, a, point) >= -epsilon)

    guard = 0
    while len(indices) > 3:
        guard += 1
        if guard > len(points) * len(points):
            raise ValueError("profile triangulation did not converge; polygon may not be simple")
        ear_found = False
        for position, current in enumerate(indices):
            previous = indices[position - 1]
            following = indices[(position + 1) % len(indices)]
            a, b, c = points[previous], points[current], points[following]
            if cross(a, b, c) <= epsilon:
                continue
            if any(
                in_triangle(points[other], a, b, c)
                for other in indices
                if other not in {previous, current, following}
            ):
                continue
            triangles.append((previous, current, following))
            del indices[position]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("profile triangulation found no valid ear")
    triangles.append(tuple(indices))
    return triangles


def _remove_collinear_profile_points(
    profile: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for point in profile:
        if result and np.allclose(point, result[-1], rtol=0.0, atol=1.0e-12):
            continue
        result.append(point)
    changed = True
    while changed and len(result) > 3:
        changed = False
        compact = []
        for index, point in enumerate(result):
            previous = np.asarray(result[index - 1])
            current = np.asarray(point)
            following = np.asarray(result[(index + 1) % len(result)])
            cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
                current[1] - previous[1]
            ) * (following[0] - current[0])
            if abs(float(cross)) <= 1.0e-12:
                changed = True
                continue
            compact.append(point)
        if len(compact) < 3:
            raise ValueError("profile simplification collapsed the stair polygon")
        result = compact
    return result


def _validate_size(size: object) -> tuple[float, float]:
    if isinstance(size, (str, bytes)):
        raise ValueError("size must contain two finite positive dimensions")
    try:
        if len(size) != 2:  # type: ignore[arg-type]
            raise ValueError("size must contain two finite positive dimensions")
        length = _finite_positive(size[0], "size[0]")  # type: ignore[index]
        width = _finite_positive(size[1], "size[1]")  # type: ignore[index]
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError("size must contain two finite positive dimensions") from error
    return length, width


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return normalized


def _validate_seed(seed: object) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")


def _validate_interval(interval: object, name: str) -> tuple[float, float]:
    if isinstance(interval, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two finite values")
    try:
        if len(interval) != 2:  # type: ignore[arg-type]
            raise ValueError(f"{name} must contain exactly two finite values")
        low, high = float(interval[0]), float(interval[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError, KeyError) as error:
        raise ValueError(f"{name} must contain exactly two finite values") from error
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} must contain exactly two finite values")
    return low, high


def _validate_roughness(peak_valley: object, scales: object) -> None:
    _finite_positive(peak_valley, "rough_peak_valley_m")
    if isinstance(scales, (str, bytes)):
        raise ValueError("spatial_scales_m must contain at least two positive scales")
    try:
        values = tuple(_finite_positive(value, "spatial_scales_m") for value in scales)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("spatial_scales_m must contain at least two positive scales") from error
    if len(values) < 2 or len(set(values)) < 2:
        raise ValueError("spatial_scales_m must contain at least two distinct positive scales")
