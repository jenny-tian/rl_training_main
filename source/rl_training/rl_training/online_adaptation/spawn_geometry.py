"""Conservative continuous-range spawn proof from deployed collision shapes.

The initial joint/root height is fixed. XY translation and world-Z yaw leave
vertical clearance unchanged. A horizontal enclosing disk therefore covers
all yaw angles, including every angle in the declared perturbation interval.
"""

import math
import numpy as np


def collision_bounds(shape):
    matrix = np.asarray(shape["world_from_shape"], dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("collision transform must be a finite affine 4x4 matrix")
    linear, center = matrix[:3, :3], matrix[:3, 3]
    kind = shape["type"]
    if kind == "Cube":
        size = float(shape["size"])
        if not math.isfinite(size) or size <= 0:
            raise ValueError("collision size must be finite and positive")
        extent = np.abs(linear).sum(axis=1) * size / 2
    elif kind in {"Sphere", "Capsule", "Cylinder"}:
        radius = float(shape["radius"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("collision radius must be finite and positive")
        extent = radius * np.linalg.norm(linear, axis=1)
        if kind != "Sphere":
            height = float(shape["height"])
            if not math.isfinite(height) or height < 0 or shape["axis"] not in {"X", "Y", "Z"}:
                raise ValueError("invalid collision axis/height")
            axis = "XYZ".index(shape["axis"])
            if kind == "Cylinder":
                extent = radius * np.linalg.norm(np.delete(linear, axis, axis=1), axis=1)
            extent += height / 2 * np.abs(linear[:, axis])
    else:
        raise ValueError(f"unsupported collision geometry: {kind}; refuse incomplete proof")
    return center - extent, center + extent


def certify_spawn_range(shapes, *, root_position, platform_bounds, floor_z=0., xy_half_range=.02):
    """Return evidence; invalid geometry is a failed audit, never a discarded start."""
    root = np.asarray(root_position, dtype=float)
    platform = np.asarray(platform_bounds, dtype=float)
    if (root.shape != (3,) or platform.shape != (2, 2) or not np.isfinite(root).all()
            or not np.isfinite(platform).all() or np.any(platform[:, 1] <= platform[:, 0])
            or not math.isfinite(floor_z) or not math.isfinite(xy_half_range) or xy_half_range < 0):
        raise ValueError("invalid spawn/platform dimensions")
    if not shapes:
        raise ValueError("no collision shapes; cannot certify an empty robot")
    records = []
    radius = 0.
    min_z = math.inf
    for shape in shapes:
        lower, upper = collision_bounds(shape)
        delta = np.maximum(np.abs(lower[:2] - root[:2]), np.abs(upper[:2] - root[:2]))
        radius = max(radius, float(np.linalg.norm(delta)))
        min_z = min(min_z, float(lower[2]))
        records.append({"path": shape["path"], "lower": lower.tolist(), "upper": upper.tolist()})
    margin = np.minimum(root[:2] - platform[:, 0], platform[:, 1] - root[:2]) - radius - xy_half_range
    clearance = min_z - floor_z
    return {"schema": "lite3_spawn_geometry_disk_v1", "valid": bool(np.all(margin >= 0) and clearance >= -1e-6),
            "horizontal_enclosing_radius_m": radius, "platform_margin_xy_m": margin.tolist(),
            "minimum_collision_clearance_m": clearance, "numeric_tolerance_m": 1e-6,
            "xy_half_range_m": xy_half_range, "yaw_coverage": "all_world_z_angles",
            "assumptions": ["fixed joint pose", "fixed height and roll/pitch", "planar platform at floor_z", "self-collision disabled"],
            "collision_count": len(shapes), "collision_bounds": records}


def validate_spawn_certificate(root, certificate):
    """Reject a stale model/reset certificate before a managed trial starts."""
    from pathlib import Path
    from .provenance import sha256_file

    root = Path(root)
    if certificate.get("schema") != "lite3_spawn_certificate_v1" or certificate.get("status") != "passed":
        raise ValueError("spawn certificate is not certified")
    bindings = certificate.get("file_sha256", {})
    if not bindings:
        raise ValueError("spawn certificate has no model/reset bindings")
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"spawn geometry certificate is stale: {relative}; repeat the geometry audit")
    proof = certify_spawn_range(certificate["nominal_shapes"], **certificate["proof_inputs"])
    if not proof["valid"] or proof != certificate["nominal_proof"]:
        raise ValueError("spawn geometry certificate proof does not reproduce")
    return sha256_file(root / certificate["certificate_path"])
