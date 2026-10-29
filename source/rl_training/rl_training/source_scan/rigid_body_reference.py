"""Explicit IsaacLab reference-point snapshots and independent CPU checks."""

from __future__ import annotations

import torch


BODY_APIS = {
    "body_positions_w": "body_link_pos_w",
    "body_linear_velocities_w": "body_link_lin_vel_w",
    "body_link_quats_w_wxyz": "body_link_quat_w",
    "body_com_offsets_link_m": "body_com_pos_b",
    "body_com_positions_w": "body_com_pos_w",
    "body_com_linear_velocities_w": "body_com_lin_vel_w",
    "body_angular_velocities_w": "body_link_ang_vel_w",
}
REFERENCE_POINTS = {
    "position": "link_origin", "linear_velocity": "link_origin",
    "vector_frame": "world", "quaternion_order": "wxyz",
    "com_offset_frame": "link", "com_offset_direction": "link_origin_to_com",
    "velocity_relation": "v_link_w = v_com_w - omega_w cross (R_link_w @ r_com_link)",
}
ROOT_APIS = {
    "body_positions_w": "root_link_pos_w",
    "body_linear_velocities_w": "root_link_lin_vel_w",
    "body_link_quats_w_wxyz": "root_link_quat_w",
    "body_com_positions_w": "root_com_pos_w",
    "body_com_linear_velocities_w": "root_com_lin_vel_w",
    "body_angular_velocities_w": "root_link_ang_vel_w",
}


def capture_root_reference_state(data):
    """Root uses body index zero's COM offset, as in IsaacLab root_link_vel_w."""
    result = {}
    for field, api in ROOT_APIS.items():
        try:
            value = getattr(data, api)
        except AttributeError as error:
            raise RuntimeError(f"Source root diagnostics require explicit API: {api}") from error
        result[field] = value[:, None].detach().clone()
    result["body_com_offsets_link_m"] = data.body_com_pos_b[:, :1].detach().clone()
    result["reference_points"] = dict(REFERENCE_POINTS)
    result["state_dtype"] = str(result["body_positions_w"].dtype)
    return result


def capture_body_reference_state(data, body_ids=slice(None)):
    """Require explicit APIs; never fall back to the mixed-point legacy aliases."""
    result = {}
    for field, api in BODY_APIS.items():
        try:
            value = getattr(data, api)
        except AttributeError as error:
            raise RuntimeError(f"Source reference diagnostics require explicit API: {api}") from error
        result[field] = value[:, body_ids].detach().clone()
    result["reference_points"] = dict(REFERENCE_POINTS)
    return result


def validate_body_reference_state(state):
    """Check p_COM = p_link + R r and v_link = v_COM - omega cross R r.

    Use a float64 rotation matrix and component-wise cross product, independent
    of IsaacLab's quaternion/cross implementation. Float32 capture rounding is
    allowed by atol=2e-6, rtol=2e-6 (m or m/s); float64 uses 1e-10 each.
    Quaternion norm tolerance is 1e-5 / 1e-10. These are evidence tolerances,
    never changes to the guard bounds. Nonfinite bodies are reported separately
    and cannot count as verified conversions, even if the XY predicate is finite.
    """
    def require(condition, message):
        if not condition:
            raise ValueError("link-reference telemetry: " + message)

    require(state.get("reference_points") == REFERENCE_POINTS, "reference metadata")
    dtype = state.get("state_dtype")
    require(dtype in ("torch.float32", "torch.float64"), "state dtype")
    atol = rtol = 2e-6 if dtype == "torch.float32" else 1e-10
    norm_atol = 1e-5 if dtype == "torch.float32" else 1e-10

    def tensor(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.float64)
        def floats(tree):
            return [floats(x) for x in tree] if isinstance(tree, list) else float(tree)
        return torch.tensor(floats(value), dtype=torch.float64)

    require(all(key in state for key in BODY_APIS), "missing explicit reference fields")
    values = {key: tensor(state[key]) for key in BODY_APIS}
    pos = values["body_positions_w"]
    require(pos.ndim == 3 and pos.shape[-1] == 3 and pos.shape[1] > 0, "position shape")
    for key, value in values.items():
        width = 4 if key == "body_link_quats_w_wxyz" else 3
        require(value.shape == (*pos.shape[:2], width), "shape: " + key)
    finite = torch.stack([torch.isfinite(value).all(dim=-1) for value in values.values()]).all(dim=0)
    selected = {key: value[finite] for key, value in values.items()}
    quat = selected["body_link_quats_w_wxyz"]
    require(torch.all(torch.abs(torch.linalg.vector_norm(quat, dim=-1) - 1) <= norm_atol), "quaternion norm")
    w, x, y, z = quat.unbind(-1)
    rotation = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), dim=-1).reshape(-1, 3, 3)
    offset = (rotation @ selected["body_com_offsets_link_m"].unsqueeze(-1)).squeeze(-1)
    a, b, c = selected["body_angular_velocities_w"].unbind(-1)
    rx, ry, rz = offset.unbind(-1)
    cross = torch.stack((b*rz-c*ry, c*rx-a*rz, a*ry-b*rx), dim=-1)
    expected_pos = selected["body_positions_w"] + offset
    expected_vel = selected["body_com_linear_velocities_w"] - cross
    errors = {}
    for field, expected, label in (
        ("body_com_positions_w", expected_pos, "com_position"),
        ("body_linear_velocities_w", expected_vel, "link_velocity"),
    ):
        error = torch.abs(selected[field] - expected)
        require(torch.all(error <= atol + rtol * expected.abs()), label + " conversion mismatch")
        errors[label + "_max_abs_error"] = error.max().item() if error.numel() else None
    return {
        "status": "verified" if finite.all().item() else "nonfinite_reference_unverifiable",
        "bodies_verified": int(finite.sum()),
        "unverifiable_environment_body_indices": (~finite).nonzero().tolist(),
        "atol": atol, "rtol": rtol, "quaternion_norm_atol": norm_atol,
        **errors,
    }
