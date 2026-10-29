"""Termination terms specific to the Lite3 Source terrain task."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg

from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated
from rl_training.source_scan.rigid_body_reference import capture_body_reference_state


def source_terrain_tile_bounds(
    env,
    asset_cfg: SceneEntityCfg,
    size: tuple[float, float],
    flat_length_m: float,
    collision_radius_m: float,
    numerical_margin_m: float,
):
    """Terminate each environment before any Lite3 collision envelope exits its tile."""

    terrain = env.scene.terrain
    robot = env.scene[asset_cfg.name]
    required_terrain_tensors = (
        "terrain_origins",
        "terrain_levels",
        "terrain_types",
    )
    missing = [name for name in required_terrain_tensors if not hasattr(terrain, name)]
    if missing:
        raise RuntimeError(
            "Source tile guard requires terrain allocation state: " + ", ".join(missing)
        )
    if not hasattr(robot.data, "body_link_pos_w") or not hasattr(robot.data, "body_link_lin_vel_w"):
        raise RuntimeError("Source tile guard requires explicit body_link_pos_w and body_link_lin_vel_w APIs")
    if asset_cfg.body_ids is None:
        raise RuntimeError("Source tile guard requires resolved IDs for every robot rigid body")

    observer = getattr(env, "_source_tile_bounds_observer", None)
    diagnostics = {} if observer is not None else None
    reference = capture_body_reference_state(robot.data, asset_cfg.body_ids) if observer is not None else None
    # Both inputs describe the link origin. The old body_lin_vel_w alias is
    # COM velocity and must not be paired with a link-origin position.
    terminated = source_tile_bounds_terminated(
        reference["body_positions_w"] if reference is not None else robot.data.body_link_pos_w[:, asset_cfg.body_ids],
        reference["body_linear_velocities_w"] if reference is not None else robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids],
        terrain.terrain_origins,
        terrain.terrain_levels,
        terrain.terrain_types,
        size=size,
        flat_length_m=flat_length_m,
        collision_radius_m=collision_radius_m,
        numerical_margin_m=numerical_margin_m,
        control_period_s=float(env.step_dt),
        diagnostics=diagnostics,
    )
    if observer is not None:
        diagnostics.update(reference)
        ids = list(range(len(robot.body_names)))[asset_cfg.body_ids] if isinstance(asset_cfg.body_ids, slice) else list(asset_cfg.body_ids)
        diagnostics.update(
            body_ids=ids,
            body_names=[robot.body_names[index] for index in ids],
            common_step_counter=int(env.common_step_counter),
            episode_steps=env.episode_length_buf.detach().clone(),
        )
        observer(diagnostics)
    return terminated


__all__ = ["source_terrain_tile_bounds"]
