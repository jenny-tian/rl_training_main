from __future__ import annotations

from collections.abc import Mapping
from typing import Any

HELDOUT_TERRAIN_NAMES = ("boxes", "pyramid_stairs", "pyramid_stairs_inv")


def is_strict_flat_baseline_task(task: str) -> bool:
    return task == "Flat-Deeprobotics-Lite3-v0"


def assert_plane_only_terrain(terrain_cfg: object) -> None:
    terrain_type = getattr(terrain_cfg, "terrain_type", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_type != "plane" or terrain_generator is not None:
        raise ValueError(
            "strict plane-only initialization requires terrain_type='plane' "
            "and terrain_generator=None"
        )


def _require_flat_online_value(name: str, actual: object, expected: object) -> None:
    if isinstance(expected, bool):
        matches = actual is expected
    elif expected is None:
        matches = actual is None
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"FlatOnline deployment requires {name}={expected!r}; got {actual!r}"
        )


def _require_flat_online_attribute(
    owner: object, attribute: str, name: str, expected: object
) -> None:
    try:
        actual = getattr(owner, attribute)
    except AttributeError as error:
        raise ValueError(f"FlatOnline deployment is missing {name}") from error
    _require_flat_online_value(name, actual, expected)


def assert_flat_online_deployment_config(env_cfg: object) -> None:
    """Reject any environment configuration other than the fixed FlatOnline track."""
    try:
        scene = env_cfg.scene
        terrain_cfg = scene.terrain
        terrain_generator = terrain_cfg.terrain_generator
    except AttributeError as error:
        raise ValueError("FlatOnline deployment requires scene.terrain configuration") from error

    _require_flat_online_attribute(scene, "num_envs", "scene.num_envs", 1)
    _require_flat_online_attribute(
        terrain_cfg, "terrain_type", "scene.terrain.terrain_type", "generator"
    )
    _require_flat_online_attribute(
        terrain_cfg, "max_init_terrain_level", "scene.terrain.max_init_terrain_level", 0
    )
    if terrain_generator is None:
        raise ValueError("FlatOnline deployment requires a terrain generator")

    for attribute, expected in (
        ("seed", 0),
        ("curriculum", False),
        ("size", (20.0, 4.0)),
        ("border_width", 0.0),
        ("num_rows", 1),
        ("num_cols", 1),
        ("use_cache", False),
    ):
        _require_flat_online_attribute(
            terrain_generator,
            attribute,
            f"scene.terrain.terrain_generator.{attribute}",
            expected,
        )

    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    if not isinstance(sub_terrains, Mapping) or set(sub_terrains) != {"flat_online"}:
        raise ValueError(
            "FlatOnline deployment requires exactly one terrain subterrain named 'flat_online'"
        )
    flat_online = sub_terrains["flat_online"]
    for attribute, expected in (
        ("proportion", 1.0),
        ("seed", 0),
        ("step_height", 0.0),
        ("rough_amplitude", 0.0),
    ):
        _require_flat_online_attribute(
            flat_online,
            attribute,
            f"scene.terrain.terrain_generator.sub_terrains['flat_online'].{attribute}",
            expected,
        )

    try:
        command = env_cfg.commands.base_velocity
        command_ranges = command.ranges
        policy_observations = env_cfg.observations.policy
        curriculum = env_cfg.curriculum
        events = env_cfg.events
        terminations = env_cfg.terminations
    except AttributeError as error:
        raise ValueError(
            "FlatOnline deployment requires deterministic command, observation, "
            "curriculum, and event configuration"
        ) from error

    for attribute, expected in (
        ("resampling_time_range", (1.0e9, 1.0e9)),
        ("rel_standing_envs", 0.0),
        ("rel_heading_envs", 0.0),
        ("heading_command", False),
    ):
        _require_flat_online_attribute(
            command, attribute, f"commands.base_velocity.{attribute}", expected
        )
    for attribute, expected in (
        ("lin_vel_x", (0.5, 0.5)),
        ("lin_vel_y", (0.0, 0.0)),
        ("ang_vel_z", (0.0, 0.0)),
        ("heading", (0.0, 0.0)),
    ):
        _require_flat_online_attribute(
            command_ranges,
            attribute,
            f"commands.base_velocity.ranges.{attribute}",
            expected,
        )

    _require_flat_online_attribute(
        policy_observations,
        "enable_corruption",
        "observations.policy.enable_corruption",
        False,
    )
    for attribute in ("terrain_levels", "gait_level", "command_levels"):
        _require_flat_online_attribute(
            curriculum, attribute, f"curriculum.{attribute}", None
        )
    for attribute in (
        "randomize_apply_external_force_torque",
        "randomize_push_robot",
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia",
        "randomize_com_positions",
        "randomize_actuator_gains",
    ):
        _require_flat_online_attribute(events, attribute, f"events.{attribute}", None)

    try:
        root_reset_params = events.randomize_reset_base.params
        joint_reset_params = events.randomize_reset_joints.params
    except AttributeError as error:
        raise ValueError("FlatOnline deployment requires deterministic reset events") from error
    _require_flat_online_value(
        "events.randomize_reset_base.params",
        root_reset_params,
        deterministic_root_reset_params(),
    )
    if not isinstance(joint_reset_params, Mapping):
        raise ValueError(
            "FlatOnline deployment requires events.randomize_reset_joints.params mapping"
        )
    for name, expected in deterministic_joint_reset_params().items():
        _require_flat_online_value(
            f"events.randomize_reset_joints.params[{name!r}]",
            joint_reset_params.get(name),
            expected,
        )

    for attribute in ("time_out", "terrain_out_of_bounds", "illegal_contact"):
        if getattr(terminations, attribute, None) is None:
            raise ValueError(
                f"FlatOnline deployment requires terminations.{attribute}"
            )
    out_of_bounds_params = getattr(terminations.terrain_out_of_bounds, "params", None)
    if not isinstance(out_of_bounds_params, Mapping):
        raise ValueError(
            "FlatOnline deployment requires terminations.terrain_out_of_bounds.params mapping"
        )
    _require_flat_online_value(
        "terminations.terrain_out_of_bounds.params['distance_buffer']",
        out_of_bounds_params.get("distance_buffer"),
        0.5,
    )


def deterministic_root_reset_params() -> dict[str, dict[str, tuple[float, float]]]:
    zero_range = (0.0, 0.0)
    return {
        "pose_range": {
            "x": zero_range,
            "y": zero_range,
            "z": zero_range,
            "roll": zero_range,
            "pitch": zero_range,
            "yaw": zero_range,
        },
        "velocity_range": {
            "x": zero_range,
            "y": zero_range,
            "z": zero_range,
            "roll": zero_range,
            "pitch": zero_range,
            "yaw": zero_range,
        },
    }


def deterministic_joint_reset_params() -> dict[str, tuple[float, float]]:
    """Reset joints to their default pose without adding reset-state noise."""
    return {
        "position_range": (1.0, 1.0),
        "velocity_range": (0.0, 0.0),
    }


def assert_heldout_terrains_disabled(terrain_generator: Any) -> None:
    leaks = {
        name: float(terrain_generator.sub_terrains[name].proportion)
        for name in HELDOUT_TERRAIN_NAMES
        if float(terrain_generator.sub_terrains[name].proportion) > 0.0
    }
    if leaks:
        details = ", ".join(f"{name}={proportion:g}" for name, proportion in leaks.items())
        raise ValueError(f"held-out terrain leakage: {details}")
