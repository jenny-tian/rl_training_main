from __future__ import annotations

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
