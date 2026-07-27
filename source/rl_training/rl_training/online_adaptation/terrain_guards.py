from __future__ import annotations

from typing import Any

HELDOUT_TERRAIN_NAMES = ("boxes", "pyramid_stairs", "pyramid_stairs_inv")


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


def assert_heldout_terrains_disabled(terrain_generator: Any) -> None:
    leaks = {
        name: float(terrain_generator.sub_terrains[name].proportion)
        for name in HELDOUT_TERRAIN_NAMES
        if float(terrain_generator.sub_terrains[name].proportion) > 0.0
    }
    if leaks:
        details = ", ".join(f"{name}={proportion:g}" for name, proportion in leaks.items())
        raise ValueError(f"held-out terrain leakage: {details}")
