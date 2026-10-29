"""Opt-in predicate-time evidence for a single frozen-policy episode; CPU-safe."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .tile_bounds import source_tile_bounds_terminated
from .rigid_body_reference import validate_body_reference_state


def json_state(value):
    """Preserve nonfinite evidence as explicit strings, never invalid JSON/null."""
    if isinstance(value, torch.Tensor):
        return json_state(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: json_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_state(item) for item in value]
    return value


class TileBoundsRecorder:
    """Write the evaluated snapshot immediately, before any reset can occur."""

    def __init__(self, path: Path):
        self.path = path
        self.evaluations = 0

    def __call__(self, snapshot):
        self.evaluations += 1
        row = {
            "schema": "lite3_tile_bounds_predicate_v2" if "reference_points" in snapshot else "lite3_tile_bounds_predicate_v1",
            "evaluation_index": self.evaluations,
            "phase": "termination_predicate_pre_reset",
            "environment_ids": list(range(len(snapshot["terminated"]))),
            **json_state(snapshot),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def _float_tree(value):
    if isinstance(value, list):
        return [_float_tree(item) for item in value]
    return float(value)


def validate_tile_bounds_trace(path: Path, record: dict, expected_body_names: list[str], *, require_link_reference=False):
    """Reconstruct every captured predicate, preserving the original dtype.

    Diagnostic completeness is separate from policy/episode acceptance: a
    faithfully captured boundary failure is still a failed safety condition.
    """
    def require(condition, message):
        if not condition:
            raise ValueError("tile-bound telemetry: " + message)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == record["steps"], "missing or extra predicate evaluations")
    previous_counter = None
    reference_reviews = []
    schemas = {row.get("schema") for row in rows}
    require(len(schemas) <= 1, "mixed schemas")
    for step, row in enumerate(rows, 1):
        require(row["schema"] in ("lite3_tile_bounds_predicate_v1", "lite3_tile_bounds_predicate_v2"), "schema")
        require(not require_link_reference or row["schema"] == "lite3_tile_bounds_predicate_v2", "explicit link reference required")
        require(row["phase"] == "termination_predicate_pre_reset", "sampling phase")
        require(row["evaluation_index"] == step and row["episode_steps"] == [step], "step/reset alignment")
        counter = row["common_step_counter"]
        require(type(counter) is int and (previous_counter is None or counter == previous_counter + 1), "global step gap")
        previous_counter = counter
        require(row["environment_ids"] == [0], "single-environment scope")
        require(len(expected_body_names) == 17 and row["body_ids"] == list(range(17))
                and row["body_names"] == expected_body_names, "body mapping")
        require(row["state_dtype"] in ("torch.float32", "torch.float64"), "unsupported state dtype")
        dtype = getattr(torch, row["state_dtype"].split(".")[1])
        positions = torch.tensor(_float_tree(row["body_positions_w"]), dtype=dtype)
        velocities = torch.tensor(_float_tree(row["body_linear_velocities_w"]), dtype=dtype)
        require(positions.shape == velocities.shape == (1, 17, 3), "body state shape")
        if row["schema"] == "lite3_tile_bounds_predicate_v2":
            reference_reviews.append(validate_body_reference_state(row))
        origins = torch.tensor(_float_tree(row["terrain_origins_w"]), dtype=dtype)
        parameters = row["parameters"]
        require(parameters == dict(size_x=12.0, size_y=4.0, flat_length_m=3.0,
                                   collision_radius_m=0.27, numerical_margin_m=0.001,
                                   control_period_s=0.02), "frozen guard parameters")
        reconstructed = {}
        source_tile_bounds_terminated(
            positions, velocities, origins,
            torch.tensor(row["terrain_levels"], dtype=torch.long),
            torch.tensor(row["terrain_types"], dtype=torch.long),
            size=(parameters["size_x"], parameters["size_y"]),
            **{k: parameters[k] for k in ("flat_length_m", "collision_radius_m", "numerical_margin_m", "control_period_s")},
            diagnostics=reconstructed,
        )
        for key, expected in json_state(reconstructed).items():
            require(row[key] == expected, "predicate/input mismatch: " + key)
        should_terminate = step == len(rows) and "source_terrain_tile_bounds" in record["termination_terms"]
        require(row["terminated"] == [should_terminate], "predicate/episode termination mismatch")
    terminal = rows[-1] if rows else None
    hits = []
    if terminal:
        for branch, field in (("current", "outside_now_by_axis"), ("prediction", "outside_prediction_by_axis")):
            for body_id, flags in enumerate(terminal[field][0]):
                for axis, hit in zip(("x", "y"), flags, strict=True):
                    if hit:
                        hits.append(dict(branch=branch, body_id=body_id,
                                         body_name=terminal["body_names"][body_id], axis=axis))
    review = {
        "schema": "lite3_tile_bounds_trace_review_v2" if reference_reviews else "lite3_tile_bounds_trace_review_v1",
        "status": "complete_predicate_evidence",
        "evaluations_verified": len(rows),
        "body_count": len(expected_body_names),
        "terminal_boundary_flag": terminal["terminated"][0] if terminal else False,
        "terminal_geometric_hits": hits,
        "terminal_invalid_state": terminal["invalid_state"][0] if terminal else None,
        "terminal_valid_indices": terminal["valid_indices"][0] if terminal else None,
        "condition_completed": record["completed"],
    }
    if reference_reviews:
        review["reference_conversion"] = {
            "status": "verified" if all(r["status"] == "verified" for r in reference_reviews) else "nonfinite_reference_unverifiable",
            "bodies_verified": sum(r["bodies_verified"] for r in reference_reviews),
            "unverifiable_evaluations": [i for i, r in enumerate(reference_reviews, 1) if r["status"] != "verified"],
            "atol": reference_reviews[0]["atol"], "rtol": reference_reviews[0]["rtol"],
            "quaternion_norm_atol": reference_reviews[0]["quaternion_norm_atol"],
            **{key: max((r[key] for r in reference_reviews if r[key] is not None), default=None)
               for key in ("com_position_max_abs_error", "link_velocity_max_abs_error")},
        }
    return review
