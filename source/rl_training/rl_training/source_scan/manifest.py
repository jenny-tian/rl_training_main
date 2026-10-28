"""Executable schema and preflight checks for a frozen Source-only scan.

The v0.1 draft remains readable for diagnostics. Execution preflight requires
the explicit v0.2 contract; this module neither migrates nor authorizes it.
Only the standard library is used, and no checkpoint is deserialized.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import stat
from typing import Any


DRAFT_SCHEMA = "source_qualification_scan_manifest_v0.1"
EXECUTION_SCHEMA = "source_qualification_scan_manifest_v0.2"
SCAN_SPLIT = "source_qualification_scan"
DRAFT_STATUS = "draft_not_for_run"
LOCKED_STATUS = "locked_for_run_after_user_approval_and_bridge_qualification"
MAX_SEED = 2**63 - 1
DIRECTIONS = {
    "flat": ("forward_flat",),
    "multi_scale_rough": ("forward_lane",),
    "slope": ("uphill", "downhill", "cross_slope_left", "cross_slope_right"),
    "multi_step_stairs": ("ascending", "descending"),
    "slope_plus_rough": ("uphill", "downhill", "cross_slope_left", "cross_slope_right"),
    "friction_variation": ("left_right_bands", "front_back_bands", "alternating_regions"),
}
RESERVED_SPLITS = ("source_train", "source_validation", "development", "probe", "sealed")
GUARDS = {
    "read_only_checkpoint": True,
    "allow_actor_update": False,
    "allow_online_adaptation": False,
    "allow_sealed_access": False,
    "allow_development_selection_from_results": False,
    "scan_output_may_select_source_range_only": True,
}
# Each role identifies a file whose exact bytes must be bound before execution.
ARTIFACT_ROLES = (
    "environment_config", "geometry_generator", "manifest_validator", "scan_runner",
    "safety_filter", "safety_config", "observation_contract", "action_scale", "normalizer",
    "dynamics_config", "material_config", "reset_config", "termination_config", "metrics_config",
    "classification_config", "geometry_audit", "checkpoint_parity_report", "bridge_smoke_report",
    "user_approval_record",
)
REQUIRED_EPISODE_METRICS = (
    "completed", "termination_reason", "steps", "forward_progress_m", "max_forward_progress_m",
    "mean_and_rms_velocity_error_mps", "roll_mean_rms_peak_rad", "pitch_mean_rms_peak_rad",
    "lateral_offset_mean_peak_m", "nonfoot_contact_count_and_duration",
    "foot_slip_contact_fraction_and_tangential_speed", "safety_filter_projection_fraction",
    "safety_filter_projection_reasons", "safety_filter_rejection_or_emergency_count",
    "action_compute_and_execute_latency_ms", "invalid_state_count",
)
REQUIRED_FAMILY_METRICS = {
    "slope_specific": (
        "direction", "slope_angle_deg", "pitch_response", "roll_response", "transition_progress",
    ),
    "stairs_specific": (
        "direction", "step_count", "completed_step_index", "front_foot_collision_count",
        "stuck_or_catch_step_count", "swing_foot_clearance_m", "per_step_progress",
    ),
    "friction_specific": (
        "region_layout", "material_parameters", "actual_contact_friction", "slip_by_foot_and_region",
    ),
    "rough_specific": (
        "audited_local_peak_valley_m", "spatial_scale_statistics", "adjacent_footfall_height_difference_m",
    ),
}
QUALIFICATION_BINDINGS = {
    "geometry_audit": ("geometry_generator",),
    "checkpoint_parity_report": ("base_checkpoint", "observation_contract", "action_scale", "normalizer"),
    "bridge_smoke_report": ("base_checkpoint", "environment_config", "geometry_generator", "scan_runner",
                            "safety_filter", "safety_config", "material_config", "dynamics_config",
                            "reset_config", "termination_config", "metrics_config"),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLACEHOLDER = re.compile(r"\b(pending|tbd|todo|placeholder|to be filled|not_implemented)\b|待定|待填写|未锁定", re.I)


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ManifestValidation:
    issues: tuple[ValidationIssue, ...]
    cell_count: int
    episode_count: int
    files_verified: bool = False

    @property
    def schema_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def lock_fields_complete(self) -> bool:
        return not self.issues

    @property
    def execution_ready(self) -> bool:
        # A structural audit never represents a completed runtime preflight.
        return self.lock_fields_complete and self.files_verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_valid": self.schema_valid,
            "lock_fields_complete": self.lock_fields_complete,
            "files_verified": self.files_verified,
            "execution_ready": self.execution_ready,
            "cell_count": self.cell_count,
            "episode_count": self.episode_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


class ManifestRejected(ValueError):
    def __init__(self, report: ManifestValidation):
        self.report = report
        super().__init__("Source scan manifest rejected: " + "; ".join(
            f"{i.path}: {i.message}" for i in report.issues
        ))


def _parse_manifest_bytes(content: bytes) -> dict[str, Any]:
    """Parse strict JSON; reject duplicate keys and non-finite constants."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite JSON number: {value}")
        return result

    result = json.loads(content,
                        object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError("manifest must be a JSON object")
    return result


def load_manifest(path: str | Path) -> dict[str, Any]:
    return _parse_manifest_bytes(Path(path).read_bytes())


def manifest_payload_sha256(manifest: dict[str, Any]) -> str:
    """Canonical approval payload, excluding the approval and its own file record.

    Excluding those two records avoids a self-referential hash. All policy,
    split, seed, parameter, qualification and other artifact records are bound.
    This checksum is an integrity binding, not an authorization signature.
    """
    payload = copy.deepcopy(manifest)
    contract = payload.get("execution_contract", {})
    if isinstance(contract, dict):
        contract.pop("user_approval", None)
        artifacts = contract.get("artifacts", {})
        if isinstance(artifacts, dict):
            artifacts.pop("user_approval_record", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _Validator:
    def __init__(self):
        self.issues: list[ValidationIssue] = []
        self.cells = 0
        self.episodes = 0

    def issue(self, path, code, message, severity="error"):
        self.issues.append(ValidationIssue(severity, code, path, message))

    def obj(self, value, path):
        if not isinstance(value, dict):
            self.issue(path, "object_required", "must be an object")
            return {}
        return value

    def array(self, value, path):
        if not isinstance(value, list):
            self.issue(path, "array_required", "must be an array")
            return []
        return value

    def exact(self, value, expected, path, severity="error"):
        if type(value) is not type(expected) or value != expected:
            self.issue(path, "fixed_contract", f"must equal {expected!r}", severity)

    def number(self, value, path, *, positive=False):
        valid = type(value) in (int, float)
        try:
            valid = valid and math.isfinite(value) and (not positive or value > 0)
        except (OverflowError, TypeError):
            valid = False
        if not valid:
            self.issue(path, "finite_number", "must be a finite number" + (" > 0" if positive else ""))
        return valid

    def integer(self, value, path, *, low=0, high=MAX_SEED):
        valid = type(value) is int and low <= value <= high
        if not valid:
            self.issue(path, "integer_range", f"must be an integer in [{low}, {high}]")
        return valid

    def interval(self, value, path):
        if (not isinstance(value, list) or len(value) != 2
                or not all(type(v) is int and 0 <= v <= MAX_SEED for v in value)
                or value[0] > value[1]):
            self.issue(path, "seed_interval", "must be an inclusive ordered pair of integer seeds")
            return None
        return tuple(value)

    def file_record(self, value, path):
        record = self.obj(value, path)
        if not isinstance(record.get("path"), str) or not Path(record["path"]).is_absolute():
            self.issue(path + ".path", "absolute_path", "must be an absolute file path")
        if not isinstance(record.get("sha256"), str) or not _SHA256.fullmatch(record["sha256"]):
            self.issue(path + ".sha256", "sha256", "must contain 64 lowercase hex digits")

    def placeholders(self, value, path):
        if value is None or isinstance(value, str) and (not value.strip() or _PLACEHOLDER.search(value)):
            self.issue(path, "unresolved_field", "locked fields cannot contain null, blank or placeholder values")
        elif isinstance(value, dict):
            for key, item in value.items():
                self.placeholders(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.placeholders(item, f"{path}[{index}]")
        elif type(value) in (int, float):
            self.number(value, path)

    def threshold_fraction(self, value, path):
        if not self.number(value, path) or not 0 <= value <= 1:
            self.issue(path, "threshold_fraction", "must be a finite fraction in [0, 1]")

    def report(self, *, files_verified=False):
        return ManifestValidation(tuple(self.issues), self.cells, self.episodes, files_verified)


_MOTION_QUALITY_THRESHOLDS = {
    "mean_velocity_error_mps": "positive",
    "rms_velocity_error_mps": "positive",
    "roll_peak_rad": "positive",
    "pitch_peak_rad": "positive",
    "lateral_offset_peak_m": "positive",
    "foot_slip_contact_fraction": "fraction",
    "safety_filter_projection_fraction": "fraction",
}


def _check_threshold_values(v, mapping, path):
    if set(mapping) != set(_MOTION_QUALITY_THRESHOLDS):
        v.issue(path, "threshold_metrics", "all registered motion-quality thresholds must be specified exactly once")
        return
    for name, kind in _MOTION_QUALITY_THRESHOLDS.items():
        value = mapping[name]
        item_path = path + "." + name
        if kind == "fraction":
            v.threshold_fraction(value, item_path)
        elif not v.number(value, item_path, positive=True):
            continue


def _check_classification_thresholds(v, thresholds, m):
    """Validate the locked classification contract without selecting cutoffs."""
    retain = thresholds["retain_or_low_share"]
    retain_path = "execution_contract.classification_thresholds.retain_or_low_share"
    retain_keys = {"minimum_stable_instance_fraction", "motion_quality_maxima"}
    if not isinstance(retain, dict) or set(retain) != retain_keys:
        v.issue(retain_path, "threshold_rule", f"exact keys required: {sorted(retain_keys)}")
    else:
        v.threshold_fraction(retain["minimum_stable_instance_fraction"], retain_path + ".minimum_stable_instance_fraction")
        maxima = v.obj(retain["motion_quality_maxima"], retain_path + ".motion_quality_maxima")
        _check_threshold_values(v, maxima, retain_path + ".motion_quality_maxima")

    candidate = thresholds["source_candidate"]
    candidate_path = "execution_contract.classification_thresholds.source_candidate"
    candidate_keys = {"minimum_recoverable_failure_fraction", "motion_quality_degradation_cutoffs", "safe_execution_required"}
    if not isinstance(candidate, dict) or set(candidate) != candidate_keys:
        v.issue(candidate_path, "threshold_rule", f"exact keys required: {sorted(candidate_keys)}")
    else:
        v.threshold_fraction(candidate["minimum_recoverable_failure_fraction"], candidate_path + ".minimum_recoverable_failure_fraction")
        cutoffs = v.obj(candidate["motion_quality_degradation_cutoffs"], candidate_path + ".motion_quality_degradation_cutoffs")
        _check_threshold_values(v, cutoffs, candidate_path + ".motion_quality_degradation_cutoffs")
        v.exact(candidate["safe_execution_required"], True, candidate_path + ".safe_execution_required")

    exclude = thresholds["not_source_main_domain"]
    exclude_path = "execution_contract.classification_thresholds.not_source_main_domain"
    exclude_keys = {"minimum_unrecoverable_failure_fraction", "near_zero_forward_progress_m_max", "safety_qualification_failure_excludes"}
    if not isinstance(exclude, dict) or set(exclude) != exclude_keys:
        v.issue(exclude_path, "threshold_rule", f"exact keys required: {sorted(exclude_keys)}")
    else:
        v.threshold_fraction(exclude["minimum_unrecoverable_failure_fraction"], exclude_path + ".minimum_unrecoverable_failure_fraction")
        progress = exclude["near_zero_forward_progress_m_max"]
        if not v.number(progress, exclude_path + ".near_zero_forward_progress_m_max") or progress < 0:
            v.issue(exclude_path + ".near_zero_forward_progress_m_max", "progress_threshold", "must be a finite nonnegative distance")
        v.exact(exclude["safety_qualification_failure_excludes"], True, exclude_path + ".safety_qualification_failure_excludes")

    uncertainty = thresholds["sample_count_and_uncertainty"]
    uncertainty_path = "execution_contract.classification_thresholds.sample_count_and_uncertainty"
    uncertainty_keys = {"episodes_per_cell", "confidence_level", "interval_method"}
    if not isinstance(uncertainty, dict) or set(uncertainty) != uncertainty_keys:
        v.issue(uncertainty_path, "threshold_rule", f"exact keys required: {sorted(uncertainty_keys)}")
    else:
        episode_protocol = m.get("episode_protocol")
        expected_count = episode_protocol.get("episodes_per_cell") if isinstance(episode_protocol, dict) else None
        v.exact(uncertainty["episodes_per_cell"], expected_count, uncertainty_path + ".episodes_per_cell")
        level = uncertainty["confidence_level"]
        if not v.number(level, uncertainty_path + ".confidence_level") or not 0 < level < 1:
            v.issue(uncertainty_path + ".confidence_level", "confidence_level", "must be between 0 and 1")
        method = uncertainty["interval_method"]
        if not isinstance(method, str) or not method.strip() or _PLACEHOLDER.search(method):
            v.issue(uncertainty_path + ".interval_method", "uncertainty_method", "must name the pre-registered uncertainty method")

def _check_boundary(v, m):
    b = v.obj(m.get("research_boundary"), "research_boundary")
    for name, expected in (("control_frequency_hz", 50), ("control_period_s", 0.02),
                           ("max_steps_per_episode", 2000), ("max_duration_s", 40.0),
                           ("target_speed_mps", 0.3)):
        value = b.get(name)
        if v.number(value, "research_boundary." + name) and value != expected:
            v.issue("research_boundary." + name, "fixed_contract", f"must equal {expected}")
    v.exact(b.get("online_update"), False, "research_boundary.online_update")
    command = v.obj(b.get("command"), "research_boundary.command")
    for name, expected in (("lin_vel_x_mps", 0.3), ("lin_vel_y_mps", 0.0), ("ang_vel_z_rps", 0.0)):
        if v.number(command.get(name), "research_boundary.command." + name) and command[name] != expected:
            v.issue("research_boundary.command." + name, "fixed_command", f"must equal {expected}")
    if m.get("schema_version") == EXECUTION_SCHEMA:
        for name, expected in (("observation_dim", 45), ("action_dim", 12), ("height_scan", False),
                               ("privileged_observations", False), ("checkpoint_selection", "none")):
            v.exact(b.get(name), expected, "research_boundary." + name)
        v.exact(b.get("dynamics_profile"), "deployment_matched_frozen", "research_boundary.dynamics_profile")
    if b.get("dynamics_profile") != "deployment_matched_frozen":
        v.issue("research_boundary.dynamics_profile", "dynamics_semantics",
                "first scan must explicitly use deployment_matched_frozen; legacy dynamics seeds are not independent perturbations",
                "blocker")
    guards = v.obj(m.get("data_isolation_and_execution_guards"), "data_isolation_and_execution_guards")
    for name, expected in GUARDS.items():
        v.exact(guards.get(name), expected, "data_isolation_and_execution_guards." + name)
    v.exact(guards.get("required_status_for_execution"), LOCKED_STATUS,
            "data_isolation_and_execution_guards.required_status_for_execution")
    if DRAFT_STATUS not in v.array(guards.get("runner_must_refuse_status"),
                                   "data_isolation_and_execution_guards.runner_must_refuse_status"):
        v.issue("data_isolation_and_execution_guards.runner_must_refuse_status", "draft_guard", "must refuse draft_not_for_run")
    protocol = v.obj(m.get("episode_protocol"), "episode_protocol")
    v.integer(protocol.get("episodes_per_cell"), "episode_protocol.episodes_per_cell", low=1)
    v.exact(protocol.get("partial_progress_is_success"), False, "episode_protocol.partial_progress_is_success")
    if m.get("schema_version") == EXECUTION_SCHEMA:
        v.exact(protocol.get("early_stop"), "registered_safety_stop_only", "episode_protocol.early_stop")
        v.exact(protocol.get("additional_samples"), "forbidden", "episode_protocol.additional_samples")
    completion = v.obj(m.get("completion_and_termination"), "completion_and_termination")
    v.exact(completion.get("stable_completion_hold_steps"), 50, "completion_and_termination.stable_completion_hold_steps")
    if m.get("schema_version") == EXECUTION_SCHEMA:
        metrics = v.obj(m.get("metrics"), "metrics")
        episode_metrics = v.array(metrics.get("per_episode_required"), "metrics.per_episode_required")
        if (any(not isinstance(item, str) for item in episode_metrics)
                or len(episode_metrics) != len(REQUIRED_EPISODE_METRICS)
                or set(episode_metrics) != set(REQUIRED_EPISODE_METRICS)):
            v.issue("metrics.per_episode_required", "episode_metrics", "all registered per-episode metrics must be present exactly once")
        for key, required in REQUIRED_FAMILY_METRICS.items():
            values = v.array(metrics.get(key), "metrics." + key)
            if (any(not isinstance(item, str) for item in values)
                    or len(values) != len(required) or set(values) != set(required)):
                v.issue("metrics." + key, "family_metrics",
                        "all registered family-specific diagnostic metrics must be present exactly once")
        v.exact(metrics.get("aggregation"), "per_cell_and_per_family_no_direction_or_level_hidden", "metrics.aggregation")


def _check_seeds_and_cells(v, m):
    plan = v.obj(m.get("seed_plan"), "seed_plan")
    legacy = m.get("schema_version") == DRAFT_SCHEMA
    role = "dynamics" if legacy else "initial_state"
    if legacy:
        v.issue("seed_plan.dynamics_partition", "ambiguous_seed_role",
                "v0.2 must register initial_state seeds explicitly; do not silently relabel dynamics seeds", "blocker")
    elif "dynamics_partition" in plan:
        v.issue("seed_plan.dynamics_partition", "ambiguous_seed_role", "fixed-dynamics scan cannot declare dynamics seed partitions")
    partitions = {}
    for key in ("geometry_partition", role + "_partition"):
        interval = v.interval(plan.get(key), "seed_plan." + key)
        if interval:
            partitions[key] = interval
    for split in RESERVED_SPLITS:
        key = split + "_reserved_partition"
        if key not in plan and legacy:
            v.issue("seed_plan." + key, "missing_reserved_partition", "must reserve independent geometry and initial-state seeds", "blocker")
            continue
        reserved = v.obj(plan.get(key), "seed_plan." + key)
        for axis in ("geometry", role):
            interval = v.interval(reserved.get(axis), f"seed_plan.{key}.{axis}")
            if interval:
                partitions[key + "." + axis] = interval
    entries = list(partitions.items())
    for i, (name, (lo, hi)) in enumerate(entries):
        for other, (a, b) in entries[i + 1:]:
            if max(lo, a) <= min(hi, b):
                v.issue("seed_plan." + name, "partition_overlap", f"overlaps {other}")
    cells = v.array(plan.get("cells"), "seed_plan.cells")
    v.cells = len(cells)
    v.exact(plan.get("cell_count"), len(cells), "seed_plan.cell_count")
    seen_ids, seen_coverage, used = set(), set(), {"geometry": [], role: []}
    required = {(family, level, direction) for family, directions in DIRECTIONS.items()
                for level in (1, 2, 3) for direction in directions}
    for index, cell in enumerate(cells):
        path = f"seed_plan.cells[{index}]"
        c = v.obj(cell, path)
        cid = c.get("cell_id")
        if not isinstance(cid, str) or not cid.strip() or cid in seen_ids:
            v.issue(path + ".cell_id", "cell_id", "cell IDs must be nonempty and unique")
        else:
            seen_ids.add(cid)
        family, level, direction = c.get("terrain_family"), c.get("level"), c.get("direction_or_layout")
        if not isinstance(family, str) or family not in DIRECTIONS:
            v.issue(path + ".terrain_family", "source_family", "must name one of the six Source families")
        elif (v.integer(level, path + ".level", low=1, high=3)
              and isinstance(direction, str) and direction in DIRECTIONS[family]):
            coverage = (family, level, direction)
            if coverage in seen_coverage:
                v.issue(path, "duplicate_scan_condition", "family/level/direction must be unique")
            seen_coverage.add(coverage)
        else:
            v.issue(path + ".direction_or_layout", "source_direction", "direction/layout does not match the registered family")
        episodes = c.get("episodes")
        if v.integer(episodes, path + ".episodes", low=1):
            v.episodes += episodes
            v.exact(episodes, m.get("episode_protocol", {}).get("episodes_per_cell")
                    if isinstance(m.get("episode_protocol"), dict) else None, path + ".episodes")
        for axis in ("geometry", role):
            interval = v.interval(c.get(axis + "_seed_range"), path + "." + axis + "_seed_range")
            if interval:
                lo, hi = interval
                parent = partitions.get(axis + "_partition")
                if parent and not (parent[0] <= lo <= hi <= parent[1]):
                    v.issue(path, "seed_outside_partition", f"{axis} seeds must stay inside scan partition")
                if type(episodes) is int and hi - lo + 1 != episodes:
                    v.issue(path, "seed_episode_count", f"{axis} seed count must equal episodes")
                for previous, a, b in used[axis]:
                    if max(lo, a) <= min(hi, b):
                        v.issue(path, "cell_seed_reuse", f"{axis} seeds overlap {previous}")
                used[axis].append((cid, lo, hi))
        if not legacy:
            if "dynamics_seed_range" in c:
                v.issue(path + ".dynamics_seed_range", "ambiguous_seed_role", "fixed dynamics cannot vary via episode seeds")
            v.exact(c.get("seed_pairing"), "zip_geometry_initial_state", path + ".seed_pairing")
            v.exact(c.get("dynamics_profile"), "deployment_matched_frozen", path + ".dynamics_profile")
            _check_resolved_cell(v, c, path)
        elif "resolved_parameters" not in c:
            v.issue(path + ".resolved_parameters", "unlocked_cell", "candidate parameters are not an executable geometry/reset/track contract", "blocker")
    missing = required - seen_coverage
    if missing:
        v.issue("seed_plan.cells", "incomplete_coverage", f"missing {len(missing)} required family/level/direction conditions")
    if not legacy:
        v.exact(plan.get("total_episodes"), v.episodes, "seed_plan.total_episodes")


def _check_resolved_cell(v, c, path):
    r = v.obj(c.get("resolved_parameters"), path + ".resolved_parameters")
    v.placeholders(r, path + ".resolved_parameters")
    if "candidate_parameters" in c:
        v.issue(path + ".candidate_parameters", "candidate_not_resolved", "execution schema requires resolved_parameters instead of candidate parameters")
    geometry = v.obj(r.get("geometry"), path + ".resolved_parameters.geometry")
    family = c.get("terrain_family")
    common = {"size", "flat_length_m", "transition_length_m", "grid_spacing_m"}
    extra = set()
    if family in ("multi_scale_rough", "slope_plus_rough"):
        extra |= {"rough_peak_valley_m", "rough_audit_window_m", "spatial_scales_m"}
    if family in ("slope", "slope_plus_rough"):
        extra.add("slope_angle_deg")
    if family in ("multi_scale_rough", "slope", "slope_plus_rough"):
        extra.add("exit_flat_length_m")
    if family == "multi_step_stairs":
        extra |= {"step_count", "step_height_m", "tread_depth_m"}
    if family == "friction_variation":
        extra |= {"friction_range", "friction_region_count"}
    directional = family in ("slope", "slope_plus_rough", "multi_step_stairs", "friction_variation")
    expected_geometry_keys = common | extra | ({"direction_or_layout"} if directional else set())
    if set(geometry) != expected_geometry_keys:
        v.issue(path + ".resolved_parameters.geometry", "geometry_fields", f"exact keys required: {sorted(expected_geometry_keys)}")
    if directional and geometry.get("direction_or_layout") != c.get("direction_or_layout"):
        v.issue(path + ".resolved_parameters.geometry.direction_or_layout", "direction_mismatch", "must match the registered cell direction/layout")
    for name in (common | extra) - {"size", "spatial_scales_m", "step_count", "friction_region_count", "friction_range"}:
        v.number(geometry.get(name), path + ".geometry." + name, positive=True)
    size = geometry.get("size")
    size_valid = isinstance(size, list) and len(size) == 2
    if not size_valid:
        v.issue(path + ".geometry.size", "track_size", "size must be [length, width]")
    else:
        size_valid = all([v.number(value, path + ".geometry.size", positive=True) for value in size])
    if family in ("multi_scale_rough", "slope_plus_rough"):
        scales = geometry.get("spatial_scales_m")
        if (not isinstance(scales, list) or len(scales) < 2
                or not all(_finite_number(x) and x > 0 for x in scales)
                or len(set(scales)) < 2):
            v.issue(path + ".geometry.spatial_scales_m", "rough_scales", "at least two distinct finite positive scales are required")
    if family in ("slope", "slope_plus_rough"):
        angle = geometry.get("slope_angle_deg")
        if type(angle) in (int, float) and angle >= 45:
            v.issue(path + ".geometry.slope_angle_deg", "slope_range", "angle must be below 45 degrees")
    if family in ("multi_scale_rough", "slope", "slope_plus_rough"):
        v.number(geometry.get("exit_flat_length_m"), path + ".geometry.exit_flat_length_m", positive=True)
    if family == "multi_step_stairs":
        v.integer(geometry.get("step_count"), path + ".geometry.step_count", low=3, high=6)
        v.number(geometry.get("step_height_m"), path + ".geometry.step_height_m", positive=True)
        v.number(geometry.get("tread_depth_m"), path + ".geometry.tread_depth_m", positive=True)
        if (_finite_number(geometry.get("step_height_m")) and type(geometry.get("step_count")) is int
                and geometry["step_height_m"] * geometry["step_count"] >= 1.0):
            v.issue(path + ".geometry", "stair_total_height", "total stair height must remain below the audited support-depth bound")
    if family == "friction_variation":
        v.integer(geometry.get("friction_region_count"), path + ".geometry.friction_region_count", low=2, high=256)
        friction = geometry.get("friction_range")
        if (not isinstance(friction, list) or len(friction) != 2
                or not all(_finite_number(x) and x > 0 for x in friction)
                or friction[0] > friction[1]):
            v.issue(path + ".geometry.friction_range", "friction_range", "must be an ordered positive finite pair of candidate material coefficients")
    reset = v.obj(r.get("reset"), path + ".resolved_parameters.reset")
    if set(reset) != {"profile_id", "config_sha256"}:
        v.issue(path + ".reset", "reset_fields", "reset must name profile_id and config_sha256")
    if not isinstance(reset.get("profile_id"), str) or not reset.get("profile_id", "").strip():
        v.issue(path + ".reset.profile_id", "reset_profile", "must identify a profile in the locked reset config")
    track = v.obj(r.get("track"), path + ".resolved_parameters.track")
    if set(track) != {"start_x_m", "completion_x_m", "lateral_margin_m"}:
        v.issue(path + ".track", "track_fields", "track requires start_x_m, completion_x_m, lateral_margin_m")
    valid = all([v.number(track.get(name), path + ".track." + name) for name in
                 ("start_x_m", "completion_x_m", "lateral_margin_m")])
    if valid and size_valid:
        start, end, margin = (track[k] for k in ("start_x_m", "completion_x_m", "lateral_margin_m"))
        if not (0 <= start < end <= size[0] and 0 <= margin < size[1] / 2):
            v.issue(path + ".track", "track_bounds", "start/completion/margin must lie within the track")
        if (end - start) / 0.3 + 50 * 0.02 > 2000 * 0.02:
            v.issue(path + ".track", "unreachable_completion", "nominal travel plus completion hold exceeds the registered episode duration")
        flat, transition, grid = (geometry.get(k) for k in ("flat_length_m", "transition_length_m", "grid_spacing_m"))
        if all(_finite_number(x) and x > 0 for x in (flat, transition, grid)):
            if flat + transition >= size[0]:
                v.issue(path + ".geometry", "challenge_length", "flat/transition must leave a nonempty challenge segment")
            if start != flat / 2:
                v.issue(path + ".track.start_x_m", "start_origin", "must match the generator origin at flat_length_m / 2")
            if family == "multi_step_stairs":
                count, depth = geometry.get("step_count"), geometry.get("tread_depth_m")
                if type(count) is int and _finite_number(depth):
                    stair_end = flat + transition + count * depth
                    if stair_end >= size[0] or end <= stair_end:
                        v.issue(path + ".track", "stairs_completion", "completion must follow all treads and leave an exit segment")
            if family in ("slope", "slope_plus_rough"):
                exit_flat = geometry.get("exit_flat_length_m")
                if _finite_number(exit_flat) and exit_flat > 0:
                    exit_start = size[0] - exit_flat - transition
                    flat_start = size[0] - exit_flat
                    if flat + transition >= exit_start:
                        v.issue(path + ".geometry.exit_flat_length_m", "slope_exit_segment", "slope exit must leave a nonempty challenge segment")
                    if end < flat_start:
                        v.issue(path + ".track.completion_x_m", "slope_completion_before_flat", "completion must be on the terminal flat buffer after the slope exit transition")
                    hold_distance = 50 * 0.02 * 0.3
                    if size[0] - end < hold_distance + 0.4:
                        v.issue(path + ".track.completion_x_m", "slope_hold_buffer", "terminal flat must fit the nominal stable hold distance and 0.4 m body-clearance margin")
            if family in ("multi_scale_rough", "slope_plus_rough"):
                window = geometry.get("rough_audit_window_m")
                if _finite_number(window) and window > min(size[1], size[0] - flat - transition):
                    v.issue(path + ".geometry.rough_audit_window_m", "audit_window", "window must fit inside the rough challenge region")


def _check_lock(v, m):
    v.exact(m.get("status"), LOCKED_STATUS, "status", "blocker")
    v.exact(m.get("execution_authorized"), True, "execution_authorized", "blocker")
    v.exact(m.get("split"), SCAN_SPLIT, "split", "blocker")
    if m.get("schema_version") != EXECUTION_SCHEMA:
        v.issue("schema_version", "execution_schema_required", "legacy v0.1 is audit-only; explicit v0.2 lock is required", "blocker")
    contract = m.get("execution_contract")
    if not isinstance(contract, dict):
        v.issue("execution_contract", "missing_execution_contract", "geometry, materials, safety, hashes and qualifications must be locked", "blocker")
        return
    # Validate data separately below: the manifest embeds an immutable approval
    # statement for the specific content hash, not just a boolean switch.
    v.placeholders({k: value for k, value in contract.items() if k != "user_approval"}, "execution_contract")
    artifacts = v.obj(contract.get("artifacts"), "execution_contract.artifacts")
    if set(artifacts) != set(ARTIFACT_ROLES):
        v.issue("execution_contract.artifacts", "artifact_roles", f"exact artifact roles required: {list(ARTIFACT_ROLES)}")
    for role in ARTIFACT_ROLES:
        v.file_record(artifacts.get(role), "execution_contract.artifacts." + role)
    for key in ("geometry_audit_passed", "checkpoint_parity_passed", "bridge_smoke_passed", "region_material_binding_verified"):
        v.exact(contract.get(key), True, "execution_contract." + key)
    v.exact(contract.get("output_directory_policy"), "new_non_overwriting_metrics_only", "execution_contract.output_directory_policy")
    v.exact(contract.get("classification_thresholds_locked"), True, "execution_contract.classification_thresholds_locked")
    thresholds = v.obj(contract.get("classification_thresholds"), "execution_contract.classification_thresholds")
    required_thresholds = {"retain_or_low_share", "source_candidate", "not_source_main_domain", "sample_count_and_uncertainty"}
    if set(thresholds) != required_thresholds:
        v.issue("execution_contract.classification_thresholds", "classification_thresholds", f"exact rule keys required: {sorted(required_thresholds)}")
    else:
        _check_classification_thresholds(v, thresholds, m)
    approval = v.obj(contract.get("user_approval"), "execution_contract.user_approval")
    for name in ("manifest_id", "manifest_sha256", "approval_reference"):
        if not isinstance(approval.get(name), str) or not approval[name].strip():
            v.issue("execution_contract.user_approval." + name, "approval_binding", "must identify this approval and locked manifest")
    try:
        datetime.strptime(approval.get("approved_at_utc", ""), "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        v.issue("execution_contract.user_approval.approved_at_utc", "approval_timestamp", "must be an explicit UTC timestamp")
    v.exact(approval.get("scope"), "frozen_source_scan_only", "execution_contract.user_approval.scope")
    if approval.get("manifest_id") != m.get("manifest_id"):
        v.issue("execution_contract.user_approval.manifest_id", "approval_manifest_mismatch", "must match manifest_id")
    if not isinstance(approval.get("manifest_sha256"), str) or not _SHA256.fullmatch(approval["manifest_sha256"]):
        v.issue("execution_contract.user_approval.manifest_sha256", "approval_hash", "must bind the canonical locked manifest approval payload")
    reset = artifacts.get("reset_config", {})
    reset_hash = reset.get("sha256") if isinstance(reset, dict) else None
    plan = m.get("seed_plan", {})
    cells = plan.get("cells", []) if isinstance(plan, dict) else []
    for index, c in enumerate(cells if isinstance(cells, list) else []):
        if not isinstance(c, dict) or not isinstance(c.get("resolved_parameters"), dict):
            continue
        r = c["resolved_parameters"].get("reset", {})
        if isinstance(r, dict) and r.get("config_sha256") != reset_hash:
            v.issue(f"seed_plan.cells[{index}].resolved_parameters.reset.config_sha256", "reset_hash_mismatch", "must match the registered reset_config artifact")


def audit_manifest(manifest: object) -> ManifestValidation:
    """Inspect structure/lock completeness without reading any referenced files."""

    v = _Validator()
    m = v.obj(manifest, "$")
    try:
        json.dumps(m, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        v.issue("$", "json_values", "must contain only finite JSON data")
        return v.report()
    if m.get("schema_version") not in (DRAFT_SCHEMA, EXECUTION_SCHEMA):
        v.issue("schema_version", "schema_version", "unsupported Source scan schema version")
    if not isinstance(m.get("manifest_id"), str) or not m.get("manifest_id", "").strip():
        v.issue("manifest_id", "manifest_id", "must be a nonempty string")
    if m.get("status") not in (DRAFT_STATUS, LOCKED_STATUS):
        v.issue("status", "status", "must be draft_not_for_run or the registered locked status")
    if type(m.get("execution_authorized")) is not bool:
        v.issue("execution_authorized", "boolean_required", "must be a JSON boolean")
    if m.get("status") == DRAFT_STATUS and m.get("execution_authorized") is True:
        v.issue("execution_authorized", "draft_authorized", "a draft cannot authorize execution")
    # A declared non-Source split is invalid even in diagnostic mode.
    if "split" in m and m["split"] != SCAN_SPLIT:
        v.issue("split", "source_split_only", "development/sealed/training splits are not valid scan inputs")
    v.file_record(m.get("base_checkpoint"), "base_checkpoint")
    _check_boundary(v, m)
    families = v.array(m.get("terrain_families"), "terrain_families")
    names = [x.get("name") for x in families if isinstance(x, dict)]
    if len(names) != len(DIRECTIONS) or any(not isinstance(n, str) for n in names) or set(names) != set(DIRECTIONS):
        v.issue("terrain_families", "six_source_families", "must list each of the six Source families once")
    shares = [x.get("pilot_training_share") for x in families if isinstance(x, dict)]
    if len(shares) == len(DIRECTIONS):
        if not all(_finite_number(x) and 0 <= x <= 1 for x in shares) or not math.isclose(sum(shares), 1.0, abs_tol=1e-9):
            v.issue("terrain_families", "training_share", "six registered pilot shares must be finite, within [0, 1], and sum to 1")
    _check_seeds_and_cells(v, m)
    _check_lock(v, m)
    if m.get("schema_version") == EXECUTION_SCHEMA and m.get("current_implementation_blockers", []):
        v.issue("current_implementation_blockers", "implementation_blockers", "execution schema cannot retain unresolved implementation blockers")
    return v.report()


def audit_preparation_manifest(manifest: object) -> ManifestValidation:
    """Audit a complete v0.2 proposal without authorizing or preparing a run.

    Unlike :func:`audit_manifest`, this mode accepts the explicit draft state
    and reports absent runtime qualifications as blockers. It never reads the
    referenced files, changes authorization, or treats proposed cutoffs as
    user-approved values.
    """

    v = _Validator()
    m = v.obj(manifest, "$")
    try:
        json.dumps(m, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        v.issue("$", "json_values", "must contain only finite JSON data")
        return v.report()

    if m.get("schema_version") != EXECUTION_SCHEMA:
        v.issue("schema_version", "execution_schema_required", "preparation audit requires a v0.2 Source scan manifest")
    if not isinstance(m.get("manifest_id"), str) or not m.get("manifest_id", "").strip():
        v.issue("manifest_id", "manifest_id", "must be a nonempty string")
    v.exact(m.get("status"), DRAFT_STATUS, "status")
    v.exact(m.get("execution_authorized"), False, "execution_authorized")
    v.exact(m.get("split"), SCAN_SPLIT, "split")
    v.file_record(m.get("base_checkpoint"), "base_checkpoint")
    _check_boundary(v, m)

    families = v.array(m.get("terrain_families"), "terrain_families")
    names = [x.get("name") for x in families if isinstance(x, dict)]
    if len(names) != len(DIRECTIONS) or any(not isinstance(n, str) for n in names) or set(names) != set(DIRECTIONS):
        v.issue("terrain_families", "six_source_families", "must list each of the six Source families once")
    shares = [x.get("pilot_training_share") for x in families if isinstance(x, dict)]
    if len(shares) == len(DIRECTIONS):
        if not all(_finite_number(x) and 0 <= x <= 1 for x in shares) or not math.isclose(sum(shares), 1.0, abs_tol=1e-9):
            v.issue("terrain_families", "training_share", "six registered pilot shares must sum to 1")
    _check_seeds_and_cells(v, m)

    contract = v.obj(m.get("execution_contract_draft"), "execution_contract_draft")
    artifacts = v.obj(contract.get("artifacts"), "execution_contract_draft.artifacts")
    if set(artifacts) != set(ARTIFACT_ROLES):
        v.issue("execution_contract_draft.artifacts", "artifact_roles", f"exact artifact roles required: {list(ARTIFACT_ROLES)}")
    for role in ARTIFACT_ROLES:
        path = f"execution_contract_draft.artifacts.{role}"
        record = artifacts.get(role)
        if isinstance(record, dict) and record.get("available") is False:
            if not isinstance(record.get("reason"), str) or not record["reason"].strip():
                v.issue(path + ".reason", "artifact_missing_reason", "unavailable artifacts require a concrete reason")
            else:
                v.issue(path, "artifact_unavailable", record["reason"], "blocker")
        else:
            v.file_record(record, path)

    qualifications = v.obj(contract.get("qualifications"), "execution_contract_draft.qualifications")
    qualification_keys = {
        "geometry_audit_passed", "checkpoint_parity_passed", "policy_safety_bridge_passed",
        "region_material_binding_verified",
    }
    if set(qualifications) != qualification_keys:
        v.issue("execution_contract_draft.qualifications", "qualification_fields", f"exact keys required: {sorted(qualification_keys)}")
    for name in sorted(qualification_keys):
        value = qualifications.get(name)
        if type(value) is not bool:
            v.issue(f"execution_contract_draft.qualifications.{name}", "boolean_required", "must be a JSON boolean")
        elif value is False:
            v.issue(f"execution_contract_draft.qualifications.{name}", "qualification_missing", "runtime qualification has not passed", "blocker")
    v.exact(contract.get("output_directory_policy"), "new_non_overwriting_metrics_only", "execution_contract_draft.output_directory_policy")
    v.exact(contract.get("classification_thresholds_locked"), False, "execution_contract_draft.classification_thresholds_locked")
    thresholds = v.obj(contract.get("proposed_classification_thresholds"), "execution_contract_draft.proposed_classification_thresholds")
    required_thresholds = {"retain_or_low_share", "source_candidate", "not_source_main_domain", "sample_count_and_uncertainty"}
    if set(thresholds) != required_thresholds:
        v.issue("execution_contract_draft.proposed_classification_thresholds", "classification_thresholds", f"exact rule keys required: {sorted(required_thresholds)}")
    else:
        _check_classification_thresholds(v, thresholds, m)
        v.issue("execution_contract_draft.proposed_classification_thresholds", "thresholds_need_user_review", "numeric cutoffs are proposals and are not locked", "blocker")

    approval = contract.get("user_approval")
    if approval is not None:
        v.issue("execution_contract_draft.user_approval", "draft_has_approval", "preparation manifest must not contain an execution approval")
    else:
        v.issue("execution_contract_draft.user_approval", "execution_approval_missing", "a separate frozen-scan approval is required after runtime qualification", "blocker")
    blockers = v.array(m.get("current_implementation_blockers"), "current_implementation_blockers")
    if not blockers:
        v.issue("current_implementation_blockers", "missing_preparation_blockers", "draft must state outstanding implementation and qualification work")
    for index, item in enumerate(blockers):
        if not isinstance(item, dict) or not isinstance(item.get("component"), str) or not isinstance(item.get("finding"), str):
            v.issue(f"current_implementation_blockers[{index}]", "preparation_blocker_shape", "each blocker requires component and finding")

    return v.report()


def validate_execution_contract(manifest: object) -> ManifestValidation:
    """Reject incomplete manifests before file access, then verify artifact bytes.

    This is an offline preflight, not a scanner or a user approval mechanism.
    The future runner must call it before creating AppLauncher, then qualify
    runtime bindings independently. File verification never loads a policy.
    """

    m = copy.deepcopy(manifest)
    report = audit_manifest(m)
    if not report.lock_fields_complete:
        raise ManifestRejected(report)
    approval = m["execution_contract"]["user_approval"]
    if approval["manifest_sha256"] != manifest_payload_sha256(m):
        issue = ValidationIssue("error", "approval_manifest_hash_mismatch", "execution_contract.user_approval.manifest_sha256", "approval is not bound to this locked manifest payload")
        raise ManifestRejected(ManifestValidation((issue,), report.cell_count, report.episode_count))
    issues = []
    reports = {}
    records = {"base_checkpoint": m["base_checkpoint"], **m["execution_contract"]["artifacts"]}
    for role, record in records.items():
        path = Path(record["path"])
        try:
            digest = hashlib.sha256()
            if not path.is_file():
                raise OSError("artifact must be a regular file")
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise OSError("artifact must be a regular file")
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(after) or identity(before) != identity(path.stat()):
                raise OSError("artifact changed during hash verification")
            if digest.hexdigest() != record["sha256"]:
                issues.append(ValidationIssue("error", "artifact_hash_mismatch", role, "file bytes differ from registered SHA-256"))
            elif role in QUALIFICATION_BINDINGS or role == "user_approval_record":
                # Re-read only small JSON reports, then verify that the bytes
                # used for semantics still match the hash just checked.
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise OSError("report changed before semantic verification")
                try:
                    reports[role] = _parse_manifest_bytes(content)
                except (ValueError, UnicodeError) as error:
                    issues.append(ValidationIssue("error", "qualification_json", role, str(error)))
        except OSError as error:
            issues.append(ValidationIssue("error", "artifact_unreadable", role, str(error)))
    for role, required in QUALIFICATION_BINDINGS.items():
        evidence = reports.get(role)
        if evidence is None:
            continue  # The missing/changed/unreadable file already has an issue.
        bindings = {name: records[name]["sha256"] for name in required}
        if evidence.get("passed") is not True or evidence.get("bindings") != bindings:
            issues.append(ValidationIssue("error", "qualification_bindings", role, "qualification must pass and bind the exact registered input hashes"))
        if role == "bridge_smoke_report" and evidence.get("region_material_binding_verified") is not True:
            issues.append(ValidationIssue("error", "material_binding_unqualified", role, "smoke report must verify actual region material binding"))
    if "user_approval_record" in reports and reports["user_approval_record"] != approval:
        issues.append(ValidationIssue("error", "approval_record_mismatch", "user_approval_record", "approval record must match the manifest approval statement"))
    result = ManifestValidation(tuple(issues), report.cell_count, report.episode_count, not issues)
    if issues:
        raise ManifestRejected(result)
    return result


def validate_manifest_file(path: str | Path) -> ManifestValidation:
    """Verify a canonical manifest payload and artifact bytes; launch nothing."""
    content = Path(path).read_bytes()
    manifest = _parse_manifest_bytes(content)
    return validate_execution_contract(manifest)
