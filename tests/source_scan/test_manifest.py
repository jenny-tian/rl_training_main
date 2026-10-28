"""Offline contract tests. Synthetic 'passed' reports are not robot evidence."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rl_training.source_scan import manifest as api

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/source_scan_v0_1_draft.json"
PREPARATION = ROOT / "docs/lite3-source-scan-preparation-v0.2-2026-10-22/source_qualification_scan_manifest_v0.2_preparation.json"
CLI = ROOT / "scripts/reinforcement_learning/online_adaptation/audit_source_scan_manifest.py"


def _record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def draft():
    return api.load_manifest(FIXTURE)


def _bind_approval(manifest):
    """Make a synthetic test approval; this never authorizes any real scan."""
    contract = manifest["execution_contract"]
    approval = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": api.manifest_payload_sha256(manifest),
        "approval_reference": "synthetic-unit-test-only",
        "approved_at_utc": "2026-10-22T00:00:00Z",
        "scope": "frozen_source_scan_only",
    }
    contract["user_approval"] = approval
    path = Path(contract["artifacts"]["user_approval_record"]["path"])
    _write_json(path, approval)
    contract["artifacts"]["user_approval_record"] = _record(path)


@pytest.fixture
def locked(tmp_path, draft):
    m = copy.deepcopy(draft)
    m.update(schema_version=api.EXECUTION_SCHEMA, manifest_id="synthetic-offline-test-only",
             status=api.LOCKED_STATUS, execution_authorized=True, split=api.SCAN_SPLIT,
             current_implementation_blockers=[])
    m["research_boundary"].update(
        dynamics_profile="deployment_matched_frozen", observation_dim=45, action_dim=12,
        height_scan=False, privileged_observations=False, checkpoint_selection="none")
    m["episode_protocol"].update(early_stop="registered_safety_stop_only", additional_samples="forbidden")
    m["metrics"]["aggregation"] = "per_cell_and_per_family_no_direction_or_level_hidden"
    checkpoint = tmp_path / "not-a-real-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic opaque checkpoint; must never be deserialized")
    m["base_checkpoint"] = _record(checkpoint)
    artifacts = {}
    for role in api.ARTIFACT_ROLES:
        path = tmp_path / f"{role}.json"
        _write_json(path, {"synthetic_role": role})
        artifacts[role] = _record(path)
    contract = {
        "artifacts": artifacts,
        "geometry_audit_passed": True, "checkpoint_parity_passed": True,
        "bridge_smoke_passed": True, "region_material_binding_verified": True,
        "output_directory_policy": "new_non_overwriting_metrics_only",
        "classification_thresholds_locked": True,
        "classification_thresholds": {
            "retain_or_low_share": {
                "minimum_stable_instance_fraction": 0.75,
                "motion_quality_maxima": {
                    "mean_velocity_error_mps": 0.2, "rms_velocity_error_mps": 0.25,
                    "roll_peak_rad": 0.4, "pitch_peak_rad": 0.4, "lateral_offset_peak_m": 0.3,
                    "foot_slip_contact_fraction": 0.1, "safety_filter_projection_fraction": 0.1,
                },
            },
            "source_candidate": {
                "minimum_recoverable_failure_fraction": 0.1,
                "motion_quality_degradation_cutoffs": {
                    "mean_velocity_error_mps": 0.05, "rms_velocity_error_mps": 0.05,
                    "roll_peak_rad": 0.1, "pitch_peak_rad": 0.1, "lateral_offset_peak_m": 0.05,
                    "foot_slip_contact_fraction": 0.05, "safety_filter_projection_fraction": 0.05,
                },
                "safe_execution_required": True,
            },
            "not_source_main_domain": {
                "minimum_unrecoverable_failure_fraction": 0.5,
                "near_zero_forward_progress_m_max": 0.2,
                "safety_qualification_failure_excludes": True,
            },
            "sample_count_and_uncertainty": {
                "episodes_per_cell": 5, "confidence_level": 0.95, "interval_method": "synthetic_wilson",
            },
        },
    }
    m["execution_contract"] = contract
    plan = m["seed_plan"]
    plan["initial_state_partition"] = plan.pop("dynamics_partition")
    for split in ("source_train", "development", "sealed"):
        reserved = plan[split + "_reserved_partition"]
        reserved["initial_state"] = reserved.pop("dynamics")
    plan["source_validation_reserved_partition"] = {"geometry": [70000, 79999], "initial_state": [80000, 89999]}
    plan["probe_reserved_partition"] = {"geometry": [110000, 119999], "initial_state": [120000, 129999]}
    plan["total_episodes"] = 225
    for c in plan["cells"]:
        c["initial_state_seed_range"] = c.pop("dynamics_seed_range")
        c.update(seed_pairing="zip_geometry_initial_state", dynamics_profile="deployment_matched_frozen")
        p = c.pop("candidate_parameters")
        family, direction, level = c["terrain_family"], c["direction_or_layout"], c["level"]
        geometry = dict(size=[12.0, 4.0], flat_length_m=3.0, transition_length_m=0.5, grid_spacing_m=0.1)
        if family in ("multi_scale_rough", "slope_plus_rough"):
            geometry.update(rough_peak_valley_m=p["local_peak_valley_difference_m"],
                            rough_audit_window_m=0.4, spatial_scales_m=[0.2, 0.4])
        if family in ("slope", "slope_plus_rough"):
            geometry["slope_angle_deg"] = p["slope_angle_deg"]
        if family in ("multi_scale_rough", "slope", "slope_plus_rough"):
            geometry["exit_flat_length_m"] = 2.0
        if family == "multi_step_stairs":
            geometry.update(step_count=(3, 5, 6)[level - 1], step_height_m=(0.02, 0.04, 0.05)[level - 1], tread_depth_m=0.5)
        if family == "friction_variation":
            geometry.update(friction_range=p["effective_friction_coefficient_range"], friction_region_count=4)
        if family not in ("flat", "multi_scale_rough"):
            geometry["direction_or_layout"] = direction
        c["resolved_parameters"] = {
            "geometry": geometry,
            "reset": {"profile_id": "synthetic_fixture_only", "config_sha256": artifacts["reset_config"]["sha256"]},
            "track": {"start_x_m": 1.5, "completion_x_m": 10.5, "lateral_margin_m": 0.2},
        }
    records = {"base_checkpoint": m["base_checkpoint"], **artifacts}
    for role, names in api.QUALIFICATION_BINDINGS.items():
        evidence = {"passed": True, "bindings": {name: records[name]["sha256"] for name in names}}
        if role == "bridge_smoke_report":
            evidence["region_material_binding_verified"] = True
        path = Path(artifacts[role]["path"])
        _write_json(path, evidence)
        artifacts[role] = _record(path)
    _bind_approval(m)
    return m


def _codes(manifest):
    return {i.code for i in api.audit_manifest(manifest).issues}


def _cell(manifest, family):
    return next(c for c in manifest["seed_plan"]["cells"] if c["terrain_family"] == family)


def test_original_draft_audits_counts_and_blockers_without_mutation(draft):
    before = copy.deepcopy(draft)
    report = api.audit_manifest(draft)
    assert report.schema_valid and not report.execution_ready
    assert (report.cell_count, report.episode_count) == (45, 225)
    assert {"ambiguous_seed_role", "missing_reserved_partition", "unlocked_cell", "missing_execution_contract"} <= _codes(draft)
    assert draft == before


def test_v02_preparation_audits_as_structural_but_never_runnable():
    manifest = api.load_manifest(PREPARATION)
    before = copy.deepcopy(manifest)

    report = api.audit_preparation_manifest(manifest)

    assert report.schema_valid
    assert not report.lock_fields_complete
    assert not report.files_verified
    assert not report.execution_ready
    assert (report.cell_count, report.episode_count) == (45, 225)
    codes = {issue.code for issue in report.issues}
    assert {
        "artifact_unavailable",
        "qualification_missing",
        "thresholds_need_user_review",
        "execution_approval_missing",
    } <= codes
    assert manifest == before


def test_preparation_audit_does_not_read_referenced_artifacts(monkeypatch):
    manifest = api.load_manifest(PREPARATION)
    manifest["execution_contract_draft"]["artifacts"]["geometry_audit"] = {
        "path": "/does/not/exist/geometry-audit.json",
        "sha256": "0" * 64,
    }

    def fail_open(*args, **kwargs):
        pytest.fail("preparation audit must not read referenced artifact files")

    monkeypatch.setattr(Path, "open", fail_open)
    report = api.audit_preparation_manifest(manifest)

    assert report.schema_valid
    assert not report.files_verified


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", api.LOCKED_STATUS), ("execution_authorized", True)],
)
def test_preparation_audit_rejects_execution_state(field, value):
    manifest = api.load_manifest(PREPARATION)
    manifest[field] = value

    report = api.audit_preparation_manifest(manifest)

    assert not report.schema_valid
    assert "fixed_contract" in {issue.code for issue in report.issues}


def test_preparation_cli_reports_non_runnable_draft():
    result = subprocess.run(
        [sys.executable, str(CLI), str(PREPARATION), "--preparation-audit"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "source/rl_training")},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_valid"] is True
    assert payload["execution_ready"] is False
    assert payload["cell_count"] == 45
    assert payload["episode_count"] == 225


def test_draft_rejection_precedes_checkpoint_and_artifact_access(draft, monkeypatch):
    def fail_open(*args, **kwargs):
        pytest.fail("preflight touched a referenced file before rejecting the draft")
    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(api.ManifestRejected):
        api.validate_execution_contract(draft)


def test_flipping_status_and_authorization_alone_cannot_enable_legacy_draft(draft):
    draft.update(status=api.LOCKED_STATUS, execution_authorized=True, split=api.SCAN_SPLIT)
    with pytest.raises(api.ManifestRejected) as caught:
        api.validate_execution_contract(draft)
    assert "execution_schema_required" in {i.code for i in caught.value.report.issues}


def test_complete_synthetic_contract_verifies_opaque_files_without_loading_policy(locked, tmp_path):
    before = copy.deepcopy(locked)
    audit = api.audit_manifest(locked)
    assert audit.schema_valid and audit.lock_fields_complete
    assert not audit.files_verified and not audit.execution_ready
    path = tmp_path / "manifest.json"
    _write_json(path, locked)
    result = api.validate_manifest_file(path)
    assert result.execution_ready and result.files_verified
    assert (result.cell_count, result.episode_count) == (45, 225)
    assert locked == before


@pytest.mark.parametrize("split", ["source_train", "development_target", "sealed_transfer_eval", "sealed_online_adapt", "sealed", "development"])
def test_non_scan_split_is_rejected_before_file_access(locked, split, monkeypatch):
    locked["split"] = split
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("unexpected artifact access"))
    with pytest.raises(api.ManifestRejected) as caught:
        api.validate_execution_contract(locked)
    assert "source_split_only" in {i.code for i in caught.value.report.issues}


@pytest.mark.parametrize("key", list(api.GUARDS))
def test_each_immutability_or_data_isolation_guard_is_enforced(locked, key):
    locked["data_isolation_and_execution_guards"][key] = not api.GUARDS[key]
    assert "fixed_contract" in _codes(locked)


@pytest.mark.parametrize("seed_range", [[10000, 10003], [10000, 20000], [True, 10004], [10004, 10000], [0.0, 4.0], [-1, 3]])
def test_invalid_cell_seed_ranges_and_episode_mismatches(locked, seed_range):
    locked["seed_plan"]["cells"][0]["geometry_seed_range"] = seed_range
    assert not api.audit_manifest(locked).schema_valid


def test_duplicate_cells_reused_seeds_and_missing_conditions_are_separate_errors(locked):
    locked["seed_plan"]["cells"][1] = copy.deepcopy(locked["seed_plan"]["cells"][0])
    assert {"cell_id", "duplicate_scan_condition", "cell_seed_reuse", "incomplete_coverage"} <= _codes(locked)


@pytest.mark.parametrize("split", api.RESERVED_SPLITS)
def test_reserved_partitions_cannot_overlap_scan_or_each_other(locked, split):
    locked["seed_plan"][split + "_reserved_partition"]["geometry"] = [10004, 10020]
    assert "partition_overlap" in _codes(locked)


def test_cross_axis_partition_overlap_is_rejected(locked):
    locked["seed_plan"]["initial_state_partition"] = [10000, 19999]
    assert {"partition_overlap", "seed_outside_partition"} <= _codes(locked)


@pytest.mark.parametrize("field", ["cell_count", "total_episodes"])
def test_aggregate_counts_must_match_cells(locked, field):
    locked["seed_plan"][field] += 1
    assert "fixed_contract" in _codes(locked)


@pytest.mark.parametrize("value", [True, 0, 4.0])
def test_episodes_require_exact_positive_integers(locked, value):
    locked["seed_plan"]["cells"][0]["episodes"] = value
    assert not api.audit_manifest(locked).schema_valid


def test_fixed_dynamics_rejects_legacy_or_randomized_semantics(locked):
    c = locked["seed_plan"]["cells"][0]
    c["dynamics_seed_range"] = [20000, 20004]
    c["dynamics_profile"] = "randomized_mass"
    assert {"ambiguous_seed_role", "fixed_contract"} <= _codes(locked)


@pytest.mark.parametrize("placeholder", ["pending Lite3 calibration", "TBD", "to be filled", "待填写", None, ""])
def test_unresolved_geometry_is_rejected(locked, placeholder):
    _cell(locked, "multi_step_stairs")["resolved_parameters"]["geometry"]["tread_depth_m"] = placeholder
    assert "unresolved_field" in _codes(locked)


@pytest.mark.parametrize("role", api.ARTIFACT_ROLES)
def test_missing_locked_artifact_roles_cannot_pass(locked, role):
    del locked["execution_contract"]["artifacts"][role]
    assert not api.audit_manifest(locked).lock_fields_complete


def test_rough_and_friction_and_stairs_require_actual_resolved_geometry(locked):
    _cell(locked, "multi_scale_rough")["resolved_parameters"]["geometry"]["spatial_scales_m"] = [0.2, 0.2]
    _cell(locked, "friction_variation")["resolved_parameters"]["geometry"]["friction_range"] = [1.0, 0.3]
    _cell(locked, "multi_step_stairs")["resolved_parameters"]["geometry"]["step_count"] = 1
    assert {"rough_scales", "friction_range", "integer_range"} <= _codes(locked)


def test_20m_track_with_40s_budget_is_rejected(locked):
    c = locked["seed_plan"]["cells"][0]
    c["resolved_parameters"]["geometry"]["size"] = [20.0, 4.0]
    c["resolved_parameters"]["track"]["completion_x_m"] = 19.5
    assert "unreachable_completion" in _codes(locked)


def test_finishing_before_last_stair_is_rejected(locked):
    _cell(locked, "multi_step_stairs")["resolved_parameters"]["track"]["completion_x_m"] = 4.0
    assert "stairs_completion" in _codes(locked)


def test_slope_direction_cannot_disagree_with_geometry(locked):
    _cell(locked, "slope")["resolved_parameters"]["geometry"]["direction_or_layout"] = "downhill"
    assert "direction_mismatch" in _codes(locked)


def test_slope_completion_must_be_after_exit_transition_with_hold_buffer(locked):
    cell = _cell(locked, "slope")
    cell["resolved_parameters"]["track"]["completion_x_m"] = 9.9
    assert "slope_completion_before_flat" in _codes(locked)
    cell["resolved_parameters"]["track"]["completion_x_m"] = 11.5
    assert "slope_hold_buffer" in _codes(locked)


def test_declared_reset_hash_and_metrics_are_enforced(locked):
    locked["seed_plan"]["cells"][0]["resolved_parameters"]["reset"]["config_sha256"] = "0" * 64
    locked["metrics"]["per_episode_required"].remove("safety_filter_projection_fraction")
    assert {"reset_hash_mismatch", "episode_metrics"} <= _codes(locked)


def test_family_specific_metrics_must_match_registered_diagnostics(locked):
    locked["metrics"]["stairs_specific"].append("unregistered_metric")
    assert "family_metrics" in _codes(locked)


def test_classification_rules_must_be_numeric_and_nonempty(locked):
    locked["execution_contract"]["classification_thresholds"]["source_candidate"][
        "minimum_recoverable_failure_fraction"
    ] = "recoverable"
    assert "threshold_fraction" in _codes(locked)


def test_classification_rules_require_the_registered_metric_set(locked):
    del locked["execution_contract"]["classification_thresholds"]["retain_or_low_share"][
        "motion_quality_maxima"
    ]["roll_peak_rad"]
    assert "threshold_metrics" in _codes(locked)


def test_edit_after_approval_is_rejected_before_referenced_files(locked, monkeypatch):
    locked["seed_plan"]["cells"][0]["resolved_parameters"]["track"]["completion_x_m"] = 9.4
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("unapproved manifest caused artifact access"))
    with pytest.raises(api.ManifestRejected) as caught:
        api.validate_execution_contract(locked)
    assert {i.code for i in caught.value.report.issues} == {"approval_manifest_hash_mismatch"}


@pytest.mark.parametrize("role", ["base_checkpoint", "geometry_generator", "safety_config", "normalizer"])
def test_changed_file_bytes_fail_integrity(locked, role):
    record = locked["base_checkpoint"] if role == "base_checkpoint" else locked["execution_contract"]["artifacts"][role]
    Path(record["path"]).write_bytes(b"different bytes")
    with pytest.raises(api.ManifestRejected) as caught:
        api.validate_execution_contract(locked)
    assert "artifact_hash_mismatch" in {i.code for i in caught.value.report.issues}


def test_hash_matched_but_failed_or_wrong_checkpoint_qualification_is_rejected(locked):
    role = "checkpoint_parity_report"
    path = Path(locked["execution_contract"]["artifacts"][role]["path"])
    _write_json(path, {"passed": True, "bindings": {"base_checkpoint": "0" * 64}})
    locked["execution_contract"]["artifacts"][role] = _record(path)
    _bind_approval(locked)
    with pytest.raises(api.ManifestRejected) as caught:
        api.validate_execution_contract(locked)
    assert "qualification_bindings" in {i.code for i in caught.value.report.issues}


def test_material_binding_flag_alone_cannot_replace_smoke_report(locked):
    role = "bridge_smoke_report"
    path = Path(locked["execution_contract"]["artifacts"][role]["path"])
    report = api.load_manifest(path)
    report["region_material_binding_verified"] = False
    _write_json(path, report)
    locked["execution_contract"]["artifacts"][role] = _record(path)
    _bind_approval(locked)
    with pytest.raises(api.ManifestRejected) as caught:
        api.validate_execution_contract(locked)
    assert "material_binding_unqualified" in {i.code for i in caught.value.report.issues}


@pytest.mark.parametrize("text", ['{"status":1,"status":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '[]'])
def test_strict_json_rejects_ambiguous_or_nonfinite_inputs(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        api.load_manifest(path)


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": []}, {"x": float("inf")}])
def test_malformed_root_returns_errors_instead_of_crashing(value):
    assert not api.audit_manifest(value).schema_valid


def test_malformed_cell_container_returns_schema_issues_without_crashing(locked):
    locked["seed_plan"]["cells"] = 45
    report = api.audit_manifest(locked)
    assert not report.schema_valid
    assert "array_required" in {issue.code for issue in report.issues}


def test_cli_audits_draft_and_refuses_preflight_without_external_python_path():
    audited = subprocess.run([sys.executable, str(CLI), str(FIXTURE)], capture_output=True, text=True, check=False)
    assert audited.returncode == 0, audited.stderr
    report = json.loads(audited.stdout)
    assert report["schema_valid"] and not report["execution_ready"]
    refused = subprocess.run([sys.executable, str(CLI), str(FIXTURE), "--preflight"], capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert not json.loads(refused.stdout)["execution_ready"]


def test_import_and_audit_do_not_load_simulation_torch_or_training():
    code = """
import sys
from rl_training.source_scan.manifest import load_manifest, audit_manifest
audit_manifest(load_manifest(sys.argv[1]))
assert not any(name.split('.')[0] in {'torch','isaaclab','isaacsim','omni','pxr'} for name in sys.modules)
assert 'rl_training.online_adaptation' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code, str(FIXTURE)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
