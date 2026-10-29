"""Validate the archived eight-environment run without starting a simulator."""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT / "scripts/tools"), str(ROOT / "source/rl_training")]
from source_shared_contact_report import classify_contact
from source_shared_report_validation import shared_report_passed, shared_lifecycle_passed
from source_shared_tensor_contact import tensor_summary_passed, FORCE_ATOL, FORCE_RTOL
from rl_training.source_scan.shared_terrain_bank import SharedBankTerrainAllocator
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS


def read(name):
    return json.loads((OUT / name).read_text())


def rows(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt, manifest = read("execution_receipt.json"), read("manifest.json")
    for name, digest in receipt["original_artifact_sha256"].items():
        assert sha(OUT / name) == digest, name
    assert manifest["revision"] == receipt["runtime_revision"] == "fba100c5ee15eb069f4a35ebd141bc2b0ecc50f3"
    for field in ("source_file_sha256", "dependency_file_sha256"):
        for name, digest in manifest[field].items():
            assert sha(ROOT / name) == digest, name
    assert manifest["num_envs"] == 8 and manifest["seed"] == 130196
    assert manifest["device"] == "cuda:0" and manifest["minimum_gpu_available_mib"] == 12288
    assert manifest["policy_loaded"] is False and manifest["ppo_updates"] == 0
    reference = json.loads((OUT.parent / "lite3-source-shared-bank-g2-2env-recheck-20261029/manifest.json").read_text())
    for field in ("source_file_sha256", "dependency_file_sha256", "model_revision"):
        assert manifest[field] == reference[field], field
    outcome, execution, report = read("outcome.json"), read("execution.json"), read("result.json")
    assert outcome["status"] == "diagnostic_smoke_passed" and execution["exit_code"] == 0
    for key in ("stop_reason", "failure_marker", "collision_cooking_failure", "fatal_exception", "finalization_error"):
        assert outcome[key] is None and execution[key] is None, key
    assert outcome["code_hashes_unchanged"] is execution["code_hashes_unchanged"] is True
    assert outcome["training_started"] is False
    assert execution["startup_timeout_s"] == 120 and execution["total_timeout_s"] == 240
    assert shared_report_passed(report, reset_lifecycle=True)
    assert shared_lifecycle_passed(report["reset_checks"])
    capture = report["contact_capture"]
    assert tensor_summary_passed(capture, report["ground_contact_points"])

    contacts, resets = rows("result.shared.jsonl"), rows("result.shared.resets.jsonl")
    phase_counts = Counter(row.get("phase", "") for row in contacts)
    assert phase_counts["control_step"] == 20 and phase_counts["contact_runtime"] == 81
    samples = [row for row in contacts if row.get("phase") == "tensor_contact_sample"]
    assert [row["sim_step"] for row in samples] == list(range(1, 81))
    topology = capture["tensor_topology"]
    assert len(topology["sensor_paths"]) == 136 and len(topology["ground_colliders"]) == 24
    grouped, positives = defaultdict(list), Counter()
    for row in contacts:
        if row.get("phase") != "tensor_contact":
            continue
        kind, owner = classify_contact(*row["actors"])
        assert kind == row["kind"] and kind != "cross_environment"
        grouped[row["sim_step"]].append(row)
        if kind == "ground" and np.any(np.abs(row["forces_N"]) > FORCE_ATOL):
            positives[str(owner)] += row["count"]
    assert dict(positives) == report["ground_contact_points"]
    for sample in samples:
        matrix = np.zeros((136, 160, 3))
        counts = np.zeros((136, 160), dtype=int)
        for row in grouped[sample["sim_step"]]:
            i = topology["sensor_paths"].index(row["actors"][0])
            j = topology["filter_paths"][i].index(row["actors"][1])
            force, normal = np.asarray(row["forces_N"]), np.asarray(row["normals"])
            assert len(force) == len(normal) == row["count"]
            assert np.isfinite(force).all() and np.isfinite(normal).all()
            matrix[i, j] += np.sum(force * normal, axis=0)
            counts[i, j] += row["count"]
        np.testing.assert_array_equal(counts, sample["counts"])
        np.testing.assert_allclose(matrix, sample["pair_forces_N"], atol=FORCE_ATOL, rtol=FORCE_RTOL)
        np.testing.assert_allclose(matrix.sum(axis=1), sample["net_forces_N"], atol=FORCE_ATOL, rtol=FORCE_RTOL)
        assert int(counts.sum()) == sample["used"] < sample["capacity"] == 65536

    oracle = SharedBankTerrainAllocator(8, SOURCE_RESET_BLOCK_COUNTS, 130196, bank_rows=1)
    current = [asdict(t) for t in oracle.reset(list(range(8)))]
    assert resets[0]["initial_assignments"] == current == report["initial_assignments"]
    tickets = list(current)
    prior_rows = [r for r in resets if r.get("phase") == "before_reset"]
    comparisons = [r for r in resets if r.get("phase") == "assignment_comparison"]
    checks = [r for r in resets if r.get("phase") == "reset_checked"]
    assert len(prior_rows) == len(comparisons) == len(checks) == 33
    for before, comparison, checked in zip(prior_rows, comparisons, checks, strict=True):
        ids = before["ids"]
        assert ids == sorted(before["input_ids"]) == comparison["ids"] == checked["ids"]
        assert before["prior_assignments"] == current
        actual = comparison["actual_assignments"]
        assert actual == comparison["expected_assignments"] == [asdict(t) for t in oracle.reset(ids)]
        for env_id in set(range(8)) - set(ids):
            assert comparison["current_assignments"][env_id] == current[env_id]
        current = comparison["current_assignments"]
        assert checked["assignments"] == actual and checked["untouched_buffers_equal"] is True
        assert checked["history_buffers"] == []
        tickets.extend(actual)
    assert [t["assignment_index"] for t in tickets] == list(range(129))
    assert [r["input_ids"] for r in prior_rows[:-1]] == [[7], [6, 1, 3], [4, 2, 0], list(range(7, -1, -1))] * 8
    assert prior_rows[-1]["mode"] == "automatic_timeout" and prior_rows[-1]["ids"] == [7]
    auto, = [r for r in resets if r.get("phase") == "automatic_reset_output"]
    assert auto["terminated"] == [False] * 8 and auto["truncated"] == [False] * 7 + [True]
    recovery, = [r for r in resets if r.get("phase") == "post_automatic_reset_step"]
    assert recovery["passed"] is True
    console = (OUT / "console.log").read_text()
    for marker in ("Traceback (most recent call last)", "[Error]", "[ERROR]", "fatal-exception-recorded"):
        assert marker not in console, marker
    assert read("processes_after.json")["matching_runtime_processes"] == []
    result = {
        "status": "passed_with_documented_limitations", "revision": manifest["revision"],
        "num_envs": 8, "runtime_inputs_match_updated_two_env_run": True,
        "source_hash_count": len(manifest["source_file_sha256"]),
        "dependency_hash_count": len(manifest["dependency_file_sha256"]),
        "original_runtime_files_verified": len(receipt["original_artifact_sha256"]),
        "elapsed_s": execution["elapsed_s"], "sampled_gpu_peak_mib": execution["sampled_gpu_peak_mib"],
        "environment_construction_s": report["environment_construction_seconds"],
        "cpu_peak_rss_kib": report["cpu_peak_rss_kib"],
        "contact_phase_counts": dict(phase_counts), "ground_contact_points": dict(positives),
        "cross_environment_contacts": 0, "contact_matrices_reconstructed": 80,
        "contact_capacity": 65536, "contact_peak_used": max(r["used"] for r in samples),
        "reset_assignments": 129, "reset_batches": 33, "deterministic_reset_replay_passed": True,
        "automatic_timeout_env": 7, "post_automatic_reset_step_passed": True,
        "unresolved_visual_reference_warnings": console.count("Unresolved reference prim path"),
        "policy_loaded": False, "ppo_updates": 0, "g3_qualified": False, "g4_qualified": False,
        "limitations": ["Zero-action contact and reset checks only; no policy motion.",
            "Untouched-state equality excludes physics integration; history buffers were empty.",
            "No per-substep sensor timestamp or throughput qualification.",
            "Missing foot visual references remain unresolved."]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
