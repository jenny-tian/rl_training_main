"""Recheck this frozen preparation on CPU; never dispatch a runtime command."""

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/rl_training")]

from scripts.tools.source_flat_drift_audit import analyze
from scripts.tools.source_policy_prequalification import make_run_manifest, select_case, verify_hashes
from scripts.tools.source_parallel_terrain_startup_supervisor import minimum_gpu_available_mib


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preparation = read("preparation.json")
    assert preparation["simulation_run"] is preparation["runtime_authorized"] is False
    assert preparation["ppo_authorized"] is False
    verify_hashes(preparation["input_sha256"])
    assert sha(Path(preparation["checkpoint"])) == preparation["checkpoint_sha256"]
    geometry = read("geometry_equivalence.json")
    for name, expected in geometry["sources_sha256"].items():
        assert sha(Path(name)) == expected, name
    assert geometry["bank_grid"] == [1, 15] and geometry["bank_seed"] == 10001
    assert geometry["historical_flat_geometry_matches_bank"] is True
    assert [c["bank_column"] for c in geometry["cases"]] == list(range(15))
    assert all(c["isolated_mesh_matches_bank"] is True for c in geometry["cases"])
    assert preparation["cases"] == geometry["cases"][1:]
    manifest_count = 0
    for candidate in preparation["cases"]:
        case = select_case(preparation, candidate["case_id"])
        manifest = make_run_manifest(preparation, case, revision="offline-only",
                                    gpu="not-queried", processes="not-queried",
                                    preparation_sha256=sha(OUT / "preparation.json"))
        assert not {"status", "simulation_run", "cases"}.intersection(manifest)
        assert manifest["episode_count"] == manifest["cell"]["episodes"] == 1
        assert manifest["cell"]["geometry_seed_range"] == [10001, 10001]
        assert manifest["cell"]["initial_state_seed_range"] == [case["initial_state_seed"]] * 2
        manifest_count += 1

    archive = ROOT / "docs/lite3-source-policy-prequalification-flat-20261029"
    archive_files = 0
    for line in (archive / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(archive / name) == expected, name
        archive_files += 1
    recomputed, _ = analyze(archive)
    assert recomputed == read("drift_analysis.json")

    suites = ET.parse(OUT / "regression.junit.xml").getroot().findall("testsuite")
    counts = {key: sum(int(s.attrib[key]) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    assert counts == dict(tests=589, failures=0, errors=0, skipped=0), counts
    plan = read("shared_bank_recheck.json")
    assert plan["runtime_authorized"] is False and plan["next_proposal"] == "shared_bank_g2_2env"
    for stage, envs, memory, tickets, batches in (
        (plan["stages"][0], 2, 8192, 123, 91),
        (plan["stages"][1], 8, 12288, 129, 33),
    ):
        assert stage["num_envs"] == envs and stage["runtime_status"] == "not_run"
        assert stage["minimum_free_mib"] == memory == minimum_gpu_available_mib(num_envs=envs, shared_bank=True)
        assert stage["expected_assignment_count"] == tickets and stage["expected_reset_batch_count"] == batches
        argv = stage["command"]
        for key, value in (("--num-envs", str(envs)), ("--seed", "130196"), ("--gpu-index", "0"),
                           ("--startup-timeout-s", "120"), ("--total-timeout-s", "240")):
            assert argv[argv.index(key) + 1] == value
        assert "--shared-bank" in argv and "--reset-lifecycle" in argv
        assert not Path(argv[argv.index("--run-dir") + 1]).exists(), "prepared output directory is already occupied"
    report = {
        "status": "offline_review_passed_runtime_not_run", "regression": counts,
        "input_hash_count": len(preparation["input_sha256"]),
        "geometry_matches": 15, "remaining_cases_checked": manifest_count,
        "historical_flat_geometry_matches_bank": True,
        "historical_archive_files_verified": archive_files,
        "drift_recomputed_from_immutable_samples": True,
        "single_case_manifest_provenance_checked": True,
        "shared_recheck_budgets_match_supervisor": True,
        "simulation_started": False, "ppo_started": False, "runtime_authorized": False,
        "g3_qualified": False, "g4_qualified": False,
        "limitation": "CPU mesh equivalence and test doubles do not establish policy safety, PhysX cooking, sensor timing or motion isolation.",
    }
    if any(name == "pxr" or name.startswith(("isaaclab", "isaacsim", "omni.")) for name in sys.modules):
        raise RuntimeError("offline review loaded a simulator namespace")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
