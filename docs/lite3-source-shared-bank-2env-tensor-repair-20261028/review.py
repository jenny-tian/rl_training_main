"""Offline review of existing evidence. Does not launch simulation or rewrite outcome."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
archive = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "scripts/tools"))
from source_shared_tensor_contact import tensor_summary_passed
from source_parallel_terrain_startup_supervisor import find_collision_cooking_failure, find_fatal_exception

# Execute the committed predicate verbatim without importing its Kit-dependent module.
path = root / "scripts/tools/source_shared_terrain_diagnostic.py"
tree = ast.parse(path.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "shared_report_passed")
namespace = {"SHARED_CONTROL_STEPS": 20, "tensor_summary_passed": tensor_summary_passed}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
read = lambda name: json.loads((archive / name).read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest, execution, result = map(read, ("manifest.json", "execution.json", "result.json"))
assert namespace["shared_report_passed"](result)
assert execution["exit_code"] == 0 and execution["code_hashes_unchanged"] is True
assert all(execution[k] is None for k in ("stop_reason", "failure_marker", "fatal_exception", "collision_cooking_failure"))
assert all(sha(root / p) == h for p, h in manifest["source_file_sha256"].items())
assert all(sha(Path(p)) == h for p, h in manifest["dependency_file_sha256"].items())
assets = json.loads((root / "docs/lite3-usd-collision-geometry-audit-20261027.json").read_text())["assets"]
assert all(sha(root / a["path"]) == a["sha256"] for a in assets.values())
logs = {}
for name in ("console.log", "kit.log", "isaaclab.log"):
    text = (archive / name).read_text(errors="replace")
    errors = [l for l in text.splitlines() if "[Error]" in l or "Traceback (most recent call last)" in l]
    assert not errors and find_collision_cooking_failure(text) is None and find_fatal_exception(text) is None
    logs[name] = {"errors": errors, "warning_count": text.count("[Warning]")}
journal = [json.loads(l) for l in (archive / "result.shared.jsonl").read_text().splitlines()]
assert journal[-1]["phase"] == "complete"
assert sum(x.get("phase") == "tensor_contact_sample" for x in journal) == 80
assert sum(x.get("phase") == "reset_checked" for x in journal) == 5
assert not (archive / "test_processes_after.txt").read_text().strip()
gpu0 = (archive / "gpu_after.csv").read_text().splitlines()[0].split(", ")
assert gpu0[1] not in (archive / "gpu_compute_after.csv").read_text()
review = {
    "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "simulation_evidence_passed_supervisor_finalization_incomplete",
    "runtime_revision": manifest["revision"], "run_count_this_phase": 1,
    "simulation_rerun_during_review": False,
    "shared_report_predicate_passed_via_ast": True,
    "original_supervisor_outcome_present": (archive / "outcome.json").exists(),
    "original_supervisor_exit_code": None,
    "simulation_child_exit_code": execution["exit_code"],
    "supervisor_import_failure_reproduced": "ModuleNotFoundError: No module named 'pxr'",
    "original_supervisor_stderr_retained": False,
    "source_hashes_verified": len(manifest["source_file_sha256"]),
    "dependency_hashes_verified": len(manifest["dependency_file_sha256"]),
    "asset_hashes_verified": len(assets),
    "journal_phase_counts": dict(Counter(x.get("phase") for x in journal)),
    "contact_samples": result["contact_capture"]["samples"],
    "ground_contact_points": result["ground_contact_points"],
    "cross_environment_contacts": result["cross_environment_contacts"],
    "contact_peak_used": result["contact_capture"]["peak_used"],
    "contact_capacity": result["contact_capture"]["tensor_topology"]["capacity"],
    "backend_state": result["contact_capture"]["runtime_state"]["backend_state"],
    "reset_checks": result["reset_checks"],
    "elapsed_s": execution["elapsed_s"],
    "sampled_gpu_peak_mib": execution["sampled_gpu_peak_mib"],
    "gpu0_after": gpu0, "test_processes_remaining": False,
    "log_review": logs,
    "training_started": False, "full_qualification_claim": False,
}
(archive / "post_run_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(review, ensure_ascii=False, indent=2))
