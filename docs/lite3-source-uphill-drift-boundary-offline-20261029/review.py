"""Reproduce offline findings and compare the guard with its frozen predecessor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "source/rl_training")]

import torch

from rl_training.source_scan.tile_bounds import source_tile_bounds_terminated
from scripts.tools.source_uphill_drift_audit import analyze, comparisons
from scripts.tools.source_policy_prequalification import make_run_manifest, select_case, verify_hashes


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_frozen_guard():
    path = "source/rl_training/rl_training/source_scan/tile_bounds.py"
    revision = "73c10116072dc2fca2c4b4dc5f6c5b2ebc1b4af9"
    original = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    manifest = json.loads((ROOT / "docs/lite3-source-policy-prequalification-uphill-l1-20261029/manifest.json").read_text())
    digest = hashlib.sha256(original).hexdigest()
    assert digest == manifest["input_sha256"][path]
    module = types.ModuleType("frozen_guard_for_cpu_equivalence")
    exec(compile(original, path, "exec"), module.__dict__)
    generator = torch.Generator(device="cpu").manual_seed(101029)
    parameters = dict(size=(12., 4.), flat_length_m=3., collision_radius_m=0.27,
                      numerical_margin_m=0.001, control_period_s=0.02)
    evaluated = 0
    for dtype in (torch.float32, torch.float64):
        for bodies in (1, 17):
            count = 2048
            origins = torch.zeros((3, 4, 3), dtype=dtype)
            origins[:, :, 0] = torch.arange(3, dtype=dtype)[:, None] * 12
            origins[:, :, 1] = torch.arange(4, dtype=dtype)[None, :] * 4
            levels = torch.randint(0, 3, (count,), generator=generator)
            columns = torch.randint(0, 4, (count,), generator=generator)
            positions = torch.rand((count, bodies, 3), generator=generator, dtype=dtype)
            positions *= positions.new_tensor([12, 4, 1])
            positions -= positions.new_tensor([1.5, 2, 0])
            positions += origins[levels, columns, None]
            velocities = torch.randn((count, bodies, 3), generator=generator, dtype=dtype) * 4
            # Preserve per-environment fail-closed behavior and the original
            # XY-only nonfinite predicate; no new Z safety rule is introduced.
            levels[0] = -1
            columns[1] = 9
            positions[2, 0, 0] = float("nan")
            velocities[3, 0, 1] = float("inf")
            origins[2, 3, 1] = float("nan")
            positions[4, 0, 2] = float("nan")
            args = (positions, velocities, origins, levels, columns)
            expected = module.source_tile_bounds_terminated(*args, **parameters)
            actual = source_tile_bounds_terminated(*args, **parameters)
            details = {}
            instrumented = source_tile_bounds_terminated(*args, **parameters, diagnostics=details)
            assert torch.equal(expected, actual) and torch.equal(expected, instrumented)
            assert torch.equal(expected, details["terminated"])
            evaluated += count
    return dict(original_revision=revision, original_guard_sha256=digest,
                cpu_environment_cases=evaluated, mismatched_flags=0,
                dtypes=["float32", "float64"], body_counts=[1, 17],
                comparison="frozen original vs current, diagnostics both disabled and enabled")


def review():
    preparation = read("preparation.json")
    verify_hashes(preparation["input_sha256"])
    assert sha(Path(preparation["checkpoint"])) == preparation["checkpoint_sha256"]
    assert preparation["simulation_run"] is preparation["runtime_authorized"] is preparation["ppo_authorized"] is False
    assert len(preparation["cases"]) == 1
    case = select_case(preparation, "slope_L1_uphill")
    assert (case["geometry_seed"], case["initial_state_seed"], case["geometry_difficulty"]) == (10001, 20060, 0.3384848514104325)
    generated = make_run_manifest(preparation, case, revision="offline-only", gpu="not-queried",
                                  processes="not-queried", preparation_sha256=sha(OUT / "preparation.json"))
    assert generated["tile_bounds_diagnostics_required"] is True
    assert generated["max_control_steps"] == 2000
    assert not {"simulation_run", "status", "cases"}.intersection(generated)
    legacy_preparation = ROOT / "docs/lite3-source-followup-preparation-20261029/preparation.json"
    assert sha(legacy_preparation) == "a2199a7e558a0ba047adc671b941c68ac851c4d4ba3c198e609ab3e50e98f2b2"
    preserved_files = 0
    for label in ("flat", "rough-l1", "uphill-l1"):
        archive = ROOT / f"docs/lite3-source-policy-prequalification-{label}-20261029"
        for line in (archive / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert sha(archive / name) == expected, name
            preserved_files += 1
    report, _ = analyze()
    assert report == read("drift_analysis.json")
    assert comparisons() == read("historical_comparison.json")
    equivalence = compare_frozen_guard()
    suites = ET.parse(OUT / "regression.junit.xml").getroot().findall("testsuite")
    counts = {k: sum(int(s.attrib[k]) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
    assert counts["tests"] > 589 and all(counts[k] == 0 for k in ("errors", "failures", "skipped")), counts
    regression_history = {}
    for name in ("regression_initial", "supervisor_recheck"):
        history_suites = ET.parse(OUT / f"{name}.junit.xml").getroot().findall("testsuite")
        regression_history[name] = {k: sum(int(s.attrib[k]) for s in history_suites)
                                   for k in ("tests", "errors", "failures", "skipped")}
    assert regression_history["regression_initial"] == dict(tests=615, errors=0, failures=1, skipped=0)
    assert regression_history["supervisor_recheck"] == dict(tests=64, errors=0, failures=0, skipped=0)
    stage = read("next_stage.json")
    assert stage["runtime_authorized"] is False and stage["automatic_retries"] == 0
    assert stage["case"] == case and stage["preparation_sha256"] == sha(OUT / "preparation.json")
    assert not Path(stage["run_directory"]).exists()
    ledger = read("baseline_progress.json")
    previous = json.loads((ROOT / "docs/lite3-source-policy-prequalification-uphill-l1-20261029/baseline_progress.json").read_text())
    assert ledger == previous and ledger["remaining_prepared_unrun_cases"] == 12
    if any(n == "pxr" or n.startswith(("isaaclab", "isaacsim", "omni.")) for n in sys.modules):
        raise RuntimeError("offline review imported a simulator namespace")
    return {
        "status": "offline_review_passed_runtime_not_run", "simulation_run": False, "ppo_initialized": False,
        "input_hashes_verified": len(preparation["input_sha256"]),
        "prior_archive_files_unchanged": preserved_files, "original_preparation_unchanged": True,
        "guard_equivalence": equivalence, "regression": counts,
        "regression_history": regression_history,
        "initial_cleanup_test_failure": "Not reproduced in the supervisor and full-suite rechecks; cause unisolated, supervisor code unchanged. Not claimed fixed.",
        "drift_reproduced": True, "historical_comparison_reproduced": True,
        "baseline_progress_unchanged": True, "future_run_spec_checked": True,
        "runtime_authorized": False, "g3_qualified": False, "g4_qualified": False,
        "limitations": ["CPU fixtures cannot verify live PhysX tensors, logging overhead or actual reset timing.",
                        "The failed historical uphill case remains failed; new telemetry cannot recover its missing body/branch evidence."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = review()
    if args.write:
        with (OUT / "post_review.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2))
