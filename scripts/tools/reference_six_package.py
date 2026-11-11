"""Verify and package the bounded reference-six startup evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from reference_lite3_loader import ROOT, REFERENCE, REFERENCE_COMMIT

OUT = ROOT / "docs/lite3-reference-six-scratch-20261111"
RAW = ROOT / "logs/reference_six/validation"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    records, jobs = [], []
    for name, mode, steps, updates in (("smoke_v4", "smoke", 100, 0), ("full_scale_v1", "validate", 72, 3)):
        directory = RAW / name
        report, execution = read(directory / "report.json"), read(directory / "execution.json")
        audit = read(directory / "simulation_audit.json")
        config = read(directory / "configuration_audit.json")
        asset = read(directory / "asset_resolution.json")
        assert report["status"] == "passed" and execution["status"] == "completed"
        assert report["mode"] == mode and report["environment_steps"] == steps and report["completed_updates"] == updates
        assert report["from_scratch"] and report["checkpoint_loaded"] is None and report["initial_optimizer_state_entries"] == 0
        assert audit["ray_probe_tiles"] == 200 and audit["ray_probes"] == 5000 and audit["scanner_collider_max_error_m"] == 0
        assert config["non_terrain_environment_exact"] and config["native_layout_exact"] and asset["all_asset_blobs_exact"]
        for entry in report["sources"] + asset["files"]:
            assert sha(Path(entry["path"])) == entry["sha256"], entry["path"]
        runtime = (directory / "runtime.log").read_text(errors="replace")
        errors = [line for line in runtime.splitlines() if "[Error]" in line or "Traceback" in line]
        assert not errors and not execution["fatal_messages"]
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        for filename in ("report.json", "execution.json", "simulation_audit.json", "configuration_audit.json",
                         "asset_resolution.json", "runtime_overrides.json", "params/env.yaml", "params/agent.yaml"):
            target = OUT / "validation" / name / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / filename, target)
        if mode == "smoke":
            shutil.copyfile(directory / "terrain_geometry.png", OUT / "terrain_geometry.png")
        jobs.append(dict(name=name, mode=mode, status=report["status"], updates=updates,
                         interactions=report["interactions"], peak_total_gpu_used_mib=execution["peak_total_gpu_used_mib"],
                         elapsed_s=execution["elapsed_s"], error_lines=errors,
                         warning_lines=sum("[Warning]" in line or "WARNING" in line for line in runtime.splitlines())))
    attempts = []
    for name in ("smoke_v1", "smoke_v2", "smoke_v3"):
        directory = RAW / name
        error = read(directory / "error.json")
        attempts.append(dict(name=name, execution=read(directory / "execution.json"), error=error))
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    cpu = ET.parse(OUT / "validation/cpu.xml").getroot().find("testsuite")
    assert cpu is not None and int(cpu.attrib["tests"]) == 8
    assert all(int(cpu.attrib[key]) == 0 for key in ("failures", "errors", "skipped"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REFERENCE, text=True).strip()
    assert commit == REFERENCE_COMMIT
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REFERENCE, check=True)
    source_paths = list((ROOT / "scripts/tools").glob("reference_*.py"))
    source_paths += [ROOT / "tests/source_curriculum/test_reference_six_contract.py"]
    validation = dict(status="passed", jobs=jobs, cpu_tests=8, reference_commit=commit, reference_tracked_files_unchanged=True,
                      formal_training_started=False, initialization="random", stage="awaiting_next_phase_confirmation",
                      source_files=[dict(path=str(p), sha256=sha(p)) for p in sorted(source_paths)],
                      limitations=["Startup and numerical checks do not establish terrain locomotion performance.",
                                   "Raw USD visual warnings retained; collision checks passed."])
    for name, data in (("validation.json", validation), ("artifact_manifest.json", records), ("startup_attempts.json", attempts)):
        (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    for path in paths:
        if path.suffix in (".md", ".py"):
            assert all(c not in path.read_text() for c in (chr(0x201c), chr(0x201d)))
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in paths))
    print(json.dumps(dict(status="passed", files=len(paths), raw_files=len(records), cpu_tests=8, formal_training_started=False)))


if __name__ == "__main__":
    main()
