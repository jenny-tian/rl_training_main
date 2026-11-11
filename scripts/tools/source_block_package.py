"""Archive B1 receipts, statistics, and actual rendered assets with hashes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "logs/stage1/block_l2"
OUT = ROOT / "docs/lite3-block-l2-b1-20261111"


def package():
    pilot = json.loads((RAW / "pilot30_summary.json").read_text())
    assert pilot["status"] == "passed_numerical_and_budget_checks"
    run_dir = Path(pilot["checkpoint"]).parent
    for check in ("check120_v2", "check4096_v1"):
        assert json.loads((RAW / check / "checks.json").read_text())["all_passed"]
        assert json.loads((RAW / check / "execution.json").read_text())["status"] == "completed"
    for check in ("visuals_v2", "overview_v3"):
        assert json.loads((RAW / check / "execution.json").read_text())["status"] == "completed"
    assert json.loads((RAW / "geometry_final.json").read_text())["all_passed"]
    sources = {
        "geometry.json": RAW / "geometry_final.json",
        "check120.json": RAW / "check120_v2/checks.json",
        "check120_runtime.json": RAW / "check120_v2/execution.json",
        "check4096.json": RAW / "check4096_v1/checks.json",
        "check4096_runtime.json": RAW / "check4096_v1/execution.json",
        "pilot30.json": RAW / "pilot30_summary.json",
        "pilot30_runtime.json": RAW / "pilot30_v1/execution.json",
        "column_statistics.json": run_dir / "column_statistics.json",
        "visuals/metadata.json": RAW / "visuals_v2/metadata.json",
        "visuals/runtime.json": RAW / "visuals_v2/execution.json",
        "visuals/overview_metadata.json": RAW / "overview_v3/metadata.json",
        "visuals/overview_runtime.json": RAW / "overview_v3/execution.json",
        "visuals/terrain_overview.png": RAW / "overview_v3/terrain_overview.png",
    }
    for path in (RAW / "visuals_v2").iterdir():
        if path.suffix in (".png", ".mp4") and path.name not in ("terrain_overview.png", "terrain_montage.png"):
            sources[f"visuals/{path.name}"] = path
    for name, path in sources.items():
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # Only replace the overview panel with its uncropped real simulator frame.
    # All captions and closeups remain the original simulator captures.
    montage = Image.open(RAW / "visuals_v2/terrain_montage.png")
    montage.paste(Image.open(RAW / "overview_v3/terrain_overview.png").resize((512, 288)), (0, 88))
    montage.save(OUT / "visuals/terrain_montage.png")
    with (OUT / "exposure.csv").open("w", newline="") as output:
        fields = ["name", "environments", "transitions", "transitions_per_training_second", "starts", "completion", "failure", "timeout", "completed_fraction"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pilot["groups"])
    # Confirm the entire trained task implementation matches its recorded
    # commit. Later edits concern rendering/reporting only.
    task_paths = [
        "source/rl_training/rl_training/source_curriculum/block.py",
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/source_block.py",
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/terrains/source_block_terrain.py",
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/source_block_env_cfg.py",
    ]
    for path in task_paths:
        assert subprocess.check_output(["git", "show", f"{pilot['source_commit']}:{path}"], cwd=ROOT) == (ROOT / path).read_bytes()
    originals = {path for path in sources.values()}
    originals.update(path for path in RAW.rglob("*") if path.is_file())
    originals.update(path for path in run_dir.rglob("*") if path.is_file())
    originals.update(ROOT / path for path in task_paths)
    original_records = [dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        for path in sorted(originals)]
    manifest = dict(b1_complete=True, b2_started=False, task_source_commit=pilot["source_commit"],
                    task_source_unchanged_since_training=True, original_artifacts=original_records,
                    note="Visuals use initial model_3497. Pilot model_3526 is retained only as audit evidence, excluded from policy selection.")
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in files))
    print(json.dumps(dict(delivered_files=len(files), raw_artifact_hashes=len(original_records), output=str(OUT))))


if __name__ == "__main__":
    package()
