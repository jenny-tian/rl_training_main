"""Verify raw evidence and seal the B3 delivery files (without changing training)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from source_block_b3_protocol import ROOT

OUT = ROOT / "docs/lite3-block-b3-diagnosis-20261111"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    count = 0
    for filename in ("artifact_manifest.json", "video_artifact_manifest.json"):
        for row in json.loads((OUT / filename).read_text()):
            path = Path(row["path"])
            assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], path
            count += 1
    previous = ROOT / "docs/lite3-block-b2-initialization-20261111"
    old_rows = (previous / "SHA256SUMS").read_text().splitlines()
    for row in old_rows:
        expected, name = row.split("  ", 1)
        assert sha(previous / name) == expected
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["status"] == "passed" and summary["frozen_jobs"] == 24
    assert all(r["all_initial_fields_exact"] for r in summary["pairing"].values())
    metadata = json.loads((OUT / "videos/metadata.json").read_text())
    assert metadata["status"] == "passed" and len(metadata["selected"]) == 3
    for row in metadata["selected"]:
        assert row["same_initial_hashes"] and row["same_termination_flags"]
        assert row["max_velocity_difference_from_matrix"] == 0
    for directory in (ROOT / "logs/stage1/block_b3/matrix/flat/controlled").iterdir():
        a = np.load(directory / "initial.npz")
        b = np.load(ROOT / "logs/stage1/block_b3/matrix/block/controlled" / directory.name / "initial.npz")
        assert np.array_equal(a["root"][:, 3:], b["root"][:, 3:])
        assert abs((a["root"][:, :3]-a["origins"])-(b["root"][:, :3]-b["origins"])).max() <= 1e-5
    subprocess.run(["git", "apply", "--check", str(OUT / "standstill-candidate.patch")], cwd=ROOT, check=True)
    processes = subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines()
    worker_names = {"source_block_b3_diagnose.py", "source_block_b3_matrix.py"}
    remaining = [line for line in processes if any(Path(word).name in worker_names for word in line.split())]
    assert not remaining, remaining
    sources = list((ROOT / "scripts/tools").glob("source_block_b3_*.py"))
    sources += [ROOT / "tests/source_curriculum/test_block_b3_protocol.py"]
    base = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    sources += [base / name for name in ("mdp/commands.py", "mdp/rewards.py", "mdp/source_block.py",
        "config/quadruped/deeprobotics_lite3/flat_env_cfg.py",
        "config/quadruped/deeprobotics_lite3/source_curriculum_env_cfg.py",
        "config/quadruped/deeprobotics_lite3/source_block_env_cfg.py")]
    sources += [Path("/home/tjx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py"),
        Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py"),
        Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py")]
    validation = dict(status="passed", cpu_test_command="PYTHONPATH=source/rl_training python -m pytest -q -p no:typeguard tests/source_curriculum tests/ea_lpl/test_stage1_retention_metrics.py",
        cpu_test_result="108 passed in 4.86s; observed during this stage, not rerun by this sealing script",
        numerical_jobs=24, paired_controlled_starts=6144, native_starts=12288,
        actor_updated=False, repaired_reward_enabled=False, candidate_patch_applies=True,
        raw_files_verified=count, previous_b2_files_verified=len(old_rows),
        cross_domain_root_orientation_and_velocity_exact=True,
        selected_replays_verified=3, cross_gpu_block_replays_velocity_and_endings_exact=2,
        graphical_error_jobs=12, graphical_errors_disclosed=True, rejected_video_attempts=3,
        simulation_workers_remaining=len(remaining),
        source_files=[dict(path=str(path), sha256=sha(path)) for path in sorted(sources)])
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(path)}  {path.relative_to(OUT)}\n" for path in files))
    print(json.dumps(dict(delivery_files=len(files), raw_verified=count, old_b2_verified=len(old_rows), status="passed")))


if __name__ == "__main__":
    verify()
