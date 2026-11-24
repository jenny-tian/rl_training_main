"""Copy the reviewed Isaac Lab checkpoints and their original configurations."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "models/lite3_six_terrain"
RUNS = {
    "lowstairs10000": (
        "low_stairs_training/scratch_seed42_10000", "model_updates_10000.pt",
        "419921855e24b40af058ac6642083eca8538f639e596606c51c2150ae7549a8f",
        ("params/env.yaml", "params/agent.yaml", "runtime_overrides.json",
         "low_stairs_terrain_audit.json", "reward_override_audit.json", "execution.json")),
    "zero10100": (
        "low_stairs_zero_preservation_20261115/train_001_100", "model_updates_10100.pt",
        "290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497",
        ("configuration.json", "configuration_audit.json", "execution.json")),
    "B10200": (
        "stair_allocation_20261116/B/train_001_100", "model_updates_10200.pt",
        "aba3a52e362fbb6f6c860bb8db384c12a84d13f6f39b53cae16465eaf2a84b79",
        ("configuration.json", "configuration_audit.json", "allocation_migration.json",
         "implementation_sources.json", "execution.json")),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    records = []
    models = {}
    for name, (directory, filename, expected, configs) in RUNS.items():
        source = ROOT / "logs/reference_six" / directory
        checkpoint = source / filename
        if digest(checkpoint) != expected:
            raise ValueError(f"checkpoint identity mismatch: {name}")
        execution = json.loads((source / "execution.json").read_text())
        if execution["status"] != "completed" or execution["exit_code"] != 0:
            raise ValueError(f"training did not complete: {name}")
        copies = [(checkpoint, OUTPUT / "weights" / f"{name}.pt")]
        copies.extend((source / relative, OUTPUT / "configs" / name / relative) for relative in configs)
        for original, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if digest(target) != digest(original):
                raise ValueError(f"copy mismatch: {target}")
            records.append(dict(path=str(target.relative_to(OUTPUT)), bytes=target.stat().st_size,
                                sha256=digest(target)))
        models[name] = dict(path=f"weights/{name}.pt", sha256=expected,
                            original_path=str(checkpoint.relative_to(ROOT)),
                            source_commit=execution["source_commit"])
    manifest = dict(schema=1, recommended_model="B10200", models=models,
        terrain_families=["flat", "rough", "slope", "stairs", "slope_rough", "friction_variation"],
        stair_height_m=[.02, .05], policy_observations=45, critic_observations=393, actions=12,
        reference_repository="https://github.com/DeepRoboticsLab/rl_training.git",
        reference_commit="127b4ff1081b1c7a2f08842298d06dcc5e9460e4",
        project_source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        packaging_environment={name: importlib.metadata.version(name) for name in
                               ("isaacsim", "isaaclab", "isaaclab_rl", "rsl-rl-lib", "torch", "numpy", "tensordict")},
        files=records)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records.append(dict(path="manifest.json", sha256=digest(OUTPUT / "manifest.json")))
    (OUTPUT / "SHA256SUMS").write_text("".join(
        f"{record['sha256']}  {record['path']}\n" for record in records))
    print(json.dumps(dict(models=list(models), files=len(records), output=str(OUTPUT))))


if __name__ == "__main__":
    main()
