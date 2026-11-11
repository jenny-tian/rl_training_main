"""Seal B4 delivery after all bounded workers exit, preserving earlier evidence."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from source_block_b3_analyze import save
from source_block_b4_analyze import read, sha, verify_delivery
from source_block_b4_protocol import OUT, ROOT


def main():
    artifacts = read(OUT / "artifact_manifest.json")
    for row in artifacts:
        path = Path(row["path"])
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"raw evidence changed: {path}")
    summary = read(OUT / "summary.json")
    if summary["training_interactions"] != 9830400 or summary["automatic_extension"]:
        raise ValueError("trial budget or stop rule changed")
    previous = {}
    for name in ("lite3-block-b2-initialization-20261111", "lite3-block-b3-diagnosis-20261111"):
        previous[name] = verify_delivery(ROOT / "docs" / name)
    processes = subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines()
    workers = {"source_block_b4_train.py", "source_block_b4_evaluate.py", "source_block_b4_check.py",
               "source_block_b3_diagnose.py", "source_block_b2_eval.py", "stage1_flat_retention.py"}
    remaining = [line for line in processes if any(Path(word).name in workers for word in line.split())]
    if remaining:
        raise RuntimeError(f"evaluation or training processes remain: {remaining}")
    suite = ET.parse(OUT / "validation/cpu.xml").getroot().find("testsuite")
    if suite is None or any(int(suite.attrib[key]) for key in ("failures", "errors", "skipped")):
        raise ValueError("CPU suite did not fully pass")
    validation = read(OUT / "validation.json")
    validation.update(cpu_tests=int(suite.attrib["tests"]), cpu_failures=0,
        simulation_workers_remaining=0, raw_files_reverified=len(artifacts), previous_deliveries_verified=previous)
    sources = list((ROOT / "scripts/tools").glob("source_block_b4_*.py"))
    sources += [ROOT / "scripts/tools" / name for name in (
        "source_block_b3_diagnose.py", "source_block_b3_protocol.py", "source_block_b2_eval.py",
        "source_block_b2_protocol.py", "source_block_runtime.py", "stage1_flat_retention.py")]
    base = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    sources += [base / name for name in (
        "mdp/rewards.py", "mdp/source_block.py", "config/quadruped/deeprobotics_lite3/source_block_env_cfg.py",
        "config/quadruped/deeprobotics_lite3/flat_env_cfg.py", "config/quadruped/deeprobotics_lite3/__init__.py",
        "config/quadruped/deeprobotics_lite3/agents/rsl_rl_ppo_cfg.py")]
    sources += [Path("/home/tjx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py"),
        Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py"),
        Path("/home/tjx/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py")]
    validation["source_files"] = [dict(path=str(path), sha256=sha(path)) for path in sorted(sources)]
    save(OUT / "validation.json", validation)
    paths = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    for path in paths:
        if path.suffix in (".md", ".py") and any(c in path.read_text() for c in (chr(0x201c), chr(0x201d))):
            raise ValueError(f"forbidden curly quotation mark: {path}")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(path)}  {path.relative_to(OUT)}\n" for path in paths))
    print(json.dumps(dict(status="passed", delivery_files=len(paths), raw_files=len(artifacts),
                         cpu_tests=validation["cpu_tests"], remaining_workers=0)))


if __name__ == "__main__":
    main()
