"""Evaluate two frozen B4 policies per seed, only after all four jobs stopped."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

from source_block_b2_protocol import ROOT, evaluations
from source_block_b4_protocol import RAW, SEEDS, TASKS, UPDATES


def completed_models():
    result = []
    for seed in SEEDS:
        for branch in TASKS:
            name = f"{branch}_seed{seed}"
            directory = RAW / "training" / name
            report = json.loads((directory / "training_report.json").read_text())
            execution = json.loads((directory / "execution.json").read_text())
            if (report["status"] != "budget_completed" or report["completed_updates"] != UPDATES
                    or execution["status"] != "completed" or not report["restored_exact"]):
                raise ValueError(f"training incomplete: {name}")
            checkpoint = directory / "model_0025.pt"
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if sha != report["output_sha256"]:
                raise ValueError(f"checkpoint changed: {name}")
            result.append(dict(name=name, branch=branch, seed=seed, checkpoint=str(checkpoint), sha256=sha))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "source/rl_training"))
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    for spec in completed_models():
        if spec["seed"] != args.seed:
            continue
        output = RAW / "evaluations" / spec["name"]
        evaluations(spec["checkpoint"], spec["sha256"], output, args.gpu)
        for domain in ("flat", "block"):
            destination = output / "controlled" / domain
            command = [sys.executable, str(ROOT / "scripts/tools/source_block_b3_diagnose.py"),
                "--model", spec["name"], "--checkpoint", spec["checkpoint"], "--sha256", spec["sha256"],
                "--domain", domain, "--mode", "controlled", "--gpu", str(args.gpu), "--output", str(destination)]
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            report = json.loads((destination / "report.json").read_text())
            if (report["status"] != "passed" or report["probe"] or not report["actor_unchanged"]
                    or report["cleanup"] != {"environment": "returned", "application": "returned"}):
                raise ValueError(f"diagnostic incomplete: {destination}")
        print(f"B4 ALL EVALUATIONS COMPLETE {spec['name']}", flush=True)


if __name__ == "__main__":
    main()
