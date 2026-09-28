"""Bounded frozen-runner diagnostics; no policy updates or sealed terrains."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from rl_training.online_adaptation.task_objective import FiniteHorizonObjective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 109])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-objective", choices=("legacy", "finite-horizon-v1"), default="legacy")
    parser.add_argument("--task-deadline-steps", type=int, default=2000)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000 or not set(args.seeds) <= {101, 103, 107, 109, 113}:
        parser.error("diagnostics require 1..2000 steps and existing debug terrains")
    objective = None if args.task_objective == "legacy" else FiniteHorizonObjective(args.task_deadline_steps)
    if objective is not None and args.steps > objective.deadline_steps:
        parser.error("diagnostic cannot exceed the task deadline")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    base = Path(args.base_checkpoint).resolve()
    manifest = {
        "status": "running", "runs": [], "max_environment_steps": 2 * len(args.seeds) * args.steps,
        "base_sha256": sha256_file(base), "protocol_sha256": default_flat_online_protocol().sha256(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "task_objective": None if objective is None else objective.contract(),
        "task_objective_sha256": None if objective is None else objective.sha256(),
    }
    (root / "source.diff").write_text(subprocess.check_output(["git", "diff"], text=True))
    manifest["source_diff_sha256"] = sha256_file(root / "source.diff")
    manifest["runner_sha256"] = sha256_file(__file__)
    source_paths = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard",
        "source/rl_training/rl_training/online_adaptation", "scripts/reinforcement_learning/online_adaptation",
    ], text=True).splitlines()
    write_json_atomic(root / "source_files_sha256.json", {
        name: sha256_file(name) for name in sorted(set(source_paths)) if Path(name).is_file()
    })
    try:
        for seed in args.seeds:
            for algorithm in ("lora", "sac"):
                out = root / f"{algorithm}-{seed}"
                out.mkdir()
                script = "train_lora_ppo.py" if algorithm == "lora" else "play_online_adaptation.py"
                command = [sys.executable, "-u", f"scripts/reinforcement_learning/online_adaptation/{script}",
                           "--base-checkpoint", str(base), "--task", "RockyRoad-Deeprobotics-Lite3-v0",
                           "--terrain-stage", "rough_4cm", "--terrain-seed", str(seed),
                           "--terrain-split", "development", "--seed", "7", "--forward-speed", "0.3",
                           "--max-steps", str(args.steps), "--evaluation-only", "--require-track-completion",
                           "--trajectory-dtype", "float32", "--residual-rate-limit", "0.03",
                           "--headless", "--device", args.device,
                           f"--kit_args=--/renderer/activeGpu={args.device.split(':')[-1] if args.device.startswith('cuda:') else 0} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"]
                if algorithm == "lora":
                    command += ["--output", str(out), "--trial-deadline-steps", str(args.task_deadline_steps), "--stop-after-termination"]
                else:
                    command += ["--run-dir", str(out), "--mode", "base", "--terrain-feature-source", "proprio"]
                if objective is not None:
                    command += ["--task-objective", args.task_objective, "--task-deadline-steps", str(objective.deadline_steps),
                                "--task-objective-sha256", objective.sha256()]
                record = {"algorithm": algorithm, "seed": seed, "status": "running", "command": command}
                manifest["runs"].append(record)
                write_json_atomic(root / "manifest.json", manifest)
                with (out / "runtime.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                record["returncode"] = result.returncode
                if result.returncode not in (0, 1) or not (out / "summary.json").exists():
                    record["status"] = "failed"
                    raise RuntimeError(f"{algorithm}-{seed} failed: inspect runtime.log")
                summary = json.loads((out / "summary.json").read_text())
                outcome = summary if algorithm == "lora" else json.loads((out / "episode_summary.json").read_text())
                trace = json.loads((out / "trajectory/manifest.json").read_text())
                max_x = trace["metadata"]["initial_x"]
                total_reward = 0.
                for chunk in trace["chunks"]:
                    with np.load(out / "trajectory" / chunk["file"], allow_pickle=False) as data:
                        max_x = max(max_x, float(data["base_x"].max()))
                        total_reward += float(data["reward"].sum())
                assert trace["status"] == "complete" and trace["steps"] == summary["steps"]
                assert summary["updates"] == 0
                assert abs(max_x - summary["max_x"]) < 1e-6
                if "total_reward" in summary:
                    assert abs(total_reward - summary["total_reward"]) < 1e-5
                else:
                    rows = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["rows"]
                    assert len(rows) == trace["steps"]
                    assert abs(total_reward - sum(row["reward"] for row in rows)) < 1e-5
                record.update(status="complete", steps=summary["steps"], max_x=max_x,
                              passed=outcome["passed"],
                              fell=summary["fell"], reward_sum=total_reward,
                              safety_violation=summary["safety_violation"])
                write_json_atomic(root / "manifest.json", manifest)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write_json_atomic(root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
