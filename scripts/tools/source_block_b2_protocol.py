"""Registered initialization comparison and development stopping decisions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TASK = "SourceBlock-B2-Deeprobotics-Lite3-v0"
EVIDENCE = ROOT / "logs/stage1/block_b2"
Q4_LIMITS = dict(rms_xy_error=.06236837, rms_yaw_error=.08557957, falls_per_episode=.039613)


def checkpoint_for(name):
    actors = json.loads((ROOT / "docs/lite3-stage1-targeted-20261111/artifact_manifest.json").read_text())["actors"]
    spec = actors["m1998" if name == "flat1998" else "starting"]
    path = Path(spec["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("starting checkpoint hash mismatch")
    return path, spec["sha256"]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def evaluations(checkpoint, sha, output, gpu):
    output = Path(output)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "source/rl_training"))
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    commands = [
        [sys.executable, "scripts/tools/source_block_b2_eval.py", "--checkpoint", str(checkpoint),
         "--sha256", sha, "--output", str(output / "blocks"), "--gpu", str(gpu)],
        [sys.executable, "scripts/tools/stage1_flat_retention.py", "--checkpoint", str(checkpoint),
         "--sha256", sha, "--output", str(output / "flat.json"), "--gpu", str(gpu)],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=environment, check=True, timeout=2600)
    block = json.loads((output / "blocks/evaluation.json").read_text())
    flat = json.loads((output / "flat.json").read_text())
    if block["status"] != "passed" or flat["status"] != "passed":
        raise RuntimeError("evaluation execution failed")
    return block, flat


def decision(block, flat, baseline=None):
    reasons = [f"Q4/{name}: {flat['metrics'][name]} > {limit}"
               for name, limit in Q4_LIMITS.items() if flat["metrics"][name] > limit]
    floors = {}
    if baseline is not None:
        if block["initial_state_hashes_by_column"] != baseline["initial_state_hashes_by_column"]:
            raise ValueError("development initial states do not match the baseline")
        for c in (4, 5):
            floors[str(c)] = max(0, baseline["columns"][c]["complete"] - 2)
            if block["columns"][c]["complete"] < floors[str(c)]:
                reasons.append(f"column {c} completion {block['columns'][c]['complete']}/32 below baseline floor {floors[str(c)]}/32")
    return dict(continue_training=not reasons, reasons=reasons, baseline_stair_floors=floors,
                q4_limits=Q4_LIMITS, upstairs=block["columns"][4], downstairs=block["columns"][5],
                rough=[block["columns"][i] for i in (6, 15)], flat_metrics=flat["metrics"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("flat1998", "terrain3497"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    checkpoint, sha = checkpoint_for(args.baseline)
    output = EVIDENCE / "baseline" / args.baseline
    block, flat = evaluations(checkpoint, sha, output, args.gpu)
    result = decision(block, flat)
    save_json(output / "decision.json", result)
    print(json.dumps(result, indent=2), flush=True)
