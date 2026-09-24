"""Evaluate one fixed LoRA-PPO checkpoint on sealed terrain seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol


TASK = "RockyRoad-Deeprobotics-Lite3-v0"


def _seed_list(raw: str | None) -> tuple[int, ...]:
    protocol = default_flat_online_protocol()
    if raw is None:
        return protocol.sealed_seeds
    try:
        seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("--terrain-seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--terrain-seeds must contain unique seeds")
    if any(seed not in protocol.sealed_seeds for seed in seeds):
        raise ValueError("sealed final test accepts only protocol sealed seeds")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True, help="fixed candidate selected before sealed evaluation")
    parser.add_argument("--output-dir", required=True, help="new, empty directory for this final-test report")
    parser.add_argument("--terrain-seeds", default=None, help="comma-separated sealed seeds; defaults to the full sealed split")
    parser.add_argument("--simulation-seed", type=int, default=7)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 3), default=1)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--height-scan-adapter", action="store_true")
    parser.add_argument("--proprioceptive-context", action="store_true")
    parser.add_argument("--single-gpu-renderer", action="store_true")
    return parser


def build_command(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    script = Path(__file__).with_name("train_lora_ppo.py")
    command = [
        "/home/tjx/anaconda3/envs/env_isaaclab/bin/python", "-u", str(script),
        "--base-checkpoint", str(Path(args.base_checkpoint).resolve()),
        "--checkpoint-in", str(Path(args.checkpoint).resolve()),
        "--task", TASK,
        "--terrain-stage", "rough_4cm",
        "--terrain-seed", str(seed),
        "--terrain-split", "sealed",
        "--protocol-sha256", default_flat_online_protocol().sha256(),
        "--require-track-completion",
        "--seed", str(args.simulation_seed),
        "--forward-speed", "0.3",
        "--max-steps", str(args.max_steps),
        "--trial-deadline-steps", str(args.max_steps),
        "--rollout-steps", "256",
        "--output", str(run_dir),
        "--rank", "2",
        "--evaluation-only",
        "--headless",
        "--device", f"cuda:{args.physical_gpu}",
    ]
    if args.height_scan_adapter:
        command.append("--height-scan-adapter")
    if args.proprioceptive_context:
        command.append("--proprioceptive-context")
    if args.single_gpu_renderer:
        command.append(
            f"--kit_args=--/renderer/activeGpu={args.physical_gpu} "
            "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false "
            "--/renderer/multiGpu/maxGpuCount=1"
        )
    return command


def _successful(summary: dict[str, Any]) -> bool:
    return (
        summary.get("completion_qualified") is True
        and summary.get("fell") is False
        and summary.get("safety_violation") is False
        and summary.get("reset_count") == 0
    )


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint)
    base_checkpoint = Path(args.base_checkpoint)
    if not checkpoint.is_file() or not base_checkpoint.is_file():
        raise ValueError("base and candidate checkpoints must exist")
    seeds = _seed_list(args.terrain_seeds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "lora_ppo_sealed_final_test_v1",
        "selection_contract": "checkpoint_fixed_before_sealed_evaluation",
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "terrain_split": "sealed",
        "terrain_seeds": list(seeds),
        "simulation_seed": args.simulation_seed,
        "protocol_sha256": default_flat_online_protocol().sha256(),
        "height_scan_adapter": args.height_scan_adapter,
        "proprioceptive_context": args.proprioceptive_context,
        "results": [],
    }
    write_json_atomic(output / "sealed_test_manifest.json", report)
    for seed in seeds:
        run_dir = output / f"seed-{seed}"
        run_dir.mkdir()
        command = build_command(args, seed, run_dir)
        environment = dict(os.environ)
        environment.pop("DISPLAY", None)
        with (run_dir / "runtime.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report["results"].append({
            "terrain_seed": seed,
            "successful_completion": _successful(summary),
            "summary": summary,
        })
        write_json_atomic(output / "sealed_test_manifest.json", report)
    report["successes"] = sum(row["successful_completion"] for row in report["results"])
    report["count"] = len(report["results"])
    write_json_atomic(output / "sealed_test_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
