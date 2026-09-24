"""Train one height-scan PPO teacher across the formal train terrains."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import (
    FORMAL_TRAIN_SEEDS,
    default_flat_online_protocol,
)


TASK = "FlatOnline-Deeprobotics-Lite3-v0"
TEACHER_CONTRACT = "height_scan_privileged_teacher_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps-per-seed", type=int, required=True)
    parser.add_argument("--evaluation-steps", type=int, default=4_000)
    parser.add_argument("--smoke", action="store_true",
                        help="validate the Isaac pipeline only; output is never a qualified teacher")
    parser.add_argument("--terrain-stage", default="rough_4cm")
    parser.add_argument("--simulation-seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--isaaclab-sh", default="/home/tjx/IsaacLab/isaaclab.sh")
    return parser


def build_command(args: argparse.Namespace, seed: int, run_dir: Path,
                  previous_checkpoint: Path | None, *, evaluation: bool = False) -> list[str]:
    worker = Path(__file__).with_name("train_lora_ppo.py")
    command = [
        args.isaaclab_sh, "-p", str(worker),
        "--base-checkpoint", str(Path(args.base_checkpoint).resolve()),
        "--task", TASK,
        "--terrain-stage", args.terrain_stage,
        "--terrain-seed", str(seed),
        "--terrain-split", "train",
        "--protocol-sha256", default_flat_online_protocol().sha256(),
        "--seed", str(args.simulation_seed),
        "--max-steps", str(args.evaluation_steps if evaluation else args.steps_per_seed),
        "--rollout-steps", str(args.rollout_steps),
        "--output", str(run_dir),
        "--rank", str(args.rank),
        "--height-scan-adapter",
        "--teacher-training",
        "--headless",
        "--device", args.device,
    ]
    if evaluation:
        command.extend((
            "--checkpoint-in", str(previous_checkpoint.resolve()),
            "--evaluation-only", "--require-track-completion",
            "--trial-deadline-steps", str(args.evaluation_steps),
            "--stop-after-termination",
        ))
    else:
        command.append("--continue-after-termination")
    if previous_checkpoint is not None and not evaluation:
        command.extend(("--checkpoint-in", str(previous_checkpoint.resolve())))
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps_per_seed < 1 or args.rollout_steps < 1:
        raise SystemExit("step budgets must be positive")
    if args.evaluation_steps < 3_050:
        raise SystemExit("--evaluation-steps must cover the 18 m track plus qualification margin")
    if args.steps_per_seed < 2_000 and not args.smoke:
        raise SystemExit("budgets below 2000 steps require --smoke and cannot qualify a teacher")
    if args.rank < 1 or args.rank > 10:
        raise SystemExit("height-scan teacher rank must be in [1, 10]")
    base_checkpoint = Path(args.base_checkpoint).resolve()
    if not base_checkpoint.is_file():
        raise SystemExit("base checkpoint does not exist")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = default_flat_online_protocol()
    manifest = {
        "schema": "rgls_height_scan_teacher_run_v1",
        "teacher_contract": TEACHER_CONTRACT,
        "terrain_split": "train",
        "train_seeds": list(FORMAL_TRAIN_SEEDS),
        "development_seeds": [109],
        "sealed_seeds": list(protocol.sealed_seeds),
        "protocol_sha256": protocol.sha256(),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "working_tree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()),
        "steps_per_seed": args.steps_per_seed,
        "evaluation_steps": args.evaluation_steps,
        "run_kind": "smoke" if args.smoke else "teacher_training",
        "terrain_stage": args.terrain_stage,
        "runs": [],
        "frozen_probes": [],
        "status": "running",
    }
    manifest_path = output / "teacher_manifest.json"
    write_json_atomic(manifest_path, manifest)
    previous_checkpoint = None
    active_record = None
    try:
        for seed in FORMAL_TRAIN_SEEDS:
            run_dir = output / f"seed-{seed}"
            command = build_command(args, seed, run_dir, previous_checkpoint)
            record = {"terrain_seed": seed, "command": command, "status": "running"}
            active_record = record
            manifest["runs"].append(record)
            write_json_atomic(manifest_path, manifest)
            subprocess.run(command, check=True)
            checkpoint = run_dir / "latest.pt"
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"teacher worker did not produce {summary_path}")
            import json

            training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not checkpoint.is_file():
                raise RuntimeError(f"teacher worker did not produce {checkpoint}")
            import torch

            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (
                saved.get("teacher_contract") != TEACHER_CONTRACT
                or saved.get("terrain_split") != "train"
                or saved.get("terrain_seed") != seed
                or saved.get("protocol_sha256") != protocol.sha256()
                or saved.get("base_checkpoint_sha256") != manifest["base_checkpoint_sha256"]
                or saved.get("actor_observation_schema") != "base45_heightscan10_gated_residual_v2"
            ):
                raise RuntimeError(f"teacher checkpoint contract validation failed for seed {seed}")
            record.update({"status": "trained",
                           "checkpoint": str(checkpoint),
                           "checkpoint_sha256": sha256_file(checkpoint),
                           "training_steps": training_summary.get("steps"),
                           "training_resets": training_summary.get("reset_count")})
            previous_checkpoint = checkpoint
            write_json_atomic(manifest_path, manifest)
        if not args.smoke:
            for seed in FORMAL_TRAIN_SEEDS:
                run_dir = output / f"frozen-probe-seed-{seed}"
                command = build_command(args, seed, run_dir, previous_checkpoint, evaluation=True)
                record = {"terrain_seed": seed, "command": command, "status": "running"}
                active_record = record
                manifest["frozen_probes"].append(record)
                write_json_atomic(manifest_path, manifest)
                subprocess.run(command, check=True)
                summary_path = run_dir / "summary.json"
                if not summary_path.is_file():
                    raise RuntimeError(f"teacher probe did not produce {summary_path}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                qualified = (
                    summary.get("passed") is True
                    and summary.get("completion_qualification_passed") is True
                    and summary.get("fell") is False
                    and summary.get("safety_violation") is False
                    and summary.get("reset_count") == 0
                )
                record.update({"status": "complete" if qualified else "failed",
                               "qualified": qualified,
                               "completion_reached": bool(summary.get("completion_reached")),
                               "steps": summary.get("steps"),
                               "stop_reason": summary.get("stop_reason"),
                               "max_x": summary.get("max_x"),
                               "completion_target_x": summary.get("completion_target_x")})
                write_json_atomic(manifest_path, manifest)
                if not qualified:
                    raise RuntimeError(f"frozen teacher qualification failed for train seed {seed}")
    except BaseException as error:
        if active_record is not None:
            active_record["status"] = "failed"
            active_record["error"] = f"{type(error).__name__}: {error}"
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json_atomic(manifest_path, manifest)
        raise
    manifest["status"] = "smoke_complete_unqualified" if args.smoke else "complete"
    manifest["final_checkpoint"] = None if args.smoke else str(previous_checkpoint)
    manifest["final_checkpoint_sha256"] = None if args.smoke else sha256_file(previous_checkpoint)
    write_json_atomic(manifest_path, manifest)
    print(f"saved train-only height-scan teacher manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
