from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from rl_training.online_adaptation.acceptance import evaluate_flat_online_experiment
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from scripts.reinforcement_learning.online_adaptation.freeze_flat_online_configuration import (
    verify_formal_configuration,
)
from scripts.reinforcement_learning.online_adaptation.run_flat_online_curriculum import (
    FLAT_RETENTION_SEEDS,
    FLAT_RETENTION_STEPS,
    flat_retention_summary_is_disabled,
    validate_track_completion_summary,
)


RESULT_FILENAMES = (
    "sealed_runs.jsonl",
    "sealed_comparison.csv",
    "acceptance.json",
    "acceptance.md",
)
MARKER_FILENAME = "sealed_evaluation_started.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--formal-configuration", required=True)
    parser.add_argument("--output-dir", default="logs/online_adaptation/sealed_evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def prepare_sealed_output(formal_configuration: str | Path, output_dir: str | Path) -> dict[str, Any]:
    formal_path = Path(formal_configuration)
    formal = verify_formal_configuration(formal_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / MARKER_FILENAME
    if marker.exists() or any((output / name).exists() for name in RESULT_FILENAMES):
        raise FileExistsError("sealed evaluation output has already been used")
    if (output / "sealed").exists() or (output / "flat").exists():
        raise FileExistsError("sealed evaluation worker results already exist")
    marker_payload = {
        "schema_version": 1,
        "formal_configuration_sha256": sha256_file(formal_path),
    }
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(marker_payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return formal


def _base_checkpoint(formal: Mapping[str, Any]) -> str:
    selected = formal["selected_configuration"]
    explicit = selected.get("base_checkpoint")
    if isinstance(explicit, str):
        return explicit
    manifest_path = Path(formal["inputs"]["base_manifest"]["path"])
    manifest = _json_object(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 1:
        raise ValueError("base manifest must identify exactly one checkpoint")
    return next(iter(files))


def _worker_command(
    *,
    formal: Mapping[str, Any],
    mode: str,
    seed: int,
    run_dir: Path,
    device: str,
    headless: bool,
    flat_probe: bool = False,
) -> list[str]:
    selected = formal["selected_configuration"]
    configuration = selected.get("target_configuration", selected.get("configuration", selected))
    if not isinstance(configuration, Mapping):
        raise ValueError("formal algorithm configuration is missing")
    checkpoint = formal["inputs"]["experiment_checkpoint"]["path"]
    worker_mode = "base" if mode == "base" else "residual_frozen"
    command = [
        sys.executable,
        str(Path(__file__).with_name("play_online_adaptation.py")),
        "--base-checkpoint", _base_checkpoint(formal),
        "--mode", worker_mode,
        "--run-dir", str(run_dir),
        "--terrain-seed", str(seed),
        "--terrain-stage", "rough_4cm",
        "--seed", str(seed),
        "--max-steps", str(FLAT_RETENTION_STEPS if flat_probe else 2_000),
        "--residual-gain", "0.0" if mode in {"base", "initial_zero_residual"} else "1.0",
        "--device", device,
    ]
    option_defaults = {
        "target_semantics": "timeout-aware",
        "n_step": 3,
        "critic_replay_mode": "dual",
        "flat_fraction_initial": 0.75,
        "flat_fraction_ready": 0.25,
        "flat_anchor_weight": 0.5,
        "max_policy_residual_mse": None,
        "flat_critic_warmup_updates": 1_000,
        "max_episodes": 100,
        "max_total_steps": 200_000,
        "max_stage_episodes": 20,
    }
    for name, default in option_defaults.items():
        value = configuration.get(name, default)
        if value is None:
            continue
        worker_name = name.replace("max_", "protocol_max_", 1) if name in {
            "max_episodes", "max_total_steps", "max_stage_episodes"
        } else name
        command.extend((f"--{worker_name.replace('_', '-')}", str(value)))
    terrain_source = configuration.get("terrain_feature_source", "geometry")
    if terrain_source == "height_scan":
        command.extend(("--terrain-feature-source", "height_scan"))
        command.extend(
            (
                "--terrain-min-valid-fraction",
                str(configuration.get("terrain_min_valid_fraction", 0.90)),
            )
        )
    command.extend(("--protocol-sha256", formal["protocol_sha256"]))
    if mode != "base":
        command.extend(("--experiment-checkpoint-in", checkpoint))
    if flat_probe:
        command.append("--flat-retention-probe")
    else:
        command.append("--require-track-completion")
        command.append("--evaluation-only")
    if headless:
        command.append("--headless")
    return command


def _run_worker(command: list[str], run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(command, check=False)
    summary_path = run_dir / "episode_summary.json"
    if result.returncode not in {0, 1} or not summary_path.is_file():
        raise RuntimeError(f"sealed worker failed in {run_dir}")
    summary = _json_object(summary_path)
    if "--require-track-completion" in command:
        validate_track_completion_summary(summary)
        seed = int(command[command.index("--terrain-seed") + 1])
        stage = command[command.index("--terrain-stage") + 1]
        if summary.get("terrain_seed") != seed or summary.get("terrain_stage") != stage:
            raise RuntimeError("sealed worker summary geometry differs from command")
        protocol_sha256 = command[command.index("--protocol-sha256") + 1]
        if summary.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("sealed worker summary protocol hash differs from command")
    return summary


def _flat_aggregate(summaries: list[Mapping[str, object]]) -> dict[str, object]:
    count = len(summaries)
    if count != len(FLAT_RETENTION_SEEDS):
        raise RuntimeError("flat evaluation is incomplete")
    if any(not flat_retention_summary_is_disabled(summary) for summary in summaries):
        raise RuntimeError("flat evaluation received strict or malformed completion evidence")
    def mean(name: str) -> float:
        values = [item.get(name) for item in summaries]
        if any(value is None for value in values):
            raise RuntimeError(f"flat summaries lack {name}")
        return sum(float(value) for value in values) / count
    return {
        "completion_rate": sum(bool(item.get("passed")) for item in summaries) / count,
        "velocity_error": mean("second_half_velocity_error"),
        "tilt_rms": mean("second_half_tilt_rms"),
        "contacts": sum(float(item.get("contacts", 0)) for item in summaries),
        "falls": sum(bool(item.get("fell")) for item in summaries),
    }


def _training_record(formal: Mapping[str, Any]) -> dict[str, object]:
    selected = formal["selected_configuration"]
    return {
        "total_episodes": int(selected["total_episodes"]),
        "total_steps": int(selected["total_steps"]),
        "falls": int(selected.get("training_falls", 0)),
    }


def write_reports(
    output_dir: str | Path,
    records: list[Mapping[str, object]],
    report: Mapping[str, object],
) -> None:
    output = Path(output_dir)
    with (output / "sealed_runs.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    with (output / "sealed_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("mode", "seed", "passed", "fell", "updates"))
        writer.writeheader()
        writer.writerows({name: record[name] for name in writer.fieldnames} for record in records)
    write_json_atomic(output / "acceptance.json", dict(report))
    checks = report["checks"]
    lines = ["# Flat-online sealed acceptance", "", f"Overall: {'PASS' if report['passed'] else 'FAIL'}", ""]
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    (output / "acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    formal = prepare_sealed_output(args.formal_configuration, args.output_dir)
    output = Path(args.output_dir)
    protocol = default_flat_online_protocol()
    records: list[dict[str, Any]] = []
    modes: dict[str, object] = {"online_learning": _training_record(formal)}
    for mode in ("base", "initial_zero_residual", "learned_frozen"):
        mode_records = []
        for seed in protocol.sealed_seeds:
            run_dir = output / "sealed" / mode / f"seed-{seed}"
            summary = _run_worker(
                _worker_command(
                    formal=formal, mode=mode, seed=seed, run_dir=run_dir,
                    device=args.device, headless=args.headless,
                ),
                run_dir,
            )
            record = {
                "mode": mode,
                "seed": seed,
                "passed": bool(summary["passed"]),
                "fell": bool(summary["fell"]),
                "updates": int(summary.get("updates", 0)),
                "deterministic": True,
            }
            records.append(record)
            mode_records.append(record)
        modes[mode] = mode_records
    flat = {}
    for mode in ("base", "learned_frozen"):
        summaries = []
        for seed in FLAT_RETENTION_SEEDS:
            run_dir = output / "flat" / mode / f"seed-{seed}"
            summaries.append(
                _run_worker(
                    _worker_command(
                        formal=formal, mode=mode, seed=seed, run_dir=run_dir,
                        device=args.device, headless=args.headless, flat_probe=True,
                    ),
                    run_dir,
                )
            )
        flat[mode] = _flat_aggregate(summaries)
    report = evaluate_flat_online_experiment(
        mode_records=modes, flat_base=flat["base"], flat_learned=flat["learned_frozen"]
    )
    write_reports(output, records, report)
    return report


def main() -> int:
    run_evaluation(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
