"""Run and report base, frozen-residual, and online Lite3 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any


MODES = ("base", "residual_frozen", "online")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--task", default="HeldoutBoxes-Deeprobotics-Lite3-v0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31, 37, 41, 43, 47, 53])
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output-dir", default="logs/online_adaptation/comparison")
    return parser


def _numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]


def _mean_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric(rows, key)
    return mean(values) if values else None


def _run_one(args: argparse.Namespace, output_dir: Path, mode: str, seed: int) -> dict[str, Any]:
    from rl_training.online_adaptation.acceptance import evaluate_run

    run_dir = output_dir / mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/reinforcement_learning/online_adaptation/play_online_adaptation.py",
        "--base-checkpoint",
        args.base_checkpoint,
        "--mode",
        mode,
        "--task",
        args.task,
        "--seed",
        str(seed),
        "--max-steps",
        str(args.max_steps),
        "--device",
        args.device,
        "--run-dir",
        str(run_dir),
    ]
    if mode != "base":
        command.extend(("--residual-checkpoint", args.residual_checkpoint))
    if args.headless:
        command.append("--headless")
    if args.video:
        command.append("--video")

    return_code = 0
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        return_code = error.returncode

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "reset_count": 1,
            "fell": True,
            "parameter_changed": False,
            "safety_violation": True,
        }

    evaluation: dict[str, Any]
    try:
        evaluation = evaluate_run(summary)
    except (KeyError, TypeError, ValueError):
        evaluation = {
            "passed": False,
            "velocity_improvement": None,
            "tilt_improvement": None,
            "checks": {"complete_summary": False},
        }
    baseline_passed = (
        int(summary.get("reset_count", 1)) == 0
        and not bool(summary.get("fell", True))
        and not bool(summary.get("safety_violation", True))
    )
    return {
        "mode": mode,
        "seed": seed,
        "run_dir": str(run_dir),
        "return_code": return_code,
        "passed": bool(evaluation["passed"] if mode == "online" else baseline_passed),
        "acceptance_passed": bool(evaluation["passed"]),
        "velocity_improvement": evaluation["velocity_improvement"],
        "tilt_improvement": evaluation["tilt_improvement"],
        "checks": evaluation["checks"],
        **summary,
    }


def _write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode in MODES}
    for key, filename, ylabel in (
        ("second_half_velocity_error", "velocity_error.png", "Second-half velocity error (m/s)"),
        ("second_half_tilt_rms", "tilt_rms.png", "Second-half tilt RMS"),
    ):
        values = [_mean_or_none(grouped[mode], key) for mode in MODES]
        plt.figure(figsize=(7, 4))
        plt.bar(MODES, [float("nan") if value is None else value for value in values])
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    rates = [sum(bool(row["passed"]) for row in grouped[mode]) / max(len(grouped[mode]), 1) for mode in MODES]
    plt.figure(figsize=(7, 4))
    plt.bar(MODES, rates)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success rate")
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate.png", dpi=160)
    plt.close()


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_run_one(args, output_dir, mode, seed) for seed in args.seeds for mode in MODES]
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode in MODES}

    online_successes = sum(bool(row["acceptance_passed"]) for row in grouped["online"])
    required_successes = math.ceil(0.7 * len(args.seeds))
    online_velocity = _mean_or_none(grouped["online"], "second_half_velocity_error")
    frozen_velocity = _mean_or_none(grouped["residual_frozen"], "second_half_velocity_error")
    online_tilt = _mean_or_none(grouped["online"], "second_half_tilt_rms")
    frozen_tilt = _mean_or_none(grouped["residual_frozen"], "second_half_tilt_rms")
    aggregate_passed = (
        online_successes >= required_successes
        and online_velocity is not None
        and frozen_velocity is not None
        and online_velocity <= 0.90 * frozen_velocity
        and online_tilt is not None
        and frozen_tilt is not None
        and online_tilt <= frozen_tilt
    )
    aggregate = {
        "requested_seeds": args.seeds,
        "online_successes": online_successes,
        "required_successes": required_successes,
        "online_mean_second_half_velocity_error": online_velocity,
        "residual_frozen_mean_second_half_velocity_error": frozen_velocity,
        "online_mean_second_half_tilt_rms": online_tilt,
        "residual_frozen_mean_second_half_tilt_rms": frozen_tilt,
        "passed": aggregate_passed,
    }

    csv_fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.json").write_text(
        json.dumps({"runs": rows, "aggregate": aggregate}, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# Lite3 online adaptation comparison",
        "",
        f"Aggregate passed: **{aggregate_passed}**",
        "",
        "| Mode | Runs | Successes | Mean second-half velocity error | Mean second-half tilt RMS |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        mode_rows = grouped[mode]
        markdown.append(
            f"| {mode} | {len(mode_rows)} | {sum(bool(row['passed']) for row in mode_rows)} | "
            f"{_mean_or_none(mode_rows, 'second_half_velocity_error')} | "
            f"{_mean_or_none(mode_rows, 'second_half_tilt_rms')} |"
        )
    markdown.extend(
        (
            "",
            "Each run row in `comparison.csv` includes within-run first/second-half improvements. "
            "The aggregate fields compare online performance against the frozen residual baseline. "
            "Failed seeds are retained with their run directory and return code.",
        )
    )
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    _write_plots(output_dir, rows)
    return 0 if aggregate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
