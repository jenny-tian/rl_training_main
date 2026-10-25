"""Run the five registered L2 ascending-stair diagnostics, without training.

The orchestrator launches one bounded, headless runtime episode per registered
geometry/state seed pair. Each episode writes an independent non-overwriting
report. The aggregate is diagnostic evidence only and never a qualification
decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/rl_training"))

from scripts.tools.runtime_progress import RuntimeProgressJournal
from scripts.tools.source_family_runtime_smoke import (
    MANIFEST,
    MANIFEST_SHA256,
    ROOT as RUNTIME_ROOT,
    select_cell,
)

TARGET_CELL = "multi_step_stairs_L2_ascending"
ISAACLAB_SH = Path("/home/tjx/IsaacLab/isaaclab.sh")
RUNTIME_SCRIPT = ROOT / "scripts/tools/source_family_runtime_smoke.py"
MULTI_GPU_OFF = (
    "--/renderer/activeGpu=0 "
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1"
)


def registered_seed_pairs(cell: dict[str, Any]) -> list[dict[str, int]]:
    """Return the exact registered, zipped geometry/state seed pairs."""

    geometry_lo, geometry_hi = cell["geometry_seed_range"]
    state_lo, state_hi = cell["initial_state_seed_range"]
    geometry_seeds = list(range(int(geometry_lo), int(geometry_hi) + 1))
    state_seeds = list(range(int(state_lo), int(state_hi) + 1))
    if len(geometry_seeds) != len(state_seeds):
        raise ValueError("registered geometry and initial-state seed counts differ")
    if len(geometry_seeds) != int(cell["episodes"]):
        raise ValueError("registered seed count does not match the cell episode count")
    if cell.get("seed_pairing") != "zip_geometry_initial_state":
        raise ValueError("only the registered zipped seed-pairing rule is supported")
    return [
        {"geometry_seed": geometry_seed, "initial_state_seed": state_seed}
        for geometry_seed, state_seed in zip(geometry_seeds, state_seeds, strict=True)
    ]


def build_episode_command(
    pair: dict[str, int], *, output: Path, device: str, heartbeat_steps: int
) -> list[str]:
    """Build one child command using only a registered L2 seed pair."""

    return [
        str(ISAACLAB_SH),
        "-p",
        str(RUNTIME_SCRIPT),
        "--cell-id",
        TARGET_CELL,
        "--output",
        str(output),
        "--geometry-seed",
        str(pair["geometry_seed"]),
        "--initial-state-seed",
        str(pair["initial_state_seed"]),
        "--heartbeat-steps",
        str(heartbeat_steps),
        "--headless",
        "--device",
        device,
        "--kit_args",
        MULTI_GPU_OFF,
    ]


def _summary_record(report: dict[str, Any], pair: dict[str, int]) -> dict[str, Any]:
    if report.get("status") != "bounded_runtime_diagnostic_completed":
        raise RuntimeError(f"episode report is not complete: {report.get('status')}")
    if report.get("cell", {}).get("cell_id") != TARGET_CELL:
        raise RuntimeError("child report has the wrong terrain cell")
    record = report.get("record", {})
    if record.get("geometry_seed") != pair["geometry_seed"]:
        raise RuntimeError("child report geometry seed does not match the requested pair")
    if record.get("initial_state_seed") != pair["initial_state_seed"]:
        raise RuntimeError("child report initial-state seed does not match the requested pair")
    if report.get("policy_updated") is not False:
        raise RuntimeError("child report indicates a policy update")
    if report.get("sealed_accessed") is not False or report.get("qualification_claim") is not False:
        raise RuntimeError("child report crossed the diagnostic scope boundary")
    if report.get("frozen_module_sha256_before") != report.get("frozen_module_sha256_after"):
        raise RuntimeError("child report frozen policy hashes differ")
    return {
        **pair,
        **{
            key: record.get(key)
            for key in (
                "steps",
                "completed",
                "termination_reason",
                "termination_terms",
                "forward_progress_m",
                "max_forward_progress_m",
                "max_projected_gravity_xy",
                "lateral_offset_peak_m",
                "nonfoot_contact_count",
                "nonfoot_contact_max_force_n",
                "safety_filter_projection_fraction",
                "invalid_state_count",
            )
        },
        "completed_step_index": record.get("family_metrics", {}).get("completed_step_index"),
        "front_foot_collision_count": record.get("family_metrics", {}).get(
            "front_foot_collision_count"
        ),
        "report_sha256": report.get("_report_sha256"),
        "report_path": report.get("_report_path"),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize diagnostic outcomes without applying qualification thresholds."""

    termination_counts: dict[str, int] = {}
    step_index_counts: dict[str, int] = {}
    for record in records:
        reason = str(record.get("termination_reason"))
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
        step_index = record.get("completed_step_index")
        key = "unavailable" if step_index is None else str(step_index)
        step_index_counts[key] = step_index_counts.get(key, 0) + 1
    return {
        "episode_count": len(records),
        "termination_reason_counts": termination_counts,
        "completed_step_index_counts": step_index_counts,
        "qualification_claim": False,
        "interpretation": (
            "descriptive multi-seed evidence only; no promotion, rollback, or family-removal "
            "threshold is applied by this runner"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heartbeat-steps", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="write the plan without launching Isaac Sim")
    args = parser.parse_args(argv)
    if not 1 <= args.heartbeat_steps <= 200:
        parser.error("--heartbeat-steps must be in [1, 200]")
    if not ISAACLAB_SH.is_file():
        parser.error(f"Isaac Lab launcher does not exist: {ISAACLAB_SH}")

    manifest, cell = select_cell(TARGET_CELL)
    if cell.get("episodes") != 5:
        parser.error("the registered L2 ascending cell must specify exactly five episodes")
    pairs = registered_seed_pairs(cell)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"refusing to overwrite existing output directory: {output_dir}")

    commands = []
    for index, pair in enumerate(pairs, start=1):
        report_path = output_dir / (
            f"episode_{index:02d}_geometry_{pair['geometry_seed']}_"
            f"state_{pair['initial_state_seed']}.json"
        )
        commands.append({
            "episode_index": index,
            **pair,
            "report_path": str(report_path),
            "log_path": str(output_dir / f"episode_{index:02d}.log"),
            "command": build_episode_command(
                pair,
                output=report_path,
                device=args.device,
                heartbeat_steps=args.heartbeat_steps,
            ),
        })

    plan = {
        "schema_version": "lite3_l2_stair_multiseed_plan_v1",
        "status": "prepared_not_started" if args.dry_run else "started_not_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_cell": TARGET_CELL,
        "preparation_manifest_sha256": MANIFEST_SHA256,
        "checkpoint_sha256": manifest["base_checkpoint"]["sha256"],
        "episode_count": len(commands),
        "headless": True,
        "device": args.device,
        "heartbeat_steps": args.heartbeat_steps,
        "policy_updated": False,
        "sealed_accessed": False,
        "qualification_claim": False,
        "dry_run": args.dry_run,
        "source_sha256": {
            str(path): hashlib_sha256(path)
            for path in (
                Path(__file__).resolve(),
                RUNTIME_SCRIPT,
                Path(__file__).resolve().parents[0] / "runtime_progress.py",
            )
        },
        "episodes": commands,
    }
    journal = RuntimeProgressJournal(output_dir / "progress.json", plan)
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "status": plan["status"], "episodes": len(commands)}))
        return 0

    completed: list[dict[str, Any]] = []
    try:
        for command in commands:
            log_path = Path(command["log_path"])
            journal.update(
                phase="running_episode",
                current_episode=command["episode_index"],
                current_geometry_seed=command["geometry_seed"],
                current_initial_state_seed=command["initial_state_seed"],
                completed_episode_count=len(completed),
            )
            with log_path.open("x", encoding="utf-8") as log:
                result = subprocess.run(
                    command["command"],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"episode {command['episode_index']} exited with {result.returncode}; "
                    f"see {log_path}"
                )
            report_path = Path(command["report_path"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            from scripts.reinforcement_learning.online_adaptation.run_source_qualification_scan import _sha256

            report["_report_path"] = str(report_path)
            report["_report_sha256"] = _sha256(report_path)
            summary = _summary_record(
                report,
                {"geometry_seed": command["geometry_seed"], "initial_state_seed": command["initial_state_seed"]},
            )
            summary["episode_index"] = command["episode_index"]
            completed.append(summary)
            journal.update(
                phase="episode_completed",
                completed_episode_count=len(completed),
                completed_episodes=completed,
            )

        aggregate = aggregate_records(completed)
        source_paths = [Path(path) for path in plan["source_sha256"]]
        source_hashes_after = {str(path): hashlib_sha256(path) for path in source_paths}
        if source_hashes_after != plan["source_sha256"]:
            raise RuntimeError("runner source changed during the multi-seed diagnostic")
        journal.finalize({
            **plan,
            "status": "bounded_multiseed_diagnostic_completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "completed",
            "source_sha256_after": source_hashes_after,
            "episodes_completed": completed,
            "aggregate": aggregate,
        })
        print(json.dumps({
            "output_dir": str(output_dir),
            "status": "bounded_multiseed_diagnostic_completed",
            "episode_count": len(completed),
            "termination_reason_counts": aggregate["termination_reason_counts"],
        }))
        return 0
    except BaseException as error:
        journal.update(
            status="failed_or_interrupted",
            phase="failed",
            completed_episode_count=len(completed),
            completed_episodes=completed,
            failure={
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def hashlib_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
