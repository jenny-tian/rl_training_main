"""Run Source startup smoke with cooking/fatal-exception monitoring.

This supervisor is intended for an explicitly approved startup check. It does
not load a policy or start PPO training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any


_POLL_INTERVAL_S = 0.1
_TERMINATE_GRACE_S = 10.0
_GPU_SAMPLE_TIMEOUT_S = 1.0
_STARTUP_TRACE_MARKER = "initial-reset-complete"
_FATAL_TRACE_LINES = frozenset({
    "[parallel-terrain-smoke] fatal-exception-recorded",
    "[parallel-terrain-smoke] simulation-app-close-exception",
})


def find_collision_cooking_failure(text: str) -> str | None:
    """Return the first PhysX triangle-mesh cooking failure line, if present."""

    for line in text.splitlines():
        lowered = line.casefold()
        if "/world/ground/terrain/" not in lowered:
            continue
        if "cooking failure for" in lowered or "unable to create triangle mesh for" in lowered:
            return line.strip()
    return None


def find_fatal_exception(text: str) -> str | None:
    """Reject explicit diagnostic failures and PhysX contact-capacity loss."""
    for line in text.splitlines():
        if line.strip() in _FATAL_TRACE_LINES:
            return line.strip()
        lowered = line.casefold()
        if 'dropping contacts' in lowered or 'dropped contacts' in lowered:
            return line.strip()
        capacity = any(word in lowered for word in (
            'contact buffer', 'contact data buffer', 'patch buffer', 'contactbuffercapacity',
            'patchbuffercapacity', 'foundlostpairscapacity', 'gpumaxrigidcontactcount',
            'gpumaxrigidpatchcount', 'max_contact_data_count', 'contact data count',
            'maxnbcontactdatablocks',
        ))
        failure = any(word in lowered for word in ('overflow', 'exceed', 'increase', 'insufficient', 'buffer is full'))
        if capacity and failure:
            return line.strip()
    return None


def _fatal_exception_record(console_text: str, trace_path: Path, phase: str, started: float) -> dict | None:
    marker = find_fatal_exception(console_text)
    stream = "console"
    if marker is None:
        try:
            marker = find_fatal_exception(trace_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        stream = "trace"
    if marker is None:
        return None
    return {"marker": marker, "stream": stream, "detected_phase": phase,
            "detected_elapsed_s": time.monotonic() - started}


def _scan_final_console(console_path: Path, finder=find_collision_cooking_failure) -> str | None:
    """Scan all output after cleanup, including writes made by SIGTERM handlers."""

    with console_path.open(encoding="utf-8", errors="replace") as console_file:
        for line in console_file:
            marker = finder(line)
            if marker is not None:
                return marker
    return None


def _trace_has_completed_initial_reset(trace_path: Path) -> bool:
    try:
        return _STARTUP_TRACE_MARKER in trace_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False


def _terminate_process_group(process: subprocess.Popen[Any], grace_s: float) -> None:
    if os.name == "posix":
        # The leader may have exited while descendants still own GPU resources.
        # Wait for the group, not just the leader, before escalating to SIGKILL.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + grace_s
        while True:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(min(0.02, remaining_s))
        process.wait()
        return

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def run_monitored_process(
    command: Sequence[str],
    *,
    console_path: Path,
    trace_path: Path,
    startup_timeout_s: float,
    total_timeout_s: float,
    poll_interval_s: float = _POLL_INTERVAL_S,
    terminate_grace_s: float = _TERMINATE_GRACE_S,
    sample_interval_s: float | None = None,
    sample_callback: Callable[[float], None] | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Stop on cooking errors, explicit fatal markers or the fixed timeouts."""

    if not command:
        raise ValueError("command must not be empty")
    if startup_timeout_s <= 0 or total_timeout_s <= 0:
        raise ValueError("startup and total timeouts must be positive")
    if poll_interval_s <= 0 or terminate_grace_s < 0:
        raise ValueError("poll interval must be positive and terminate grace nonnegative")
    if sample_callback is not None and (
        sample_interval_s is None or sample_interval_s <= 0
    ):
        raise ValueError("a positive sample interval is required with a sample callback")

    console_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stop_reason: str | None = None
    failure_marker: str | None = None
    collision_failure: dict[str, Any] | None = None
    fatal_exception: dict[str, Any] | None = None
    stop_requested_elapsed_s: float | None = None
    recent_output = ""
    next_sample_s = 0.0
    with console_path.open("xb") as console_file:
        process = subprocess.Popen(
            list(command),
            cwd=None if cwd is None else str(cwd),
            env=env,
            stdout=console_file,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        try:
            with console_path.open("rb") as monitor_file:
                while True:
                    new_bytes = monitor_file.read()
                    combined_output = recent_output
                    if new_bytes:
                        new_text = new_bytes.decode("utf-8", errors="replace")
                        combined_output = recent_output[-1024:] + new_text
                        failure_marker = find_collision_cooking_failure(
                            combined_output
                        )
                        recent_output = (recent_output + new_text)[-8192:]
                        if failure_marker is not None:
                            stop_reason = "physx_cooking_failure"
                            collision_failure = {
                                "marker": failure_marker,
                                "detected_phase": "monitor",
                                "detected_elapsed_s": time.monotonic() - started,
                            }
                            break

                    fatal_exception = _fatal_exception_record(combined_output, trace_path, "monitor", started)
                    if fatal_exception is not None:
                        stop_reason = "fatal_exception"
                        failure_marker = fatal_exception["marker"]
                        break

                    if process.poll() is not None:
                        # Inspect the full final write before truncating history.
                        final_bytes = monitor_file.read()
                        if final_bytes:
                            failure_marker = find_collision_cooking_failure(
                                recent_output + final_bytes.decode("utf-8", errors="replace")
                            )
                            if failure_marker is not None:
                                stop_reason = "physx_cooking_failure"
                                collision_failure = {
                                    "marker": failure_marker,
                                    "detected_phase": "process_exit",
                                    "detected_elapsed_s": time.monotonic() - started,
                                }
                        fatal_exception = _fatal_exception_record(
                            recent_output + final_bytes.decode("utf-8", errors="replace"),
                            trace_path, "process_exit", started,
                        )
                        if fatal_exception is not None and stop_reason is None:
                            stop_reason = "fatal_exception"
                            failure_marker = fatal_exception["marker"]
                        break

                    elapsed_s = time.monotonic() - started
                    if (
                        sample_callback is not None
                        and sample_interval_s is not None
                        and elapsed_s >= next_sample_s
                    ):
                        try:
                            sample_callback(elapsed_s)
                        except Exception as error:
                            stop_reason = "resource_monitor_failure"
                            failure_marker = f"resource sampling failed: {error}"
                            break
                        next_sample_s = elapsed_s + sample_interval_s
                        elapsed_s = time.monotonic() - started
                    if (
                        elapsed_s >= startup_timeout_s
                        and not _trace_has_completed_initial_reset(trace_path)
                    ):
                        stop_reason = "startup_timeout"
                    elif elapsed_s >= total_timeout_s:
                        stop_reason = "total_timeout"
                    if stop_reason is not None:
                        break
                    time.sleep(poll_interval_s)
        finally:
            # Also clean up on monitoring errors, interrupts, or an exited leader.
            if stop_reason is not None:
                stop_requested_elapsed_s = time.monotonic() - started
            _terminate_process_group(process, terminate_grace_s)

    exit_code = process.wait()
    final_marker = _scan_final_console(console_path)
    if collision_failure is None and final_marker is not None:
        # This did not cause the stop. Preserve the real trigger (e.g. timeout)
        # and separately record the diagnostic that became visible on cleanup.
        collision_failure = {
            "marker": final_marker,
            "detected_phase": "post_termination",
            "detected_elapsed_s": time.monotonic() - started,
        }
    if fatal_exception is None:
        # A late fatal marker may appear when SIGTERM flushes buffers.
        # Keep the original timeout/resource trigger and still reject success.
        final_fatal_marker = _scan_final_console(console_path, find_fatal_exception)
        fatal_exception = _fatal_exception_record(final_fatal_marker or "", trace_path, "post_termination", started)
    return {
        "exit_code": exit_code,
        "stop_reason": stop_reason,
        "elapsed_s": time.monotonic() - started,
        "startup_timeout_s": startup_timeout_s,
        "total_timeout_s": total_timeout_s,
        "monitor_poll_interval_s": poll_interval_s,
        "failure_marker": failure_marker,
        "stop_requested_elapsed_s": stop_requested_elapsed_s,
        "collision_cooking_failure": collision_failure,
        "fatal_exception": fatal_exception,
        "terminate_grace_s": terminate_grace_s,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture(command: Sequence[str], *, cwd: Path, timeout_s: float = 5.0) -> str:
    return subprocess.check_output(
        list(command), cwd=str(cwd), text=True, stderr=subprocess.STDOUT,
        timeout=timeout_s,
    ).strip()


def _gpu_preflight(gpu_index: int, *, root: Path) -> tuple[str, str]:
    gpu_snapshot = _capture(
        [
            "nvidia-smi",
            f"--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        cwd=root,
    )
    gpu_rows = [line.split(", ") for line in gpu_snapshot.splitlines()]
    matching = [row for row in gpu_rows if row and row[0] == str(gpu_index)]
    if len(matching) != 1:
        raise RuntimeError(f"could not uniquely identify GPU index {gpu_index}")
    row = matching[0]
    if len(row) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {row}")
    gpu_uuid = row[1]
    available_mib = int(row[4].split()[0]) - int(row[3].split()[0])
    if available_mib < 8 * 1024:
        raise RuntimeError(
            f"GPU {gpu_index} has only {available_mib} MiB available; "
            "at least 8192 MiB is required"
        )
    process_snapshot = _capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        cwd=root,
    )
    if any(line.split(", ", maxsplit=1)[0] == gpu_uuid for line in process_snapshot.splitlines()):
        raise RuntimeError(f"GPU {gpu_index} already has a compute process")
    return gpu_snapshot, process_snapshot


def lifecycle_report_passed(report) -> bool:
    return (
        isinstance(report, dict)
        and report.get("status") == "reset_lifecycle_passed"
        and report.get("assignment_count") == 363
        and report.get("complete_blocks") == 3
        and report.get("direct_reset_count") == 300
        and report.get("automatic_reset_ids") == [2, 23, 58]
        and report.get("reset_batch_count") == 10
    )


def finalize_run(*, run_dir: Path, root: Path, execution: dict,
                 source_hashes: dict, dependency_hashes: dict,
                 num_envs: int, shared_bank: bool = False, reset_lifecycle: bool = False) -> dict:
    """Persist a fail-closed outcome even when post-run validation raises.

    Called only after child cleanup. No simulator imports or process launches.
    Filesystem write failures still propagate: an unwritable archive cannot be guaranteed.
    """
    result_path = run_dir / "result.json"
    hashes_unchanged = False
    smoke_passed = False
    finalization_error = None
    execution = dict(execution)
    execution.update(sampled_gpu_peak_mib=None, gpu_sampling_interval_s=1.0,
                     gpu_sample_timeout_s=_GPU_SAMPLE_TIMEOUT_S)
    phase = "hash_verification"
    try:
        hashes_unchanged = all(
            _sha256(root / path) == expected for path, expected in source_hashes.items()
        ) and all(_sha256(Path(path)) == expected for path, expected in dependency_hashes.items())
        phase = "gpu_sample_summary"
        samples = [json.loads(line) for line in (run_dir / "gpu_samples.jsonl").read_text().splitlines()
                   if line.strip()]
        execution["sampled_gpu_peak_mib"] = max(
            (sample["memory_used_mib"] for sample in samples), default=None)
        phase = "report_validation"
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if shared_bank:
            from source_shared_report_validation import shared_report_passed

            smoke_passed = shared_report_passed(report, reset_lifecycle=reset_lifecycle)
        else:
            smoke_passed = (
                isinstance(report, dict)
                and report.get("status") == "diagnostic_smoke_passed"
                and report.get("num_envs") == num_envs
                and report.get("zero_action_steps") == (2 if reset_lifecycle else 1)
                and (not reset_lifecycle or lifecycle_report_passed(report.get("reset_lifecycle")))
                and report.get("policy_loaded") is False and report.get("ppo_updates") == 0
            )
    except Exception as exc:
        # Import, schema, read and validation failures must never omit the outcome.
        finalization_error = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
    execution["code_hashes_unchanged"] = hashes_unchanged
    execution["finalization_error"] = finalization_error
    passed = (
        finalization_error is None and hashes_unchanged and smoke_passed
        and execution["stop_reason"] is None and execution["collision_cooking_failure"] is None
        and execution["fatal_exception"] is None and execution["exit_code"] == 0
    )
    outcome = {
        "status": "diagnostic_smoke_passed" if passed else "startup_check_failed",
        "stop_reason": execution["stop_reason"], "failure_marker": execution["failure_marker"],
        "collision_cooking_failure": execution["collision_cooking_failure"],
        "fatal_exception": execution["fatal_exception"],
        "code_hashes_unchanged": hashes_unchanged,
        "result_json_produced": result_path.is_file(),
        "report_validation_passed": smoke_passed,
        "finalization_error": finalization_error, "training_started": False,
    }
    (run_dir / "execution.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "outcome.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, choices=(2, 8, 16, 32, 60), required=True)
    parser.add_argument("--reset-lifecycle", action="store_true")
    parser.add_argument("--shared-bank", action="store_true")
    parser.add_argument("--seed", type=int, default=130_001)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--total-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.shared_bank and (args.num_envs != 2 or args.seed != 130196):
        parser.error("--shared-bank requires --num-envs 2 --seed 130196")
    if args.num_envs == 2 and not args.shared_bank:
        parser.error("this supervisor requires --shared-bank for two environments")
    if args.reset_lifecycle and not args.shared_bank and args.num_envs != 60:
        parser.error("--reset-lifecycle requires --num-envs 60")
    if args.seed < 0 or args.gpu_index < 0:
        parser.error("--seed and --gpu-index must be nonnegative")
    if args.startup_timeout_s <= 0 or args.total_timeout_s <= 0:
        parser.error("timeouts must be positive")

    supervisor_path = Path(__file__).resolve()
    root = supervisor_path.parents[2]
    smoke_script = supervisor_path.with_name("source_parallel_terrain_startup_smoke.py")
    run_dir = args.run_dir.resolve()

    dependency_hashes = {}
    dependency_audit_path = root / "docs/lite3-source-shared-bank-preparation-20261028/dependency_audit.json"
    if args.shared_bank:
        dependency_audit = json.loads(dependency_audit_path.read_text(encoding="utf-8"))
        for item in dependency_audit["dependencies"]:
            path = Path(item["path"])
            if _sha256(path) != item["sha256"]:
                raise RuntimeError(f"shared-bank dependency changed since offline review: {path}")
            dependency_hashes[str(path)] = item["sha256"]

    if _capture(["git", "status", "--porcelain"], cwd=root):
        raise RuntimeError("refusing to launch from a dirty worktree")
    model_root = root / "deep_robotics_model"
    if _capture(["git", "-C", str(model_root), "status", "--porcelain"], cwd=root):
        raise RuntimeError("refusing to launch with a dirty robot-model submodule")
    gpu_snapshot, gpu_process_snapshot = _gpu_preflight(args.gpu_index, root=root)
    asset_audit_path = root / "docs/lite3-usd-collision-geometry-audit-20261027.json"
    asset_audit = json.loads(asset_audit_path.read_text(encoding="utf-8"))
    for asset in asset_audit["assets"].values():
        asset_path = root / asset["path"]
        if _sha256(asset_path) != asset["sha256"]:
            raise RuntimeError(f"robot-model asset hash mismatch: {asset['path']}")

    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "result.json"
    trace_path = run_dir / "trace.log"
    console_path = run_dir / "console.log"
    command = [
        sys.executable,
        "-u",
        str(smoke_script),
        "--headless",
        "--device",
        f"cuda:{args.gpu_index}",
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--output",
        str(result_path),
        "--trace-output",
        str(trace_path),
        "--kit_args=--/log/flushStandardStreamOutput=true",
    ]
    if args.reset_lifecycle:
        command.append("--reset-lifecycle")
    if args.shared_bank:
        command.append("--shared-bank")
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    source_path = str(root / "source/rl_training")
    inherited_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        source_path
        if not inherited_pythonpath
        else source_path + os.pathsep + inherited_pythonpath
    )

    source_files = [
        root / relative_path
        for relative_path in _capture(["git", "ls-files", "source/"], cwd=root).splitlines()
    ]
    monitored_files = [
        *source_files,
        smoke_script,
        supervisor_path.with_name("source_parallel_reset_lifecycle.py"),
        supervisor_path.with_name("source_shared_terrain_diagnostic.py"),
        supervisor_path.with_name("source_shared_contact_report.py"),
        supervisor_path.with_name("source_shared_tensor_contact.py"),
        supervisor_path.with_name("source_shared_report_validation.py"),
        supervisor_path,
        asset_audit_path,
    ]
    source_hashes = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(set(monitored_files))
    }
    if args.shared_bank:
        source_hashes[str(dependency_audit_path.relative_to(root))] = _sha256(dependency_audit_path)
    gpu_samples_path = run_dir / "gpu_samples.jsonl"
    gpu_samples_path.touch(exist_ok=False)

    def sample_gpu(elapsed_s: float) -> None:
        sample = _capture(
            [
                "nvidia-smi",
                f"--id={args.gpu_index}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=root,
            timeout_s=_GPU_SAMPLE_TIMEOUT_S,
        )
        fields = [field.strip() for field in sample.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"unexpected nvidia-smi sample: {sample}")
        sample_record = {
            "elapsed_s": elapsed_s,
            "gpu_index": args.gpu_index,
            "memory_used_mib": int(fields[0]),
            "utilization_percent": int(fields[1]),
        }
        with gpu_samples_path.open("a", encoding="utf-8") as samples_file:
            samples_file.write(json.dumps(sample_record) + "\n")
            samples_file.flush()

    manifest = {
        "schema_version": "source_parallel_terrain_startup_supervisor_v3",
        "revision": _capture(["git", "rev-parse", "HEAD"], cwd=root),
        "model_revision": _capture(
            ["git", "-C", str(model_root), "rev-parse", "HEAD"], cwd=root
        ),
        "command": command,
        "logging": {
            "python_unbuffered": True,
            "carb_flush_standard_stream_output": True,
            "post_termination_console_scan": True,
            "fatal_exception_trace_and_console_monitoring": True,
        },
        "check": "shared_bank_g2" if args.shared_bank and args.reset_lifecycle else "shared_bank" if args.shared_bank else ("reset_lifecycle" if args.reset_lifecycle else "startup"),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": f"cuda:{args.gpu_index}",
        "gpu_snapshot": gpu_snapshot,
        "gpu_process_snapshot": gpu_process_snapshot,
        "startup_timeout_s": args.startup_timeout_s,
        "total_timeout_s": args.total_timeout_s,
        "policy_loaded": False,
        "ppo_updates": 0,
        "source_file_sha256": source_hashes,
        "dependency_file_sha256": dependency_hashes,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    execution = run_monitored_process(
        command,
        console_path=console_path,
        trace_path=trace_path,
        startup_timeout_s=args.startup_timeout_s,
        total_timeout_s=args.total_timeout_s,
        sample_interval_s=1.0,
        sample_callback=sample_gpu,
        cwd=root,
        env=child_env,
    )
    outcome = finalize_run(
        run_dir=run_dir, root=root, execution=execution,
        source_hashes=source_hashes, dependency_hashes=dependency_hashes,
        num_envs=args.num_envs, shared_bank=args.shared_bank, reset_lifecycle=args.reset_lifecycle,
    )
    print(json.dumps({"run_dir": str(run_dir), **outcome}, ensure_ascii=False), flush=True)
    return 0 if outcome["status"] == "diagnostic_smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
