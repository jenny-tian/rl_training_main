"""The supervisor must stop on cooking errors or explicit fatal trace markers."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/tools/source_parallel_terrain_startup_supervisor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "source_parallel_terrain_supervisor_under_test", MODULE_PATH
)
supervisor_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor_module
SPEC.loader.exec_module(supervisor_module)


def test_cooking_failure_parser_matches_terrain_collider_and_ignores_other_meshes():
    assert supervisor_module.find_collision_cooking_failure(
        "[Warning] [omni.physx.cooking.plugin] UjitsoMeshCookingContext: "
        "cooking failure for /World/ground/terrain/ordinary/mesh"
    )
    assert supervisor_module.find_collision_cooking_failure(
        "[Warning] [omni.physx.plugin] Unable to create triangle mesh for:"
        "/World/ground/terrain/ordinary/mesh"
    )
    assert supervisor_module.find_collision_cooking_failure(
        "[Warning] cooking failure for /World/ground/terrain/ordinary/mesh"
    )
    assert (
        supervisor_module.find_collision_cooking_failure(
            "[Warning] [omni.physx.plugin] Unable to create triangle mesh for:"
            "/World/envs/env_0/Robot/torso"
        )
        is None
    )


def test_supervisor_terminates_process_group_when_cooking_fails(tmp_path):
    console_path = tmp_path / "console.log"
    trace_path = tmp_path / "trace.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        "import sys,time; "
        "sys.stderr.write('[Warning] [omni.physx.cooking.plugin] "
        "UjitsoMeshCookingContext: cooking failure for "
        "/World/ground/terrain/ordinary/mesh\\n'); "
        "sys.stderr.flush(); time.sleep(20)",
    ]

    result = supervisor_module.run_monitored_process(
        command,
        console_path=console_path,
        trace_path=trace_path,
        startup_timeout_s=5,
        total_timeout_s=10,
        poll_interval_s=0.02,
        terminate_grace_s=0.5,
    )

    assert result["stop_reason"] == "physx_cooking_failure"
    assert result["failure_marker"].endswith(
        "/World/ground/terrain/ordinary/mesh"
    )
    assert result["elapsed_s"] < 3
    assert result["collision_cooking_failure"]["detected_phase"] == "monitor"
    assert result["stop_requested_elapsed_s"] is not None
    assert "cooking failure" in console_path.read_text(encoding="utf-8")


def test_supervisor_keeps_successful_command_result(tmp_path):
    console_path = tmp_path / "console.log"
    trace_path = tmp_path / "trace.log"
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", "print('startup okay')"],
        console_path=console_path,
        trace_path=trace_path,
        startup_timeout_s=5,
        total_timeout_s=10,
        poll_interval_s=0.02,
    )
    assert result["exit_code"] == 0
    assert result["stop_reason"] is None
    assert result["collision_cooking_failure"] is None
    assert result["stop_requested_elapsed_s"] is None


def test_supervisor_classifies_unfinished_initial_reset_as_startup_timeout(tmp_path):
    console_path = tmp_path / "console.log"
    trace_path = tmp_path / "trace.log"
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        console_path=console_path,
        trace_path=trace_path,
        startup_timeout_s=0.15,
        total_timeout_s=1,
        poll_interval_s=0.02,
        terminate_grace_s=0.5,
    )
    assert result["stop_reason"] == "startup_timeout"


def test_supervisor_stops_if_resource_sampling_fails(tmp_path):
    def fail_sample(_elapsed_s):
        raise RuntimeError("GPU disappeared")

    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        console_path=tmp_path / "console.log",
        trace_path=tmp_path / "trace.log",
        startup_timeout_s=5,
        total_timeout_s=10,
        poll_interval_s=0.02,
        terminate_grace_s=0.5,
        sample_interval_s=0.02,
        sample_callback=fail_sample,
    )
    assert result["stop_reason"] == "resource_monitor_failure"
    assert "GPU disappeared" in result["failure_marker"]


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux process state")
@pytest.mark.parametrize("leader_exits", [False, True])
@pytest.mark.parametrize("failure_kind", ["cooking", "fatal"])
def test_supervisor_kills_signal_resistant_descendants(tmp_path, leader_exits, failure_kind):
    ready_path = tmp_path / "child.pid"
    child_code = (
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(ready_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(20)\n"
    )
    marker = ("cooking failure for /World/ground/terrain/ordinary/mesh" if failure_kind == "cooking"
              else "[parallel-terrain-smoke] fatal-exception-recorded")
    leader_code = (
        "import pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"while not pathlib.Path({str(ready_path)!r}).exists(): time.sleep(0.01)\n"
        f"print({marker!r}, flush=True)\n"
        + ("sys.exit(0)\n" if leader_exits else "time.sleep(20)\n")
    )

    def child_running():
        if not ready_path.exists():
            return False
        state_path = Path(f"/proc/{int(ready_path.read_text())}/stat")
        try:
            return state_path.read_text().split(") ", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    try:
        result = supervisor_module.run_monitored_process(
            [sys.executable, "-c", leader_code],
            console_path=tmp_path / "console.log",
            trace_path=tmp_path / "trace.log",
            startup_timeout_s=5,
            total_timeout_s=10,
            poll_interval_s=0.02,
            terminate_grace_s=0.1,
        )
        assert result["stop_reason"] == ("physx_cooking_failure" if failure_kind == "cooking" else "fatal_exception")
        deadline = time.monotonic() + 2
        while child_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_path.is_file()
        assert not child_running()
    finally:
        if child_running():
            try:
                os.kill(int(ready_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_supervisor_stops_when_resource_query_hangs(tmp_path):
    def hanging_sample(_elapsed_s):
        supervisor_module._capture(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            cwd=tmp_path,
            timeout_s=0.05,
        )

    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        console_path=tmp_path / "console.log",
        trace_path=tmp_path / "trace.log",
        startup_timeout_s=5,
        total_timeout_s=10,
        sample_interval_s=0.02,
        sample_callback=hanging_sample,
        terminate_grace_s=0.1,
    )
    assert result["stop_reason"] == "resource_monitor_failure"
    assert "timed out" in result["failure_marker"]
    assert result["elapsed_s"] < 3


def test_supervisor_cleans_up_after_trace_read_error(tmp_path, monkeypatch):
    processes = []
    real_popen = subprocess.Popen

    def record_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def unreadable_trace(_path):
        raise OSError("trace became unreadable")

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", record_process)
    monkeypatch.setattr(supervisor_module, "_trace_has_completed_initial_reset", unreadable_trace)
    with pytest.raises(OSError, match="trace became unreadable"):
        supervisor_module.run_monitored_process(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            console_path=tmp_path / "console.log",
            trace_path=tmp_path / "trace.log",
            startup_timeout_s=0.05,
            total_timeout_s=1,
            poll_interval_s=0.01,
            terminate_grace_s=0.1,
        )
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_completed_initial_reset_still_obeys_total_timeout(tmp_path):
    trace_path = tmp_path / "trace.log"
    trace_path.write_text("initial-reset-complete\n", encoding="utf-8")
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        console_path=tmp_path / "console.log",
        trace_path=trace_path,
        startup_timeout_s=0.05,
        total_timeout_s=0.15,
        poll_interval_s=0.01,
        terminate_grace_s=0.1,
    )
    assert result["stop_reason"] == "total_timeout"


@pytest.mark.skipif(os.name != "posix", reason="requires SIGTERM handlers")
@pytest.mark.parametrize("trigger", ["startup_timeout", "total_timeout", "resource_monitor_failure"])
@pytest.mark.parametrize("failure_kind", ["cooking", "fatal"])
def test_cleanup_captures_buffered_failure_without_rewriting_stop_trigger(tmp_path, trigger, failure_kind):
    ready_path = tmp_path / "ready"
    trace_path = tmp_path / "trace.log"
    if trigger == "total_timeout":
        trace_path.write_text("initial-reset-complete\n")
    marker = ("cooking failure for /World/ground/terrain/ordinary/part_001/mesh" if failure_kind == "cooking"
              else "[parallel-terrain-smoke] simulation-app-close-exception")
    code = (
        "import pathlib, signal, sys, time\n"
        "def stop(*_):\n"
        # The warning remains buffered until graceful termination flushes it.
        "    sys.stdout.flush()\n"
        "    print('tail noise' * 2000, flush=True)\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"sys.stdout.write({marker!r} + '\\n')\n"
        f"pathlib.Path({str(ready_path)!r}).touch()\n"
        "time.sleep(20)\n"
    )

    def sample(_elapsed):
        if trigger == "resource_monitor_failure" and ready_path.exists():
            raise RuntimeError("GPU sample failed after child became ready")

    child_env = os.environ.copy()
    child_env.pop("PYTHONUNBUFFERED", None)
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", code],
        console_path=tmp_path / "console.log",
        trace_path=trace_path,
        startup_timeout_s=0.5,
        total_timeout_s=0.5 if trigger == "total_timeout" else 3,
        poll_interval_s=0.01,
        terminate_grace_s=0.5,
        sample_interval_s=0.01,
        sample_callback=sample,
        env=child_env,
    )
    assert ready_path.exists()
    assert result["exit_code"] == 0
    assert result["stop_reason"] == trigger
    if trigger == "resource_monitor_failure":
        assert "GPU sample failed" in result["failure_marker"]
    else:
        assert result["failure_marker"] is None
    diagnostic = result["collision_cooking_failure" if failure_kind == "cooking" else "fatal_exception"]
    assert diagnostic["marker"] == marker
    assert diagnostic["detected_phase"] == "post_termination"
    assert diagnostic["detected_elapsed_s"] >= result["stop_requested_elapsed_s"]


def test_failure_survives_a_large_final_output_burst(tmp_path):
    marker = "Unable to create triangle mesh for:/World/ground/terrain/ordinary/part_000/mesh"
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", f"print({marker!r}); print('noise' * 10000)"],
        console_path=tmp_path / "console.log",
        trace_path=tmp_path / "trace.log",
        startup_timeout_s=5,
        total_timeout_s=10,
        poll_interval_s=0.02,
        terminate_grace_s=0.1,
    )
    assert result["collision_cooking_failure"]["marker"] == marker
    assert result["stop_reason"] == "physx_cooking_failure"


@pytest.mark.parametrize("late_collision,late_fatal", [(False, False), (True, False), (False, True)])
def test_main_rejects_late_failure_even_with_successful_smoke_report(
    tmp_path, monkeypatch, late_collision, late_fatal
):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [str(MODULE_PATH), "--run-dir", str(run_dir), "--num-envs", "60"])
    # Exercise report/launch assembly without accessing a GPU or starting Kit.
    monkeypatch.setattr(supervisor_module, "_capture", lambda *a, **k: "")
    monkeypatch.setattr(supervisor_module, "_gpu_preflight", lambda *a, **k: ("fixture", ""))
    audit_path = MODULE_PATH.parents[2] / "docs/lite3-usd-collision-geometry-audit-20261027.json"
    audited_hashes = {
        str((MODULE_PATH.parents[2] / item["path"]).resolve()): item["sha256"]
        for item in json.loads(audit_path.read_text())["assets"].values()
    }
    monkeypatch.setattr(
        supervisor_module, "_sha256", lambda path: audited_hashes.get(str(path.resolve()), "fixture")
    )
    diagnostic = (
        {"marker": "cooking failure for /World/ground/terrain/ordinary/mesh",
         "detected_phase": "post_termination", "detected_elapsed_s": 0.1}
        if late_collision else None
    )

    def completed_smoke(command, **kwargs):
        assert command[1] == "-u"
        assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        assert "--kit_args=--/log/flushStandardStreamOutput=true" in command
        (run_dir / "result.json").write_text(json.dumps({
            "status": "diagnostic_smoke_passed", "num_envs": 60,
            "zero_action_steps": 1, "policy_loaded": False, "ppo_updates": 0,
        }))
        return {"exit_code": 0, "stop_reason": None, "failure_marker": None,
                "collision_cooking_failure": diagnostic,
                "fatal_exception": {"detected_phase": "post_termination", "marker": "fatal"} if late_fatal else None}

    monkeypatch.setattr(supervisor_module, "run_monitored_process", completed_smoke)
    failed = late_collision or late_fatal
    assert supervisor_module.main() == (1 if failed else 0)
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["status"] == ("startup_check_failed" if failed else "diagnostic_smoke_passed")
    assert outcome["stop_reason"] is None
    assert outcome["collision_cooking_failure"] == diagnostic
    assert (outcome["fatal_exception"] is not None) == late_fatal


def test_fatal_parser_matches_only_explicit_diagnostic_lines():
    for marker in supervisor_module._FATAL_TRACE_LINES:
        assert supervisor_module.find_fatal_exception("noise\n" + marker + "\n") == marker
    for noise in ("fatal-exception-recorded", "[Warning] fatal-exception-recorded",
                  "[parallel-terrain-smoke] simulation-app-close-start",
                  "[parallel-terrain-smoke] report-write-complete",
                  "mentioned: [parallel-terrain-smoke] fatal-exception-recorded",
                  "[parallel-terrain-smoke] fatal-exception-recorded-example"):
        assert supervisor_module.find_fatal_exception(noise) is None


@pytest.mark.skipif(os.name != "posix", reason="requires SIGTERM")
@pytest.mark.parametrize("stream", ["console", "trace"])
@pytest.mark.parametrize("ignore_sigterm", [False, True])
def test_fatal_marker_ends_stuck_close_within_cleanup_grace(tmp_path, stream, ignore_sigterm):
    trace_path = tmp_path / "trace.log"
    marker = "[parallel-terrain-smoke] fatal-exception-recorded"
    code = "import pathlib, signal, sys, time\n"
    if ignore_sigterm:
        code += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    if stream == "trace":
        code += f"pathlib.Path({str(trace_path)!r}).write_text({marker!r} + '\\n')\n"
    else:
        code += f"print({marker!r}, flush=True)\n"
    code += "time.sleep(20)\n"
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", code], console_path=tmp_path / "console.log", trace_path=trace_path,
        startup_timeout_s=5, total_timeout_s=10, poll_interval_s=0.01, terminate_grace_s=0.1,
    )
    assert result["stop_reason"] == "fatal_exception"
    assert result["failure_marker"] == marker
    assert result["fatal_exception"]["stream"] == stream
    assert result["fatal_exception"]["detected_phase"] == "monitor"
    assert result["elapsed_s"] < 3  # Neither the 5s startup limit nor the 20s child sleep.
    assert result["elapsed_s"] - result["stop_requested_elapsed_s"] < 1
    assert result["terminate_grace_s"] == 0.1
    assert result["exit_code"] == (-signal.SIGKILL if ignore_sigterm else -signal.SIGTERM)


@pytest.mark.parametrize("stream", ["console", "trace"])
def test_fragmented_fatal_marker_is_detected(tmp_path, stream):
    trace_path = tmp_path / "trace.log"
    marker = "[parallel-terrain-smoke] fatal-exception-recorded"
    code = "import pathlib, sys, time\n"
    if stream == "console":
        code += f"sys.stdout.write({marker[:20]!r}); sys.stdout.flush(); time.sleep(0.05)\n"
        code += f"sys.stdout.write({marker[20:]!r} + '\\n'); sys.stdout.flush()\n"
    else:
        code += f"p=pathlib.Path({str(trace_path)!r}); p.write_text({marker[:20]!r}); time.sleep(0.05)\n"
        code += f"p.write_text({marker!r} + '\\n')\n"
    code += "time.sleep(20)\n"
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", code], console_path=tmp_path / "console.log", trace_path=trace_path,
        startup_timeout_s=5, total_timeout_s=10, poll_interval_s=0.01, terminate_grace_s=0.1,
    )
    assert result["stop_reason"] == "fatal_exception"
    assert result["fatal_exception"]["marker"] == marker


def test_fatal_marker_cannot_be_hidden_by_zero_exit_or_large_final_output(tmp_path):
    marker = "[parallel-terrain-smoke] simulation-app-close-exception"
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", f"print({marker!r}); print('noise' * 10000)"],
        console_path=tmp_path / "console.log", trace_path=tmp_path / "trace.log",
        startup_timeout_s=5, total_timeout_s=10, poll_interval_s=0.02, terminate_grace_s=0.1,
    )
    assert result["stop_reason"] == "fatal_exception"
    assert result["fatal_exception"]["marker"] == marker


def test_archived_shared_bank_failure_now_triggers_automatic_stop(tmp_path):
    root = MODULE_PATH.parents[2]
    archived = root / "docs/lite3-source-shared-bank-2env-20261028"
    trace_path = tmp_path / "trace.log"
    code = (
        "import pathlib, sys, time\n"
        f"pathlib.Path({str(trace_path)!r}).write_bytes(pathlib.Path({str(archived / 'trace.log')!r}).read_bytes())\n"
        f"sys.stdout.write(pathlib.Path({str(archived / 'console.log')!r}).read_text()); sys.stdout.flush()\n"
        "time.sleep(20)\n"
    )
    result = supervisor_module.run_monitored_process(
        [sys.executable, "-c", code], console_path=tmp_path / "console.log", trace_path=trace_path,
        startup_timeout_s=5, total_timeout_s=10, poll_interval_s=0.01, terminate_grace_s=0.1,
    )
    assert result["stop_reason"] == "fatal_exception"
    assert result["fatal_exception"]["marker"] == "[parallel-terrain-smoke] fatal-exception-recorded"
    assert result["elapsed_s"] < 3


def test_lifecycle_report_requires_complete_direct_and_automatic_reset_coverage():
    report = dict(status='reset_lifecycle_passed', assignment_count=363, complete_blocks=3,
                  direct_reset_count=300, automatic_reset_ids=[2, 23, 58], reset_batch_count=10)
    assert supervisor_module.lifecycle_report_passed(report)
    assert not supervisor_module.lifecycle_report_passed(None)
    for key in report:
        incomplete = dict(report)
        del incomplete[key]
        assert not supervisor_module.lifecycle_report_passed(incomplete)
    assert not supervisor_module.lifecycle_report_passed(dict(report, automatic_reset_ids=[2, 23]))
