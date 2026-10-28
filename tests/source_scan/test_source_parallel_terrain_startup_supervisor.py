"""The startup supervisor must stop immediately on a terrain cooking error."""

import importlib.util
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
def test_supervisor_kills_signal_resistant_descendants(tmp_path, leader_exits):
    ready_path = tmp_path / "child.pid"
    child_code = (
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(ready_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(20)\n"
    )
    leader_code = (
        "import pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"while not pathlib.Path({str(ready_path)!r}).exists(): time.sleep(0.01)\n"
        "print('cooking failure for /World/ground/terrain/ordinary/mesh', flush=True)\n"
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
        assert result["stop_reason"] == "physx_cooking_failure"
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
