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


@pytest.mark.parametrize('marker', [
    '[Error] PhysX contact buffer overflow',
    '[Warning] contact data count exceeded max_contact_data_count',
    '[Error] increase gpuMaxRigidPatchCount',
    '[Error] PxgDynamicsMemoryConfig::foundLostPairsCapacity insufficient',
    'Dropping contacts in solver because we exceeded limit of 32 friction patches.',
    'Dropping contacts in contact reduction due to full contact buffer.',
    'Reached limit set by PxSceneDesc::maxNbContactDataBlocks - ran out of buffer space for narrow phase. '
    'Either accept dropped contacts or increase buffer size allocated for narrow phase.',
])
def test_contact_capacity_failure_cannot_be_reported_as_success(marker):
    assert supervisor_module.find_fatal_exception(marker) == marker
    assert supervisor_module.find_fatal_exception('contact buffer capacity configured to 16384') is None


def test_supervisor_stops_on_contact_capacity_loss(tmp_path):
    marker = '[Error] contact data buffer overflow'
    result = supervisor_module.run_monitored_process(
        [sys.executable, '-u', '-c', f'import time; print({marker!r}, flush=True); time.sleep(10)'],
        console_path=tmp_path / 'console.log', trace_path=tmp_path / 'trace.log',
        startup_timeout_s=5, total_timeout_s=10, poll_interval_s=.02, terminate_grace_s=.5,
    )
    assert result['stop_reason'] == 'fatal_exception'
    assert result['elapsed_s'] < 3


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
        except (FileNotFoundError, ProcessLookupError):
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


def test_eight_environment_shared_bank_requires_complete_lifecycle_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", [str(MODULE_PATH), "--shared-bank", "--num-envs", "8", "--seed", "130196",
                       "--run-dir", str(tmp_path / "run")]
    )
    with pytest.raises(SystemExit) as error:
        supervisor_module.main()
    assert error.value.code == 2


def test_eight_environment_smoke_rejects_missing_lifecycle_before_output(tmp_path):
    smoke_path = MODULE_PATH.with_name("source_parallel_terrain_startup_smoke.py")
    output = tmp_path / "must-not-exist.json"
    code = r'''
import runpy, sys, types
app = types.ModuleType('isaaclab.app')
class AppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        return None
app.AppLauncher = AppLauncher
sys.modules['isaaclab.app'] = app
tools, output = sys.argv[1:]
sys.path.insert(0, tools)
sys.argv = ['smoke', '--shared-bank', '--num-envs', '8', '--seed', '130196', '--output', output]
runpy.run_path(tools + '/source_parallel_terrain_startup_smoke.py', run_name='__main__')
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(smoke_path.parent), str(output)],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 2
    assert "eight-environment shared-bank checks require --reset-lifecycle" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("num_envs", "shared_bank", "expected"),
    [(2, True, 8192), (8, True, 12288), (8, False, 8192)],
)
def test_gpu_preflight_budget_is_stricter_for_eight_environment_shared_bank(
    num_envs, shared_bank, expected
):
    assert supervisor_module.minimum_gpu_available_mib(
        num_envs=num_envs, shared_bank=shared_bank
    ) == expected


@pytest.mark.parametrize(('available_mib', 'accepted'), [(12288, True), (12287, False)])
def test_gpu_preflight_enforces_the_configured_eight_environment_floor(
    monkeypatch, available_mib, accepted
):
    total_mib = 24576
    used_mib = total_mib - available_mib
    snapshots = iter((
        f"0, GPU-fixture, RTX fixture, {used_mib} MiB, {total_mib} MiB, 0 %",
        "",
    ))
    monkeypatch.setattr(supervisor_module, "_capture", lambda *args, **kwargs: next(snapshots))
    if accepted:
        supervisor_module._gpu_preflight(0, root=Path('/tmp'), minimum_available_mib=12288)
    else:
        with pytest.raises(RuntimeError, match='at least 12288 MiB is required'):
            supervisor_module._gpu_preflight(0, root=Path('/tmp'), minimum_available_mib=12288)


def test_fatal_parser_matches_only_explicit_diagnostic_lines():
    for marker in supervisor_module._FATAL_TRACE_LINES:
        assert supervisor_module.find_fatal_exception("noise\n" + marker + "\n") == marker
    for noise in ("fatal-exception-recorded", "[Warning] fatal-exception-recorded",
                  "[parallel-terrain-smoke] simulation-app-close-start",
                  "[parallel-terrain-smoke] report-write-complete",
                  "mentioned: [parallel-terrain-smoke] fatal-exception-recorded",
                  "[parallel-terrain-smoke] fatal-exception-recorded-example"):
        assert supervisor_module.find_fatal_exception(noise) is None


@pytest.mark.parametrize("allow_sharing,available_mib,accepted", [
    (False, 16384, False), (True, 16384, True),
    (True, 12288, True), (True, 12287, False),
])
def test_gpu_memory_admission_can_allow_occupants_without_lowering_floor(
    monkeypatch, allow_sharing, available_mib, accepted
):
    gpu = f"0, GPU-fixture, RTX fixture, {24576-available_mib} MiB, 24576 MiB, 70 %"
    occupants = "GPU-fixture, 1234, /other/user/python, 4096 MiB"
    snapshots = iter((gpu, occupants))
    monkeypatch.setattr(supervisor_module, "_capture", lambda *a, **k: next(snapshots))
    kwargs = dict(root=Path('/tmp'), minimum_available_mib=12288)
    if allow_sharing:
        kwargs["allow_existing_compute_processes"] = True
    if accepted:
        assert supervisor_module._gpu_preflight(0, **kwargs) == (gpu, occupants)
    else:
        message = "at least 12288 MiB" if available_mib < 12288 else "already has a compute process"
        with pytest.raises(RuntimeError, match=message):
            supervisor_module._gpu_preflight(0, **kwargs)


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


ARCHIVED_PROBE = MODULE_PATH.parents[2] / 'docs/lite3-source-shared-bank-2env-tensor-repair-20261028'


@pytest.mark.parametrize('invalid_report', [False, True])
@pytest.mark.parametrize(('num_envs', 'shared_g2'), [(2, False), (2, True), (8, True)])
def test_shared_main_in_fresh_interpreter_without_kit(tmp_path, invalid_report, num_envs, shared_g2):
    """Real imports + main finalization; only launch/preflight are replaced by archived evidence."""
    code = r'''
import importlib.abc
import json
from pathlib import Path
import sys
from unittest.mock import patch

class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rl_training', 'isaaclab', 'isaacsim', 'omni', 'pxr', 'torch'}:
            raise AssertionError('unexpected simulator/task import: ' + fullname)
sys.meta_path.insert(0, NoSimulator())
tools, archive, output, invalid, env_count, g2 = sys.argv[1:]
env_count = int(env_count)
sys.path.insert(0, tools)
import source_parallel_terrain_startup_supervisor as supervisor
archive, output = Path(archive), Path(output)
def completed_child(command, **kwargs):
    report = json.loads((archive / 'result.json').read_text())
    assert ('--reset-lifecycle' in command) == (g2 == 'True')
    assert command[command.index('--num-envs') + 1] == str(env_count)
    if env_count == 8:
        report['num_envs'] = 8
        report['ground_contact_points'] = {str(i): 1 for i in range(8)}
        report['vertical_occlusion_environment_ids'] = list(range(1, 8))
        capture = report['contact_capture']
        topology = capture['tensor_topology']
        sensors = [f'/World/envs/env_{i}/Robot/FOOT' for i in range(8)]
        filters = sensors + topology['ground_colliders']
        topology.update(sensor_paths=sensors, filter_paths=[filters] * 8, sensor_count=8,
                        filter_count=len(filters), num_envs=8, capacity=65536)
        capture['ground_contact_points'] = {str(i): 1 for i in range(8)}
    if g2 == 'True':
        report.update(status='shared_bank_g2_passed', total_zero_action_steps=22,
                      reset_checks=dict(status='shared_reset_lifecycle_passed', num_envs=env_count, bank_rows=1,
                                        assignment_count=129 if env_count == 8 else 123, complete_blocks=1,
                                        direct_reset_count=120, reset_batch_count=33 if env_count == 8 else 91,
                                        tail_ticket_count=9 if env_count == 8 else 3,
                                        automatic_reset_ids=[7] if env_count == 8 else [0],
                                        diagnostic_timeout_injected=True, post_automatic_reset_step_checked=True,
                                        physics_control_steps=2,
                                        batch_coverage={'singleton': True, 'noncontiguous': True,
                                            'unordered_input': True, 'large_batch': True,
                                            'highest_env_id': True} if env_count == 8 else None))
    if invalid == 'True':
        report['ground_contact_points']['0'] = 0
    (output / 'result.json').write_text(json.dumps(report))
    (output / 'gpu_samples.jsonl').write_bytes((archive / 'gpu_samples.jsonl').read_bytes())
    return json.loads((archive / 'execution.json').read_text())
sys.argv = [supervisor.__file__, '--shared-bank', '--num-envs', str(env_count), '--seed', '130196',
            '--run-dir', str(output)]
if g2 == 'True':
    sys.argv.append('--reset-lifecycle')
with patch.object(supervisor, '_capture', return_value=''), \
     patch.object(supervisor, '_gpu_preflight', return_value=('offline replay', '')), \
     patch.object(supervisor, 'run_monitored_process', side_effect=completed_child):
    exit_code = supervisor.main()
manifest = json.loads((output / 'manifest.json').read_text())
assert 'scripts/tools/source_shared_report_validation.py' in manifest['source_file_sha256']
assert manifest['check'] == ('shared_bank_g2' if g2 == 'True' else 'shared_bank')
assert manifest['minimum_gpu_available_mib'] == (12288 if env_count == 8 else 8192)
assert exit_code == (1 if invalid == 'True' else 0)
assert not any(k.split('.')[0] in {'rl_training', 'pxr', 'omni', 'torch'} for k in sys.modules)
'''
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    completed = subprocess.run(
        [sys.executable, '-c', code, str(MODULE_PATH.parent), str(ARCHIVED_PROBE),
         str(tmp_path / 'replay'), str(invalid_report), str(num_envs), str(shared_g2)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    outcome = json.loads((tmp_path / 'replay/outcome.json').read_text())
    assert outcome['status'] == ('startup_check_failed' if invalid_report else 'diagnostic_smoke_passed')
    assert outcome['finalization_error'] is None


@pytest.mark.parametrize('fault', [
    'missing_result', 'bad_json', 'wrong_type', 'bad_topology', 'missing_source',
    'changed_source', 'bad_samples', 'import_error', 'validator_error', 'child_failure',
    'late_contact_loss',
])
def test_finalization_failures_always_leave_failed_outcome(tmp_path, monkeypatch, fault):
    monkeypatch.syspath_prepend(str(MODULE_PATH.parent))
    import source_shared_report_validation as validator
    import builtins
    report = json.loads((ARCHIVED_PROBE / 'result.json').read_text())
    execution = json.loads((ARCHIVED_PROBE / 'execution.json').read_text())
    (tmp_path / 'gpu_samples.jsonl').write_bytes((ARCHIVED_PROBE / 'gpu_samples.jsonl').read_bytes())
    if fault == 'wrong_type':
        report = []
    if fault == 'bad_topology':
        report['contact_capture']['tensor_topology'] = []
    if fault != 'missing_result':
        (tmp_path / 'result.json').write_text('{' if fault == 'bad_json' else json.dumps(report))
    if fault == 'bad_samples':
        (tmp_path / 'gpu_samples.jsonl').write_text('{')
    source_hashes = {}
    if fault in ('missing_source', 'changed_source'):
        source_hashes['source.py'] = 'expected'
        if fault == 'changed_source':
            (tmp_path / 'source.py').write_text('changed')
    if fault == 'import_error':
        original_import = builtins.__import__
        def fail_import(name, *args, **kwargs):
            if name == 'source_shared_report_validation':
                raise ImportError('injected validator import failure')
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', fail_import)
    if fault == 'validator_error':
        def fail_validation(_report):
            raise RuntimeError('injected validation failure')
        monkeypatch.setattr(validator, 'shared_report_passed', fail_validation)
    if fault == 'child_failure':
        execution['exit_code'] = 1
        execution['stop_reason'] = 'fatal_exception'
    if fault == 'late_contact_loss':
        execution['fatal_exception'] = {'marker': 'Dropping contacts', 'detected_phase': 'post_termination'}
    outcome = supervisor_module.finalize_run(
        run_dir=tmp_path, root=tmp_path, execution=execution,
        source_hashes=source_hashes, dependency_hashes={}, num_envs=2, shared_bank=True,
    )
    assert outcome == json.loads((tmp_path / 'outcome.json').read_text())
    assert outcome['status'] == 'startup_check_failed'
    assert outcome['stop_reason'] == execution['stop_reason']
    assert outcome['fatal_exception'] == execution['fatal_exception']
    persisted = json.loads((tmp_path / 'execution.json').read_text())
    assert persisted['exit_code'] == execution['exit_code']
    if fault not in ('wrong_type', 'bad_topology', 'changed_source', 'child_failure', 'late_contact_loss'):
        assert outcome['finalization_error'] is not None
        assert persisted['finalization_error'] == outcome['finalization_error']
