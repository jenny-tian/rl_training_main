"""The observer must never mistake a model report for process completion."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
import reference_six_watch as watch


@pytest.fixture
def run(tmp_path, monkeypatch):
    def write(name, data):
        (tmp_path / name).write_text(json.dumps(data))
    write("execution.json", dict(status="running", source_commit="abc", physical_gpu=0, pid=1))
    write("runtime_overrides.json", dict(max_iterations=10000, num_envs=4096))
    write("report.json", dict(status="passed", completed_updates=10000, environment_steps=240000,
                              from_scratch=True, checkpoint_loaded=None, resets=5, terminated=3, truncated=2,
                              updates=[dict(update=10000, collect_s=5, learn_s=.25)]))
    monkeypatch.setattr(watch.subprocess, "check_output", lambda *args, **kwargs: "8000, 16000, 90")
    return tmp_path


def test_waits_for_supervisor_even_when_training_report_passes(run):
    assert watch.snapshot(run)["status"] == "running"


def test_supervisor_failure_overrides_passed_model_report(run):
    state = watch.read(run / "execution.json")
    state.update(status="failed", failure="post-save process failure")
    watch.save(run / "execution.json", state)
    assert watch.snapshot(run)["status"] == "failed"


def test_requires_exact_budget_for_completion(run):
    execution = watch.read(run / "execution.json")
    execution["status"] = "completed"
    watch.save(run / "execution.json", execution)
    assert watch.snapshot(run)["status"] == "completed"
    report = watch.read(run / "report.json")
    report["completed_updates"] = 9999
    watch.save(run / "report.json", report)
    assert watch.snapshot(run)["status"] == "incomplete"


def test_log_metrics_use_last_complete_update(tmp_path):
    path = tmp_path / "runtime.log"
    path.write_text("Mean reward: -2.0\nMetrics/base_velocity/error_vel_xy: 0.4\n"
                    "REFERENCE SIX train: 20/10000\nMean reward: -1.0\n")
    metrics = watch.log_metrics(path)
    assert metrics["completed_updates"] == 20
    assert metrics["mean_reward"] == -2.


def test_partially_written_training_report_yields_no_snapshot(run):
    (run / "report.json").write_text('{"status":')
    assert watch.snapshot(run) is None
