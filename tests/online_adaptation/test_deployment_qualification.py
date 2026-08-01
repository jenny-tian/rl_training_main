import pytest
from rl_training.online_adaptation.deployment_qualification import build_deployment_qualification_report
def _valid_record():
    return {"task": "FlatOnline-Deeprobotics-Lite3-v0", "seed": 42, "attempted_rows": 100000, "completed_episodes": 100, "timeouts": 100, "falls": 0, "premature_terminations": 0, "paused_rows": 0, "projected_rows": 0, "nonfinite_rows": 0, "max_projected_gravity_xy": 0.64, "min_base_height": 0.28, "mean_velocity_error": 0.1}
def _report(record):
    return build_deployment_qualification_report(task="FlatOnline-Deeprobotics-Lite3-v0", base_checkpoint="base.pt", record=record, expected_seed=42, expected_steps=100000, expected_episodes=100, max_mean_velocity_error=0.35, max_projected_gravity_xy=0.65)
def test_deployment_report_accepts_exact_100_episode_evidence():
    assert _report(_valid_record())["passed"] is True
@pytest.mark.parametrize(("field", "value"), [("attempted_rows", 99999), ("completed_episodes", 99), ("timeouts", 99), ("falls", 1), ("premature_terminations", 1), ("paused_rows", 1), ("projected_rows", 1), ("nonfinite_rows", 1)])
def test_deployment_report_rejects_incomplete_or_invalid_evidence(field, value):
    record = _valid_record(); record[field] = value
    assert _report(record)["passed"] is False
def test_deployment_report_rejects_tilt_over_unchanged_threshold():
    record = _valid_record(); record["max_projected_gravity_xy"] = 0.650001
    assert _report(record)["passed"] is False
