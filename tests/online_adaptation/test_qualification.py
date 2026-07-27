from rl_training.online_adaptation.qualification import build_qualification_report


def test_qualification_report_requires_zero_falls_and_error_gate():
    records = [
        {"seed": 7, "falls": 0, "steps": 1000, "mean_velocity_error": 0.30},
        {"seed": 11, "falls": 0, "steps": 1000, "mean_velocity_error": 0.34},
    ]

    report = build_qualification_report("source-task", "base.pt", records, 0.35)

    assert report["aggregate"] == {
        "falls": 0,
        "mean_velocity_error": 0.32,
        "max_mean_velocity_error": 0.35,
        "passed": True,
    }


def test_qualification_report_fails_if_any_seed_falls():
    records = [
        {"seed": 7, "falls": 1, "steps": 91, "mean_velocity_error": 0.20},
        {"seed": 11, "falls": 0, "steps": 1000, "mean_velocity_error": 0.20},
    ]

    report = build_qualification_report("source-task", "base.pt", records, 0.35)

    assert report["aggregate"]["falls"] == 1
    assert report["aggregate"]["passed"] is False
