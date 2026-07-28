import json
from argparse import Namespace

from scripts.reinforcement_learning.online_adaptation import compare_online_adaptation as compare


def test_run_one_passes_forward_speed_and_rejects_insufficient_heldout(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, check):
        commands.append(command)
        run_dir = tmp_path / "online" / "seed_7"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "reset_count": 0,
                    "fell": False,
                    "first_half_velocity_error": 1.0,
                    "second_half_velocity_error": 0.5,
                    "first_half_tilt_rms": 1.0,
                    "second_half_tilt_rms": 0.5,
                    "first_half_nonfoot_contacts": 0,
                    "second_half_nonfoot_contacts": 0,
                    "parameter_changed": True,
                    "safety_violation": False,
                    "heldout_samples": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(compare.subprocess, "run", fake_run)
    args = Namespace(
        base_checkpoint="base.pt",
        residual_checkpoint="residual.pt",
        task="HeldoutBoxes-Deeprobotics-Lite3-v0",
        max_steps=2_000,
        device="cuda:0",
        headless=True,
        video=False,
        forward_speed=0.3,
        control_timeout_ms=100.0,
        stochastic_control=False,
        min_heldout_samples=100,
        actor_updates_per_second=1.0,
        novelty_calibration_steps=256,
        novelty_enter_threshold=2.5,
        novelty_exit_threshold=1.0,
        novelty_enter_count=10,
        novelty_enter_window=20,
        novelty_recovery_steps=100,
        adaptation_min_replay_size=128,
        rollback_window=50,
        rollback_patience=2,
        rollback_degradation_ratio=0.10,
        stable_improvement_ratio=0.05,
        adaptation_cooldown_steps=100,
    )

    result = compare._run_one(args, tmp_path, "online", 7)

    assert "--forward-speed" in commands[0]
    assert commands[0][commands[0].index("--forward-speed") + 1] == "0.3"
    assert "--control-timeout-ms" in commands[0]
    assert commands[0][commands[0].index("--control-timeout-ms") + 1] == "100.0"
    expected_options = {
        "--actor-updates-per-second": "1.0",
        "--novelty-calibration-steps": "256",
        "--novelty-enter-threshold": "2.5",
        "--novelty-exit-threshold": "1.0",
        "--novelty-enter-count": "10",
        "--novelty-enter-window": "20",
        "--novelty-recovery-steps": "100",
        "--adaptation-min-replay-size": "128",
        "--rollback-window": "50",
        "--rollback-patience": "2",
        "--rollback-degradation-ratio": "0.1",
        "--stable-improvement-ratio": "0.05",
        "--adaptation-cooldown-steps": "100",
    }
    for option, value in expected_options.items():
        assert commands[0][commands[0].index(option) + 1] == value
    assert not result["passed"]
    assert not result["acceptance_passed"]
    assert result["heldout_coverage_passed"] is False
    assert result["checks"]["heldout_coverage"] is False
