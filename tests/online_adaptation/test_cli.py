from scripts.reinforcement_learning.online_adaptation.cli import (
    build_play_parser,
    build_qualify_parser,
    build_train_parser,
)


def test_train_defaults_keep_heldout_task_out_of_pretraining():
    args = build_train_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "OnlineAdapt-Deeprobotics-Lite3-v0"
    assert "HeldoutBoxes" not in args.task
    assert args.min_replay_size == 256
    assert args.replay_capacity == 10_000


def test_train_can_resume_residual_checkpoint():
    args = build_train_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--resume-checkpoint", "residual.pt"]
    )
    assert args.resume_checkpoint == "residual.pt"


def test_qualification_defaults_to_strict_flat_baseline():
    args = build_qualify_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "Flat-Deeprobotics-Lite3-v0"
    assert args.seeds == [7, 11, 19, 23, 31]
    assert args.steps_per_seed == 1_000
    assert args.max_mean_velocity_error == 0.35


def test_qualification_parser_describes_strict_flat_baseline_task():
    parser = build_qualify_parser()

    assert parser.description == "Qualify a frozen Lite3 base policy on the strict-flat baseline task."


def test_play_defaults_configure_novelty_gated_actor_updates():
    args = build_play_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt"]
    )

    assert args.updates_per_second == 5.0
    assert args.actor_updates_per_second == 1.0
    assert args.novelty_calibration_steps == 256
    assert args.novelty_enter_threshold == 2.5
    assert args.novelty_exit_threshold == 1.0
    assert args.novelty_enter_count == 10
    assert args.novelty_enter_window == 20
    assert args.novelty_recovery_steps == 100
    assert args.adaptation_min_replay_size == 128
    assert args.rollback_window == 50
    assert args.rollback_patience == 2
    assert args.rollback_degradation_ratio == 0.10
    assert args.stable_improvement_ratio == 0.05
    assert args.adaptation_cooldown_steps == 100
