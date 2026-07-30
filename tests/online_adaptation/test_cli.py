import pytest

from scripts.reinforcement_learning.online_adaptation.cli import (
    build_flat_baseline_manifest_parser,
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


def test_flat_baseline_manifest_parser_is_strict_and_repeatable():
    args = build_flat_baseline_manifest_parser().parse_args(
        [
            "--checkpoint",
            "policy.pt",
            "--config",
            "env.yaml",
            "--config",
            "agent.yaml",
            "--seed",
            "7",
            "--num-envs",
            "4096",
            "--max-iterations",
            "1500",
            "--training-command-file",
            "command.txt",
            "--output",
            "manifest.json",
        ]
    )

    assert args.task == "Flat-Deeprobotics-Lite3-v0"
    assert args.config == ["env.yaml", "agent.yaml"]
    assert args.seed == 7
    assert args.num_envs == 4096
    assert args.max_iterations == 1500


def test_play_defaults_configure_novelty_gated_actor_updates():
    args = build_play_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt",
         "--experiment-checkpoint-out", "experiment.pt"]
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


def test_play_parser_accepts_curriculum_worker_arguments():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt",
            "--terrain-stage", "step_2cm", "--terrain-seed", "101",
            "--protocol-sha256", "HASH", "--experiment-checkpoint-in", "in.pt",
            "--experiment-checkpoint-out", "out.pt", "--flat-replay", "flat.pt",
            "--flat-anchor-weight", "2.0", "--n-step", "3",
            "--residual-gain", "0.25",
        ]
    )
    assert args.terrain_stage == "step_2cm"
    assert args.terrain_seed == 101
    assert args.n_step == 3
    assert args.residual_gain == 0.25


@pytest.mark.parametrize(
    "arguments",
    [
        ["--terrain-stage", "unknown", "--terrain-seed", "101"],
        ["--terrain-stage", "step_2cm", "--terrain-seed", "1009"],
    ],
)
def test_play_parser_rejects_invalid_training_geometry(arguments):
    with pytest.raises(SystemExit):
        build_play_parser().parse_args(
            ["--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt",
             "--experiment-checkpoint-out", "out.pt", *arguments]
        )


def test_play_parser_rejects_online_mode_without_checkpoint_output():
    with pytest.raises(SystemExit):
        build_play_parser().parse_args(
            ["--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt"]
        )
