import pytest

from scripts.reinforcement_learning.online_adaptation.cli import (
    build_flat_baseline_manifest_parser,
    build_play_parser,
    build_qualify_parser,
    build_deployment_qualify_parser,
    build_flat_online_curriculum_parser,
    build_train_parser,
)


def test_train_defaults_keep_heldout_task_out_of_pretraining():
    args = build_train_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "OnlineAdapt-Deeprobotics-Lite3-v0"
    assert "HeldoutBoxes" not in args.task
    assert args.min_replay_size == 256
    assert args.replay_capacity == 10_000
    assert args.terrain_feature_source == "geometry"
    assert args.terrain_min_valid_fraction == 0.90
    assert args.latent_code_dim == 0
    assert args.latent_residual_fraction == 0.25


def test_play_parser_requires_height_scan_for_latent_actor():
    with pytest.raises(SystemExit, match="latent-code-dim requires"):
        build_play_parser().parse_args([
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--latent-code-dim", "4",
        ])
    with pytest.raises(SystemExit, match="latent-code-dim requires --risk-gate"):
        build_play_parser().parse_args([
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--terrain-feature-source", "height_scan",
            "--latent-code-dim", "4",
        ])
    args = build_play_parser().parse_args([
        "--base-checkpoint", "base.pt",
        "--experiment-checkpoint-out", "experiment.pt",
        "--terrain-feature-source", "height_scan",
        "--risk-gate",
        "--latent-code-dim", "4",
    ])
    assert args.latent_code_dim == 4

    with pytest.raises(SystemExit, match="cannot be used with --mode base"):
        build_play_parser().parse_args([
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--mode", "base",
            "--terrain-feature-source", "height_scan",
            "--risk-gate",
            "--latent-code-dim", "4",
        ])

    with pytest.raises(SystemExit, match="latent-residual-fraction must be positive"):
        build_play_parser().parse_args([
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--terrain-feature-source", "height_scan",
            "--risk-gate",
            "--latent-code-dim", "4",
            "--latent-residual-fraction", "0",
        ])

    with pytest.raises(SystemExit, match="latent-decoder requires"):
        build_play_parser().parse_args([
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--terrain-feature-source", "height_scan",
            "--latent-decoder", "decoder.pt",
        ])


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


def test_deployment_qualification_defaults_match_formal_distribution():
    args = build_deployment_qualify_parser().parse_args(["--base-checkpoint", "base.pt"])

    assert args.task == "FlatOnline-Deeprobotics-Lite3-v0"
    assert args.seed == 42
    assert args.expected_episodes == 100
    assert args.expected_steps == 100_000
    assert args.max_projected_gravity_xy == 0.65
    assert args.max_mean_velocity_error == 0.35


def test_deployment_qualification_accepts_training_style_underscore_options():
    args = build_deployment_qualify_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--expected_episodes", "3",
            "--expected_steps", "3000",
            "--max_projected_gravity_xy", "0.65",
        ]
    )

    assert args.expected_episodes == 3
    assert args.expected_steps == 3000
    assert args.max_projected_gravity_xy == 0.65


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
    assert args.actor_updates_per_second == 2.0
    assert args.update_schedule == "wall-clock"
    assert args.critic_updates_per_transition == 0.5
    assert args.actor_updates_per_transition == 0.005
    assert args.novelty_calibration_steps == 256
    assert args.novelty_enter_threshold == 2.5
    assert args.novelty_exit_threshold == 1.0
    assert args.novelty_enter_count == 10
    assert args.novelty_enter_window == 20
    assert args.novelty_recovery_steps == 100
    assert args.adaptation_min_replay_size == 128
    assert args.stalled_actor_replay_stride == 16
    assert args.rollback_window == 50
    assert args.rollback_patience == 4
    assert args.rollback_degradation_ratio == 0.10
    assert args.stable_improvement_ratio == 0.05
    assert args.adaptation_cooldown_steps == 100
    assert args.flat_fraction_ready == 0.25
    assert args.flat_anchor_weight == 0.5
    assert args.max_policy_residual_mse is None
    assert args.adapting_residual_gain == 0.5
    assert args.residual_rate_limit == 0.03
    assert args.adapting_stochastic_control is True
    assert args.max_critic_loss == 800.0
    assert args.max_mean_td_error == 10.0
    assert args.max_critic_grad_norm == 3_000.0
    assert args.critic_huber_delta is None
    assert args.post_step_residual_gain == 0.25
    assert args.post_step_recovery_steps == 25
    assert args.post_step_stability_weight == 3.0
    assert args.post_step_progress_weight == 2.0
    assert args.stall_penalty_weight == 2.0
    assert args.completion_bonus == 5.0
    assert args.pre_step_clearance_weight == 4.0
    assert args.pre_step_clearance_target == 0.045
    assert args.pre_step_approach_distance == 0.50
    assert args.require_track_completion is False
    assert args.terrain_feature_source == "geometry"
    assert args.terrain_min_valid_fraction == 0.90


def test_play_parser_accepts_control_step_update_schedule():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--update-schedule", "control-step",
        ]
    )

    assert args.update_schedule == "control-step"


def test_play_parser_accepts_height_scan_configuration():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--terrain-feature-source", "height_scan",
            "--terrain-min-valid-fraction", "0.8",
        ]
    )
    assert args.terrain_feature_source == "height_scan"
    assert args.terrain_min_valid_fraction == 0.8


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "inf"])
def test_play_parser_rejects_invalid_height_scan_fraction(value):
    with pytest.raises(SystemExit):
        build_play_parser().parse_args(
            [
                "--base-checkpoint", "base.pt",
                "--experiment-checkpoint-out", "experiment.pt",
                "--terrain-min-valid-fraction", value,
            ]
        )


def test_curriculum_parser_keeps_height_scan_overrides_optional():
    args = build_flat_online_curriculum_parser().parse_args(
        [
            "--base-checkpoint", "base.pt", "--base-manifest", "base.json",
            "--output-dir", "run", "--flat-replay", "flat.pt",
            "--terrain-feature-source", "height_scan",
            "--terrain-min-valid-fraction", "0.8",
        ]
    )
    assert args.terrain_feature_source == "height_scan"
    assert args.terrain_min_valid_fraction == 0.8


def test_play_parser_accepts_an_explicit_policy_residual_mse_limit():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--residual-checkpoint", "residual.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--max-policy-residual-mse", "0.125",
        ]
    )

    assert args.max_policy_residual_mse == 0.125


def test_play_parser_rejects_negative_policy_residual_mse_limit():
    with pytest.raises(SystemExit):
        build_play_parser().parse_args(
            [
                "--base-checkpoint", "base.pt",
                "--residual-checkpoint", "residual.pt",
                "--experiment-checkpoint-out", "experiment.pt",
                "--max-policy-residual-mse", "-0.1",
            ]
        )


def test_curriculum_parser_accepts_optional_policy_residual_mse_limit():
    args = build_flat_online_curriculum_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--base-manifest", "base.json",
            "--output-dir", "run",
            "--flat-replay", "flat.pt",
            "--max-policy-residual-mse", "0.125",
        ]
    )

    assert args.max_policy_residual_mse == 0.125


def test_play_parser_can_disable_adapting_exploration():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--no-adapting-stochastic-control",
        ]
    )

    assert args.adapting_stochastic_control is False


def test_play_parser_exposes_explicit_strict_completion_flag():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--residual-checkpoint", "residual.pt",
            "--experiment-checkpoint-out", "experiment.pt",
            "--novelty-baseline-in", "baseline.json",
            "--novelty-baseline-sha256", "a" * 64,
            "--require-track-completion",
        ]
    )

    assert args.require_track_completion is True


def test_play_parser_keeps_retention_probe_separate_from_strict_completion():
    with pytest.raises(SystemExit):
        build_play_parser().parse_args(
            [
                "--base-checkpoint", "base.pt",
                "--mode", "base",
                "--flat-retention-probe",
                "--require-track-completion",
            ]
        )


def test_play_parser_accepts_curriculum_worker_arguments():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt",
            "--terrain-stage", "step_2cm", "--terrain-seed", "101",
            "--protocol-sha256", "HASH", "--experiment-checkpoint-in", "in.pt",
            "--experiment-checkpoint-out", "out.pt", "--flat-replay", "flat.pt",
            "--flat-anchor-weight", "2.0", "--n-step", "3",
            "--target-q-min", "-1000", "--target-q-max", "500",
            "--critic-huber-delta", "1.0",
            "--deterministic-actor-weight", "0.25",
            "--residual-gain", "0.25",
        ]
    )
    assert args.terrain_stage == "step_2cm"
    assert args.terrain_seed == 101
    assert args.n_step == 3
    assert args.residual_gain == 0.25
    assert args.target_q_min == -1000.0
    assert args.target_q_max == 500.0
    assert args.critic_huber_delta == 1.0
    assert args.deterministic_actor_weight == 0.25


def test_play_parser_allows_online_updates_on_an_explicit_sealed_split():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--residual-checkpoint", "residual.pt",
            "--mode", "online",
            "--experiment-checkpoint-out", "out.pt",
            "--terrain-split", "sealed",
            "--terrain-seed", "1009",
        ]
    )

    assert args.terrain_split == "sealed"
    assert args.evaluation_only is False


def test_play_parser_keeps_evaluation_only_sealed_default():
    args = build_play_parser().parse_args(
        [
            "--base-checkpoint", "base.pt",
            "--mode", "base",
            "--evaluation-only",
            "--terrain-seed", "1009",
        ]
    )

    assert args.terrain_split == "sealed"


def test_play_parser_rejects_seed_outside_explicit_split():
    with pytest.raises(SystemExit):
        build_play_parser().parse_args(
            [
                "--base-checkpoint", "base.pt",
                "--mode", "base",
                "--terrain-split", "sealed",
                "--terrain-seed", "101",
            ]
        )


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


def test_new_objective_can_calibrate_a_clean_session_without_legacy_reward_baseline():
    arguments = ["--base-checkpoint", "base.pt", "--mode", "online",
                 "--experiment-checkpoint-out", "out.pt", "--require-track-completion"]
    with pytest.raises(SystemExit, match="novelty-baseline"):
        build_play_parser().parse_args(arguments)
    parsed = build_play_parser().parse_args([*arguments, "--task-objective", "finite-horizon-v1"])
    assert parsed.novelty_baseline_in is None
    assert parsed.residual_checkpoint is None
