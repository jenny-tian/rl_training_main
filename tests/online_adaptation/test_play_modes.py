from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser


def test_play_modes_are_explicit_and_default_to_online():
    args = build_play_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt",
         "--experiment-checkpoint-out", "experiment.pt"]
    )
    assert args.mode == "online"
    assert args.task == "FlatOnline-Deeprobotics-Lite3-v0"


def test_base_mode_does_not_require_residual_checkpoint():
    args = build_play_parser().parse_args(["--base-checkpoint", "base.pt", "--mode", "base"])
    assert args.residual_checkpoint is None


def test_forward_speed_is_configurable():
    args = build_play_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--mode", "base", "--forward-speed", "0.2"]
    )
    assert args.forward_speed == 0.2


def test_online_control_is_deterministic_unless_exploration_is_explicitly_enabled():
    deterministic = build_play_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt",
         "--experiment-checkpoint-out", "experiment.pt"]
    )
    exploratory = build_play_parser().parse_args(
        [
            "--base-checkpoint",
            "base.pt",
            "--residual-checkpoint",
            "residual.pt",
            "--experiment-checkpoint-out",
            "experiment.pt",
            "--stochastic-control",
        ]
    )
    assert not deterministic.stochastic_control
    assert exploratory.stochastic_control
