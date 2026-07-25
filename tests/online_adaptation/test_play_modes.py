from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser


def test_play_modes_are_explicit_and_default_to_online():
    args = build_play_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--residual-checkpoint", "residual.pt"]
    )
    assert args.mode == "online"
    assert args.task == "HeldoutBoxes-Deeprobotics-Lite3-v0"


def test_base_mode_does_not_require_residual_checkpoint():
    args = build_play_parser().parse_args(["--base-checkpoint", "base.pt", "--mode", "base"])
    assert args.residual_checkpoint is None
