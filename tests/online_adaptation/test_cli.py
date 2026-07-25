from scripts.reinforcement_learning.online_adaptation.cli import build_qualify_parser, build_train_parser


def test_train_defaults_keep_heldout_task_out_of_pretraining():
    args = build_train_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "OnlineAdapt-Deeprobotics-Lite3-v0"
    assert "HeldoutBoxes" not in args.task
    assert args.min_replay_size == 256
    assert args.replay_capacity == 10_000


def test_qualification_defaults_are_source_only_and_strict():
    args = build_qualify_parser().parse_args(["--base-checkpoint", "base.pt"])
    assert args.task == "OnlineAdapt-Deeprobotics-Lite3-v0"
    assert args.seeds == [7, 11, 19, 23, 31]
    assert args.steps_per_seed == 1_000
    assert args.max_mean_velocity_error == 0.35
