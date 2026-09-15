from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import build_parser


def test_parser_has_planned_low_rank_ppo_defaults():
    args = build_parser().parse_args(["--base-checkpoint", "flat.pt"])
    assert args.rank == 2
    assert args.rollout_steps == 256
    assert args.action_std == 0.1
    assert args.target_kl == 0.01


def test_parser_accepts_explicit_app_and_learning_options():
    args = build_parser().parse_args(
        ["--base-checkpoint", "flat.pt", "--headless", "--device", "cuda:1", "--rank", "4"]
    )
    assert args.headless
    assert args.device == "cuda:1"
    assert args.rank == 4
