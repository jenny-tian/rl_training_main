from argparse import Namespace
from pathlib import Path

import pytest

from scripts.reinforcement_learning.online_adaptation.run_lora_ppo_sealed_test import (
    _seed_list,
    build_command,
    build_parser,
)
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol


def test_sealed_test_defaults_to_entire_sealed_split_and_rejects_development():
    assert _seed_list(None) == default_flat_online_protocol().sealed_seeds
    with pytest.raises(ValueError, match="only protocol sealed seeds"):
        _seed_list("109")
    with pytest.raises(ValueError, match="unique seeds"):
        _seed_list("1009,1009")


def test_sealed_command_is_frozen_and_uses_sealed_protocol(tmp_path):
    args = Namespace(
        base_checkpoint=str(tmp_path / "base.pt"),
        checkpoint=str(tmp_path / "candidate.pt"),
        simulation_seed=19,
        physical_gpu=1,
        max_steps=2000,
        height_scan_adapter=True,
        proprioceptive_context=False,
        single_gpu_renderer=False,
    )
    command = build_command(args, 1009, tmp_path / "seed-1009")

    assert command[command.index("--terrain-split") + 1] == "sealed"
    assert command[command.index("--terrain-seed") + 1] == "1009"
    assert command[command.index("--checkpoint-in") + 1] == str(Path(args.checkpoint).resolve())
    assert "--evaluation-only" in command
    assert "--height-scan-adapter" in command
    assert "--stop-after-termination" not in command


def test_sealed_test_requires_explicit_fixed_checkpoint_and_report_location():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--base-checkpoint", "base.pt"])
    args = parser.parse_args([
        "--base-checkpoint", "base.pt", "--checkpoint", "selected.pt", "--output-dir", "final-test",
    ])
    assert args.terrain_seeds is None
