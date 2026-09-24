from pathlib import Path

from scripts.reinforcement_learning.online_adaptation.train_height_scan_teacher import (
    TEACHER_CONTRACT,
    build_command,
)
from rl_training.online_adaptation.terrain_protocol import (
    FORMAL_TRAIN_SEEDS,
    default_flat_online_protocol,
)


def test_teacher_command_is_train_only_height_scan_and_protocol_bound(tmp_path):
    from argparse import Namespace

    args = Namespace(
        isaaclab_sh="/opt/IsaacLab/isaaclab.sh",
        base_checkpoint="base.pt",
        terrain_stage="rough_4cm",
        simulation_seed=7,
        steps_per_seed=1000,
        evaluation_steps=4000,
        rollout_steps=256,
        output_dir=str(tmp_path),
        rank=2,
        device="cuda:0",
    )
    checkpoint = tmp_path / "previous" / "latest.pt"
    command = build_command(args, FORMAL_TRAIN_SEEDS[1], tmp_path / "seed-103", checkpoint)
    assert command[:3] == ["/opt/IsaacLab/isaaclab.sh", "-p", str(Path(__file__).parents[2] / "scripts/reinforcement_learning/online_adaptation/train_lora_ppo.py")]
    assert command[command.index("--terrain-split") + 1] == "train"
    assert command[command.index("--terrain-seed") + 1] == "103"
    assert command[command.index("--protocol-sha256") + 1] == default_flat_online_protocol().sha256()
    assert "--teacher-training" in command and "--height-scan-adapter" in command
    assert "--continue-after-termination" in command
    assert "--require-track-completion" not in command
    assert "--trial-deadline-steps" not in command
    assert command[command.index("--checkpoint-in") + 1] == str(checkpoint.resolve())
    assert TEACHER_CONTRACT == "height_scan_privileged_teacher_v2"


def test_frozen_probe_command_uses_full_completion_budget_and_no_updates(tmp_path):
    from argparse import Namespace

    args = Namespace(
        isaaclab_sh="/opt/IsaacLab/isaaclab.sh",
        base_checkpoint="base.pt",
        terrain_stage="rough_4cm",
        simulation_seed=7,
        steps_per_seed=20_000,
        evaluation_steps=4_000,
        rollout_steps=256,
        output_dir=str(tmp_path),
        rank=2,
        device="cuda:0",
    )
    checkpoint = tmp_path / "teacher.pt"
    command = build_command(args, 101, tmp_path / "probe", checkpoint, evaluation=True)
    assert command[command.index("--max-steps") + 1] == "4000"
    assert command[command.index("--trial-deadline-steps") + 1] == "4000"
    assert "--evaluation-only" in command
    assert "--continue-after-termination" not in command


def test_small_budget_requires_explicit_smoke_classification():
    import pytest
    from scripts.reinforcement_learning.online_adaptation.train_height_scan_teacher import main

    with pytest.raises(SystemExit, match="require --smoke"):
        main(["--base-checkpoint", "missing.pt", "--output-dir", "unused", "--steps-per-seed", "32"])
